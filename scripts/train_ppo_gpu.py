"""
GPU-optimized PPO fine-tuning for the RCPSP branching policy.

Same training semantics as train_ppo.py (same rewards, advantages, PPO math),
but the PPO update phase is BATCHED: observations are padded to a common
sequence length and processed in one forward pass per minibatch. This gives
5-20x speedup on the update phase when running on GPU.

Key differences from train_ppo.py:
  - PPO update uses model.forward_batch() instead of per-item forward()
  - Transitions are sorted by candidate-set size (R) before forming minibatches
    to minimize padding waste ("bucket batching")
  - Collection phase keeps model on GPU; each step moves one obs to device
  - Everything else (env interaction, subtree backup, advantages, eval) is identical
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from rcpsp_bb_rl.data.dataset import list_instance_paths  # noqa: E402
from rcpsp_bb_rl.data.parsing import load_instance  # noqa: E402
from rcpsp_bb_rl.ml.models import BranchingTransformer, load_policy_checkpoint, save_policy_checkpoint  # noqa: E402
from rcpsp_bb_rl.ml.il.featurize import global_feature_dim, candidate_feature_dim, critic_feature_dim  # noqa: E402
from rcpsp_bb_rl.ml.estimator import load_estimator_checkpoint, predict_difficulty  # noqa: E402
from rcpsp_bb_rl.ml.rl import BranchingEnv  # noqa: E402
from rcpsp_bb_rl.ml.rl.tree_return import (  # noqa: E402
    compute_episode_advantages_decoupled,
    make_cost_reward_fn,
    make_bonus_reward_fn,
)
from rcpsp_bb_rl.bnb.branching_order import make_order_fn  # noqa: E402
from rcpsp_bb_rl.bnb.solver import BnBSolver  # noqa: E402

# Type alias for reward functions
RewardFn = Callable[[Any], float]


# ---------------------------------------------------------------------------
# Episode record for multi-episode accumulation
# ---------------------------------------------------------------------------

@dataclass
class EpisodeRecord:
    """Tracks per-episode metadata for multi-episode PPO batching."""
    tree: Optional[Dict]
    cost_reward_fn: RewardFn
    bonus_reward_fn: RewardFn
    start_idx: int       # index into buffer where this episode starts
    end_idx: int         # index into buffer where this episode ends (exclusive)
    n_valid: int         # pre-counted valid transitions for this episode
    # Cached advantages to avoid recomputing subtree returns at update time
    returns: Optional[List[float]] = field(default=None)
    valid_flags: Optional[List[bool]] = field(default=None)


# ---------------------------------------------------------------------------
# Actor-Critic wrapper (same as train_ppo.py)
# ---------------------------------------------------------------------------

class ActorCritic(nn.Module):
    """Wraps BranchingTransformer for PPO (unbatched collection + batched update)."""

    def __init__(self, model: BranchingTransformer) -> None:
        super().__init__()
        self.model = model

    def forward(
        self,
        candidate_feats: torch.Tensor,
        global_feats: torch.Tensor,
        action_mask: Optional[torch.Tensor] = None,
        critic_feats: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns (logits [R], value scalar). Unbatched."""
        return self.model(candidate_feats, global_feats, action_mask, critic_feats)

    def get_action_and_value(
        self,
        obs: Dict[str, torch.Tensor],
        device: torch.device,
        action: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample or evaluate an action (unbatched, used during collection).
        Returns (action, log_prob, entropy, value).
        """
        cand = obs["candidate_feats"].to(device)
        glob = obs["global_feats"].to(device)
        mask = obs["action_mask"].to(device)
        critic = obs.get("critic_feats")
        if critic is not None:
            critic = critic.to(device)

        logits, value = self.forward(cand, glob, mask, critic)
        dist = Categorical(logits=logits)

        if action is None:
            action = dist.sample()

        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        return action, log_prob, entropy, value


# ---------------------------------------------------------------------------
# Batched PPO helpers
# ---------------------------------------------------------------------------

def batch_observations(
    obs_list: List[Dict[str, torch.Tensor]],
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, List[int]]:
    """
    Pad a list of variable-length observations into batched tensors.

    Returns
    -------
    cand_batch  : [B, R_max, Fc]  padded candidate features
    glob_batch  : [B, Fg]         global features
    mask_batch  : [B, R_max] bool action mask (False for infeasible AND padding)
    critic_batch: [B, Fk]         critic features
    pad_mask    : [B, R_max] bool True for real positions, False for padding
    seq_lens    : list of int, actual R per item
    """
    B = len(obs_list)
    seq_lens = [obs["candidate_feats"].shape[0] for obs in obs_list]
    R_max = max(seq_lens)
    Fc = obs_list[0]["candidate_feats"].shape[1]
    Fg = obs_list[0]["global_feats"].shape[0]
    Fk = obs_list[0]["critic_feats"].shape[0] if "critic_feats" in obs_list[0] else 0

    cand_batch = torch.zeros(B, R_max, Fc, device=device)
    glob_batch = torch.zeros(B, Fg, device=device)
    mask_batch = torch.zeros(B, R_max, dtype=torch.bool, device=device)
    pad_mask = torch.zeros(B, R_max, dtype=torch.bool, device=device)
    critic_batch = torch.zeros(B, Fk, device=device) if Fk > 0 else None

    for i, obs in enumerate(obs_list):
        R_i = seq_lens[i]
        cand_batch[i, :R_i] = obs["candidate_feats"]
        glob_batch[i] = obs["global_feats"]
        mask_batch[i, :R_i] = obs["action_mask"]
        pad_mask[i, :R_i] = True
        if critic_batch is not None and "critic_feats" in obs:
            critic_batch[i] = obs["critic_feats"]

    return cand_batch, glob_batch, mask_batch, critic_batch, pad_mask, seq_lens


def compute_log_probs_entropy(
    logits_batch: torch.Tensor,
    actions: torch.Tensor,
    seq_lens: List[int],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute per-item log_prob and entropy from batched logits.

    Loops over items (cheap — just softmax + indexing, no transformer).

    Parameters
    ----------
    logits_batch : [B, R_max] — masked logits from forward_batch
    actions      : [B] long   — chosen action indices
    seq_lens     : actual R per item (to slice valid logits)

    Returns
    -------
    log_probs : [B]
    entropies : [B]
    """
    B = logits_batch.shape[0]
    log_probs = torch.empty(B, device=logits_batch.device)
    entropies = torch.empty(B, device=logits_batch.device)

    for i in range(B):
        R_i = seq_lens[i]
        logits_i = logits_batch[i, :R_i]
        dist = Categorical(logits=logits_i)
        log_probs[i] = dist.log_prob(actions[i])
        entropies[i] = dist.entropy()

    return log_probs, entropies


# ---------------------------------------------------------------------------
# Rollout buffer (same as train_ppo.py)
# ---------------------------------------------------------------------------

class RolloutBuffer:
    """Stores transitions from one rollout horizon."""

    def __init__(self) -> None:
        self.obs: List[Dict[str, torch.Tensor]] = []
        self.actions: List[int] = []
        self.log_probs: List[float] = []
        self.values: List[float] = []
        self.dones: List[bool] = []
        self.terminateds: List[bool] = []
        self.node_ids: List[Optional[int]] = []
        self.parent_ids: List[Optional[int]] = []

    def add(
        self,
        obs: Dict[str, torch.Tensor],
        action: int,
        log_prob: float,
        value: float,
        done: bool,
        terminated: bool,
        node_id: Optional[int] = None,
        parent_id: Optional[int] = None,
    ) -> None:
        self.obs.append(obs)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.dones.append(done)
        self.terminateds.append(terminated)
        self.node_ids.append(node_id)
        self.parent_ids.append(parent_id)

    def __len__(self) -> int:
        return len(self.values)

    def clear(self) -> None:
        self.__init__()


class RunningMeanStd:
    """Running mean/variance for value normalisation (same as train_ppo.py)."""

    def __init__(self, eps: float = 1e-4) -> None:
        self.mean = 0.0
        self.var = 1.0
        self.count = eps

    def update(self, x: np.ndarray) -> None:
        if x.size == 0:
            return
        b_mean = float(x.mean())
        b_var = float(x.var())
        b_count = int(x.size)
        delta = b_mean - self.mean
        tot = self.count + b_count
        self.mean += delta * b_count / tot
        m_a = self.var * self.count
        m_b = b_var * b_count
        self.var = (m_a + m_b + delta * delta * self.count * b_count / tot) / tot
        self.count = tot

    @property
    def std(self) -> float:
        return float(self.var) ** 0.5


# ---------------------------------------------------------------------------
# Evaluation (same as train_ppo.py)
# ---------------------------------------------------------------------------

def evaluate(
    model: BranchingTransformer,
    instance_paths: List[Path],
    max_resources: int,
    time_limit_s: float,
    dominance: str,
    device: torch.device,
    optimal_makespans: Dict[str, int],
) -> Dict[str, float]:
    """Run the policy on eval instances. Returns solved_frac, mean_gap, mean_nodes."""
    model.eval()
    solved = 0
    gaps = []
    node_counts = []

    for path in instance_paths:
        key = path.stem.lower()
        opt = optimal_makespans.get(key)
        if opt is None:
            continue

        instance = load_instance(path)
        solver = BnBSolver(instance)
        order_fn = make_order_fn(
            "policy",
            instance=instance,
            model=model,
            max_resources=max_resources,
            device=device,
            predecessors=solver.predecessors,
        )
        result = solver.solve(
            order_ready_fn=order_fn,
            time_limit_s=time_limit_s,
            dominance=dominance,
        )
        node_counts.append(result.nodes_expanded)

        if result.best_makespan is not None:
            gap = (result.best_makespan - opt) / opt * 100.0
            gaps.append(gap)
            if result.best_makespan == opt:
                solved += 1

    n = len(gaps)
    return {
        "solved_frac": solved / n if n > 0 else 0.0,
        "mean_gap": float(np.mean(gaps)) if gaps else 0.0,
        "mean_nodes": float(np.mean(node_counts)) if node_counts else 0.0,
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="GPU-optimized PPO fine-tuning for the RCPSP branching policy.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--config", required=True, help="Path to JSON config file.")
    p.add_argument(
        "--log",
        action="store_true",
        help="Tee training output to a .txt file in the same directory as the saved model.",
    )
    return p.parse_args()


class _Tee:
    """Duplicate writes to several streams (console + log file)."""

    def __init__(self, *streams) -> None:
        self._streams = streams

    def write(self, data: str) -> int:
        for s in self._streams:
            s.write(data)
            s.flush()
        return len(data)

    def flush(self) -> None:
        for s in self._streams:
            s.flush()


def load_json(path: Path) -> Dict[str, Any]:
    with path.open() as f:
        return json.load(f)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


DEFAULT_CONFIG: Dict[str, Any] = {
    # Data
    "root": "data/train",
    "pattern": "*.RCP",
    "max_instances": None,
    "max_resources": 4,
    "dominance": "set_based",
    # Model
    "d_model": 64,
    "n_heads": 4,
    "n_layers": 2,
    "ffn_dim": 128,
    "dropout": 0.0,
    "bc_checkpoint": None,
    # PPO
    "total_env_steps": 1_000_000,
    "ppo_epochs": 4,
    "minibatches": 4,
    "clip_eps": 0.2,
    "tree_gamma": 1.0,
    "tree_gamma_cost": None,
    "tree_gamma_bonus": None,
    "tree_keep_open": False,
    "min_batch_size": 4096,
    "ent_coef_start": 0.01,
    "ent_coef_end": 0.001,
    "vf_coef": 0.5,
    "vf_loss_type": "huber",   # "mse" | "huber" — huber is robust to shallow-hard outliers
    "huber_delta": 1.0,        # error threshold (in normalized-return units) for linear regime
    "lr": 3e-4,
    "max_grad_norm": 0.5,
    "target_kl": 0.02,
    "time_limit_s": 60.0,
    # Reward
    "alpha": 0.01,          # static node-cost coef; used only when estimator_path is null
    "beta1": 1.0,
    "beta2": 1.0,
    # Dynamic reward scaling: when estimator_path is set, the node-cost coef is
    # computed per instance as alpha(I) = clip(c_target / N_hat(I), alpha_min,
    # alpha_max), where N_hat is the search-effort estimator's prediction. This
    # stabilises the cost channel's global scale across easy/hard instances and
    # keeps the beta/alpha break-even ratios comparable. Null -> static alpha.
    "estimator_path": None,
    "c_target": 1.0,
    "alpha_min": 1e-6,
    "alpha_max": 0.5,
    # Eval
    "eval_every_steps": 20_000,
    "eval_root": None,
    "eval_pattern": "*.RCP",
    "eval_time_limit_s": 60.0,
    "eval_optimal_json": None,
    # Output
    "save_path": "models/policy_ppo.pt",
    "checkpoint_dir": "models/checkpoints",
    "tensorboard": True,   # write scalar metrics to <save_path.parent>/tb for live monitoring
    "seed": 42,
    "device": "cpu",
}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    config = DEFAULT_CONFIG.copy()
    config.update(load_json(Path(args.config)))

    # --- Optional log file ---
    if args.log:
        save_path = Path(config["save_path"])
        save_path.parent.mkdir(parents=True, exist_ok=True)
        log_file_path = save_path.parent / (save_path.stem + "_train_log.txt")
        log_fh = open(log_file_path, "w")
        sys.stdout = _Tee(sys.__stdout__, log_fh)
        sys.stderr = _Tee(sys.__stderr__, log_fh)
        print(f"Logging training output to: {log_file_path}")

    set_seed(int(config["seed"]))
    device = torch.device(
        "cpu" if (config["device"] == "cuda" and not torch.cuda.is_available())
        else config["device"]
    )
    print(f"Device: {device}")

    # --- Instance paths ---
    instance_paths = list_instance_paths(config["root"], patterns=(config["pattern"],))
    if config["max_instances"] is not None:
        instance_paths = instance_paths[: int(config["max_instances"])]
    if not instance_paths:
        raise FileNotFoundError(f"No instances found under {config['root']}")
    print(f"Training instances: {len(instance_paths)}")

    # --- Eval instances ---
    eval_paths: List[Path] = []
    if config["eval_root"] is not None:
        eval_paths = list_instance_paths(
            config["eval_root"], patterns=(config["eval_pattern"],)
        )
    print(f"Eval instances: {len(eval_paths)}")

    # --- Optimal makespans for eval ---
    optimal_makespans: Optional[Dict[str, int]] = None
    if config.get("eval_optimal_json"):
        raw = json.loads(Path(config["eval_optimal_json"]).read_text())
        instances = raw.get("instances", {})
        optimal_makespans = {
            Path(k).stem.lower(): int(v["makespan"])
            for k, v in instances.items()
            if isinstance(v, dict) and "makespan" in v
        }

    # --- Model ---
    max_resources = int(config["max_resources"])
    global_dim = global_feature_dim(max_resources)
    candidate_dim = candidate_feature_dim(max_resources)
    critic_dim = critic_feature_dim()

    if config["bc_checkpoint"] is not None:
        print(f"Loading BC checkpoint: {config['bc_checkpoint']}")
        base_model = load_policy_checkpoint(
            config["bc_checkpoint"], device=device, dropout=float(config["dropout"]),
            critic_feature_dim=critic_dim,
        )
    else:
        print("No BC checkpoint — initialising from scratch.")
        base_model = BranchingTransformer(
            global_dim=global_dim,
            candidate_dim=candidate_dim,
            d_model=int(config["d_model"]),
            n_heads=int(config["n_heads"]),
            n_layers=int(config["n_layers"]),
            ffn_dim=int(config["ffn_dim"]),
            dropout=float(config["dropout"]),
            critic_feature_dim=critic_dim,
        )

    ac = ActorCritic(base_model).to(device)
    optimizer = optim.AdamW(ac.parameters(), lr=float(config["lr"]))
    print(f"Model params: {sum(p.numel() for p in ac.parameters()):,}")

    # --- Output paths ---
    save_path = Path(config["save_path"])
    save_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = Path(config["checkpoint_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_model_path = save_path.parent / (save_path.stem + "_best.pt")
    eval_log_path = save_path.parent / (save_path.stem + "_eval_log.json")

    # --- TensorBoard writer ---
    # Scalar metrics are tee'd here (in addition to stdout) so a run can be
    # monitored live. Logs land in <save_path.parent>/tb; point tensorboard at
    # the parent dir to compare runs on shared axes. x-axis is global_step.
    writer = None
    if bool(config.get("tensorboard", True)):
        from torch.utils.tensorboard import SummaryWriter  # local import: viewing needs `pip install tensorboard`
        tb_dir = save_path.parent / "tb"
        tb_dir.mkdir(parents=True, exist_ok=True)
        writer = SummaryWriter(log_dir=str(tb_dir))
        print(f"TensorBoard logging -> {tb_dir}")

    # --- Per-minibatch KL diagnostics (CSV) ---
    # One row per minibatch step across the whole run, so the KL-stop pattern can
    # be analysed offline (e.g. group by candidate-count range, compare trigger
    # vs non-trigger buckets). Kept out of stdout to avoid drowning the console.
    # x-axis for pattern-finding is the monotonic `mb_step` column.
    kl_diag_path = save_path.parent / (save_path.stem + "_kl_minibatches.csv")
    kl_diag_file = open(kl_diag_path, "w", newline="")
    kl_diag_writer = csv.writer(kl_diag_file)
    kl_diag_writer.writerow([
        "mb_step",          # global monotonic minibatch counter (x-axis)
        "update",           # PPO update index this minibatch belongs to
        "global_step",      # env steps at this update (for aligning with TB)
        "epoch",            # 1-based PPO epoch within the update
        "minibatch",        # 1-based minibatch within the epoch
        "batch_size",       # number of transitions in this minibatch
        "cand_min",         # candidate-set size (R) stats over the minibatch
        "cand_mean",
        "cand_median",
        "cand_max",
        "entropy_mean",     # mean policy entropy (nats)
        "entropy_norm_mean",  # mean entropy / log(R): 0=deterministic, 1=uniform
        "pct_deterministic",  # fraction of states with max softmax prob >= 0.95
        "approx_kl",        # this minibatch's approximate KL
        "is_trigger",       # 1 if this minibatch tripped the KL early-stop
    ])
    kl_diag_mb_step = 0  # monotonic counter across the whole run
    print(f"KL minibatch diagnostics -> {kl_diag_path}")

    # Threshold above which a state's policy counts as "nearly deterministic".
    det_prob_threshold = float(config.get("kl_diag_det_threshold", 0.95))

    # --- Training hyperparams ---
    total_env_steps = int(config["total_env_steps"])
    ppo_epochs = int(config["ppo_epochs"])
    minibatches = int(config["minibatches"])
    clip_eps = float(config["clip_eps"])
    tree_gamma = float(config["tree_gamma"])
    tree_gamma_cost = float(
        config["tree_gamma_cost"] if config["tree_gamma_cost"] is not None else tree_gamma
    )
    tree_gamma_bonus = float(
        config["tree_gamma_bonus"] if config["tree_gamma_bonus"] is not None else tree_gamma
    )
    tree_keep_open = bool(config["tree_keep_open"])
    min_batch_size = int(config["min_batch_size"])
    alpha = float(config["alpha"])
    beta1 = float(config["beta1"])
    beta2 = float(config["beta2"])

    # --- Dynamic node-cost scaling (estimator-driven alpha) ---
    # When an estimator is provided, alpha is computed per instance so the cost
    # channel's global scale is comparable across easy and hard instances. The
    # loop owns this (the env stays a pure B&B driver): alpha is derived once per
    # episode from the already-loaded instance object at the reward-build site.
    estimator = None
    estimator_scaler = None
    c_target = float(config["c_target"])
    alpha_min = float(config["alpha_min"])
    alpha_max = float(config["alpha_max"])
    estimator_path = config.get("estimator_path")
    if estimator_path:
        estimator, estimator_scaler, _est_tau = load_estimator_checkpoint(
            estimator_path, device=device
        )
    # The per-episode alpha = clip(c_target / N_est, alpha_min, alpha_max) is
    # computed inline at the reward-build site (one estimator pass per episode).
    # Entropy coefficient linear decay (with floor): coef goes from
    # ent_coef_start down to ent_coef_end over training.
    ent_coef_start = float(config["ent_coef_start"])
    ent_coef_end = float(config["ent_coef_end"])
    vf_coef = float(config["vf_coef"])
    vf_loss_type = str(config["vf_loss_type"]).strip().lower()
    if vf_loss_type not in {"mse", "huber"}:
        raise ValueError("vf_loss_type must be 'mse' or 'huber'.")
    huber_delta = float(config["huber_delta"])
    max_grad_norm = float(config["max_grad_norm"])
    target_kl = config.get("target_kl")
    eval_every = int(config["eval_every_steps"])
    time_limit_s = float(config["time_limit_s"])
    dominance = str(config["dominance"])

    # --- Environment and state ---
    env = BranchingEnv(
        instance_source=instance_paths[0],
        max_resources=max_resources,
        time_limit_s=time_limit_s,
        dominance=dominance,
    )

    buffer = RolloutBuffer()
    global_step = 0
    update_count = 0
    episode_count = 0
    next_eval_step = eval_every
    best_mean_gap = float("inf")
    eval_log: List[Dict] = []
    last_eval_step = -1
    ret_rms = RunningMeanStd()

    inst_order = list(range(len(instance_paths)))
    random.shuffle(inst_order)
    inst_idx = 0

    def next_instance() -> Tuple[Path, Any]:
        """Return (path, loaded_instance). Loading here means one load per episode,
        and the instance object is reused for both env.reset and alpha_for()."""
        nonlocal inst_idx, inst_order
        if inst_idx >= len(inst_order):
            inst_order = list(range(len(instance_paths)))
            random.shuffle(inst_order)
            inst_idx = 0
        path = instance_paths[inst_order[inst_idx]]
        inst_idx += 1
        return path, load_instance(path)

    # Start first episode
    current_instance_path, current_instance = next_instance()
    obs = env.reset(instance=current_instance)
    episode_steps = 0

    t_start = time.perf_counter()
    print(f"\n{'='*80}")
    print(f"  PPO Training (GPU-batched update, min_batch_size={min_batch_size})")
    print(f"  total_steps={total_env_steps:,}  backup=tree(cost_gamma={tree_gamma_cost},bonus_gamma={tree_gamma_bonus})  train_instances={len(instance_paths)}  eval_instances={len(eval_paths)}")
    print(f"  clip_eps={clip_eps}  ent_coef={ent_coef_start}->{ent_coef_end} (linear decay)")
    vf_desc = f"huber(delta={huber_delta})" if vf_loss_type == "huber" else "mse"
    print(f"  vf_loss={vf_desc}  vf_coef={vf_coef}")
    if estimator is not None:
        print(f"  reward_scale=DYNAMIC  alpha=clip(c_target/N_est, {alpha_min:g}, {alpha_max:g})  "
              f"c_target={c_target}  beta1={beta1}  beta2={beta2}")
        print(f"                        estimator={estimator_path}")
    else:
        print(f"  reward_scale=STATIC   alpha={alpha}  beta1={beta1}  beta2={beta2}")
    print(f"{'='*80}")
    print(f"[Episode 1] start  {current_instance_path.name}\n")

    # === MAIN TRAINING LOOP ===
    # Multi-episode accumulation: collect episodes until we have enough valid
    # transitions (>= min_batch_size) before performing a PPO update.
    episode_records: List[EpisodeRecord] = []
    accumulated_valid = 0

    while global_step < total_env_steps:

        # ---- Collect ONE complete episode ----
        ac.eval()
        ep_start_idx = len(buffer)
        episode_tree = None

        while True:
            with torch.no_grad():
                action_t, log_prob_t, _, value_t = ac.get_action_and_value(obs, device)

            action = int(action_t.item())
            step_out = env.step(action)

            buffer.add(
                obs=obs,
                action=action,
                log_prob=log_prob_t.item(),
                value=value_t.item(),
                done=step_out.done,
                terminated=bool(step_out.info.get("terminated", False)),
                node_id=step_out.info.get("node_id"),
                parent_id=step_out.info.get("parent_id"),
            )

            global_step += 1
            episode_steps += 1

            if step_out.done:
                episode_tree = env.search_tree()
                stats = env.episode_stats
                episode_count += 1
                break
            else:
                obs = step_out.observation

        # ---- Subtree-return backup ----
        # Node-cost coef for THIS episode: dynamically scaled from the instance's
        # estimated difficulty when an estimator is set, else the static alpha.
        # One estimator forward pass per episode; N_est kept for logging.
        if estimator is not None:
            episode_n_est = predict_difficulty(
                estimator, estimator_scaler, current_instance, device=device
            )
            episode_alpha = float(np.clip(c_target / episode_n_est, alpha_min, alpha_max))
        else:
            episode_n_est = None
            episode_alpha = alpha
        cost_reward_fn = make_cost_reward_fn(alpha=episode_alpha)
        bonus_reward_fn = make_bonus_reward_fn(
            episode_tree,
            beta1=beta1,
            beta2=beta2,
            root_lb=episode_tree.get("root_lb") if episode_tree else None,
        )

        if episode_tree is not None:
            tree_nodes = episode_tree.get("nodes", [])
            root_lb = episode_tree.get("root_lb")
            first_fn = make_bonus_reward_fn(
                episode_tree, beta1=beta1, beta2=0.0, root_lb=root_lb
            )
            improve_fn = make_bonus_reward_fn(
                episode_tree, beta1=0.0, beta2=beta2, root_lb=root_lb
            )
            g_cost = sum(cost_reward_fn(n) for n in tree_nodes)
            g_first = sum(first_fn(n) for n in tree_nodes)
            g_improve = sum(improve_fn(n) for n in tree_nodes)
            ep_return = g_cost + g_first + g_improve
        else:
            g_cost = g_first = g_improve = ep_return = 0.0

        elapsed = time.perf_counter() - t_start
        print(
            f"[Episode {episode_count}] Done  "
            f"Instance={current_instance_path.name}  "
            f"Reason={stats.done_reason}  "
            f"Steps={episode_steps}  "
            f"Nodes={stats.nodes_expanded}  "
            f"Best_Ms={stats.best_makespan}  "
            f"Inc_Improves={stats.incumbent_improvements}  "
            + (f"N_est={episode_n_est:.0f}  alpha={episode_alpha:.2e}  "
               if episode_n_est is not None else f"alpha={episode_alpha:.2e}  ")
            + f"rewards=(G_root:{ep_return:+.2f}, G_cost:{g_cost:+.2f}, "
            f"G_first_incum:{g_first:+.2f}, G_incum_impro:{g_improve:+.2f})  "
            f"elapsed={elapsed:.0f}s"
        )
        print()

        if writer is not None:
            writer.add_scalar("episode/return", ep_return, global_step)
            writer.add_scalar("episode/g_cost", g_cost, global_step)
            writer.add_scalar("episode/g_first_incumbent", g_first, global_step)
            writer.add_scalar("episode/g_improvement", g_improve, global_step)
            writer.add_scalar("episode/nodes_expanded", stats.nodes_expanded, global_step)
            writer.add_scalar("episode/incumbent_improvements", stats.incumbent_improvements, global_step)
            writer.add_scalar("episode/steps", episode_steps, global_step)
            if stats.best_makespan is not None:
                writer.add_scalar("episode/best_makespan", stats.best_makespan, global_step)
            writer.add_scalar("episode/alpha", episode_alpha, global_step)
            if episode_n_est is not None:
                writer.add_scalar("episode/N_est", episode_n_est, global_step)

        # ---- Compute advantages for this episode and cache them ----
        ep_end_idx = len(buffer)
        ep_node_ids = buffer.node_ids[ep_start_idx:ep_end_idx]
        ep_values = buffer.values[ep_start_idx:ep_end_idx]

        ta = compute_episode_advantages_decoupled(
            tree=episode_tree,
            node_ids=ep_node_ids,
            values=ep_values,
            cost_reward_fn=cost_reward_fn,
            bonus_reward_fn=bonus_reward_fn,
            gamma_cost=tree_gamma_cost,
            gamma_bonus=tree_gamma_bonus,
            keep_open=tree_keep_open,
        )
        n_valid_ep = sum(ta.valid)

        if n_valid_ep == 0:
            # No usable transitions from this episode — remove from buffer
            del buffer.obs[ep_start_idx:]
            del buffer.actions[ep_start_idx:]
            del buffer.log_probs[ep_start_idx:]
            del buffer.values[ep_start_idx:]
            del buffer.dones[ep_start_idx:]
            del buffer.terminateds[ep_start_idx:]
            del buffer.node_ids[ep_start_idx:]
            del buffer.parent_ids[ep_start_idx:]
            print(f"[Accumulate] no closed subtrees — skipping episode "
                  f"(accumulated={accumulated_valid})\n")
            if global_step < total_env_steps:
                current_instance_path, current_instance = next_instance()
                obs = env.reset(instance=current_instance)
                print(f"[Episode {episode_count+1}] start  {current_instance_path.name}\n")
            episode_steps = 0
            continue

        # Cache the per-episode tree advantages (avoids recomputing at update time)
        episode_records.append(EpisodeRecord(
            tree=episode_tree,
            cost_reward_fn=cost_reward_fn,
            bonus_reward_fn=bonus_reward_fn,
            start_idx=ep_start_idx,
            end_idx=ep_end_idx,
            n_valid=n_valid_ep,
            returns=ta.returns,
            valid_flags=ta.valid,
        ))
        accumulated_valid += n_valid_ep

        # ---- Check if we have enough transitions for a PPO update ----
        if accumulated_valid < min_batch_size:
            print(f"[Accumulate] valid={n_valid_ep}  accumulated={accumulated_valid}/{min_batch_size} — collecting more\n")
            if global_step < total_env_steps:
                current_instance_path, current_instance = next_instance()
                obs = env.reset(instance=current_instance)
                print(f"[Episode {episode_count+1}] start  {current_instance_path.name}\n")
            episode_steps = 0
            continue

        # ==================================================================
        # PPO UPDATE — we have accumulated >= min_batch_size valid transitions
        # ==================================================================

        # ---- Assemble combined returns and valid mask from all episodes ----
        all_returns: List[float] = []
        all_valid: List[bool] = []
        for rec in episode_records:
            all_returns.extend(rec.returns)
            all_valid.extend(rec.valid_flags)

        raw_returns = torch.tensor(all_returns, dtype=torch.float32)
        valid_mask = torch.tensor(all_valid, dtype=torch.bool)

        valid_idx = valid_mask.nonzero().squeeze(-1).tolist()
        n_valid = len(valid_idx)
        n_total = len(buffer)

        indices = np.array(valid_idx)

        # ---- Value normalisation ----
        raw_valid = raw_returns[indices].numpy().astype(np.float64)
        ret_rms.update(raw_valid)
        std = ret_rms.std + 1e-8
        returns = (raw_returns - ret_rms.mean) / std
        values_norm = torch.tensor([buffer.values[i] for i in valid_idx])
        returns_valid = returns[indices]
        advantages = returns.clone()
        advantages[indices] = returns_valid - values_norm

        var_returns = returns_valid.var()
        explained_var = (
            float("nan") if var_returns.item() == 0.0
            else (1.0 - (returns_valid - values_norm).var() / var_returns).item()
        )
        ret_std = returns_valid.std().item()

        adv_valid = advantages[indices]
        advantages = (advantages - adv_valid.mean()) / (adv_valid.std() + 1e-8)

        # ---- BATCHED PPO UPDATE (the GPU-optimized part) ----
        # Sort valid indices by candidate-set size (R) for bucket batching.
        # Transitions with similar R end up in the same minibatch, minimizing
        # padding waste.
        seq_lens_all = [buffer.obs[i]["candidate_feats"].shape[0] for i in valid_idx]
        sorted_order = np.argsort(seq_lens_all)
        sorted_indices = indices[sorted_order]

        ac.train()
        T = n_valid
        mb_size = max(T // minibatches, 1)

        # Linear entropy-coefficient decay (with floor) based on training
        # progress. progress in [0, 1] -> coef from ent_coef_start to ent_coef_end.
        progress = min(global_step / total_env_steps, 1.0)
        ent_coef_now = ent_coef_start + (ent_coef_end - ent_coef_start) * progress

        total_pg_loss = total_vf_loss = total_ent = total_kl = 0.0
        n_kl_samples = 0
        update_count += 1
        early_stop = False

        # ---- KL-stop diagnostics (logging only, no effect on the update) ----
        # Integer division leaves a remainder tail, so each epoch runs
        # ceil(T / mb_size) chunks, not the nominal `minibatches`. Compute the
        # REAL planned count so completed/planned is honest.
        chunks_per_epoch = len(range(0, T, mb_size))
        planned_steps = chunks_per_epoch * ppo_epochs
        max_kl = 0.0            # largest per-minibatch KL this update
        trigger_kl = None       # KL of the minibatch that crossed target_kl
        stop_epoch = None       # 1-based epoch the stop fired in
        stop_minibatch = None   # 1-based minibatch-within-epoch the stop fired in

        for epoch_i in range(ppo_epochs):
            if early_stop:
                break
            # Shuffle within buckets: generate start indices that cover ALL
            # transitions (including the remainder tail), then shuffle.
            chunk_starts = list(range(0, T, mb_size))
            np.random.shuffle(chunk_starts)

            for mb_i, start in enumerate(chunk_starts):
                mb_idx = sorted_indices[start: start + mb_size]

                mb_log_probs_old = torch.tensor(
                    [buffer.log_probs[i] for i in mb_idx], dtype=torch.float32, device=device
                )
                mb_actions = torch.tensor(
                    [buffer.actions[i] for i in mb_idx], dtype=torch.long, device=device
                )
                mb_advantages = advantages[mb_idx].to(device)
                mb_returns = returns[mb_idx].to(device)

                # Batched forward pass
                mb_obs_list = [buffer.obs[i] for i in mb_idx]
                cand_b, glob_b, mask_b, critic_b, pad_b, seq_lens = batch_observations(
                    mb_obs_list, device
                )

                logits_b, values_b = ac.model.forward_batch(
                    cand_b, glob_b, mask_b, critic_b, pad_b
                )

                # Per-item log_prob and entropy (loop, but cheap)
                mb_log_probs_new, mb_entropies_t = compute_log_probs_entropy(
                    logits_b, mb_actions, seq_lens
                )

                # Policy loss (clipped surrogate)
                ratio = torch.exp(mb_log_probs_new - mb_log_probs_old)
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss — Huber (robust to shallow-hard outliers) or MSE.
                # Returns are normalized to ~unit std, so huber_delta is in
                # standard-deviation units: errors within delta are quadratic
                # (precise fit), beyond delta grow linearly (outlier-robust).
                if vf_loss_type == "huber":
                    vf_loss = nn.functional.huber_loss(values_b, mb_returns, delta=huber_delta)
                else:
                    vf_loss = nn.functional.mse_loss(values_b, mb_returns)

                # Entropy bonus
                entropy_loss = -mb_entropies_t.mean()

                loss = pg_loss + vf_coef * vf_loss + ent_coef_now * entropy_loss

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(ac.parameters(), max_grad_norm)
                optimizer.step()

                total_pg_loss += pg_loss.item()
                total_vf_loss += vf_loss.item()
                total_ent += (-entropy_loss.item())

                with torch.no_grad():
                    approx_kl = ((ratio - 1) - torch.log(ratio)).mean().item()
                total_kl += approx_kl
                n_kl_samples += 1
                if approx_kl > max_kl:
                    max_kl = approx_kl

                # Does THIS minibatch trip the early-stop? Decide first so the
                # diagnostics row can flag the trigger before we break out.
                is_trigger = target_kl is not None and approx_kl > float(target_kl)

                # ---- Per-minibatch diagnostics row (logging only) ----
                # Candidate-set size (R) profile of this bucket + policy
                # concentration, so stops can be attributed to an R range offline.
                with torch.no_grad():
                    seq_arr = np.asarray(seq_lens, dtype=np.float64)
                    # Fraction of states whose policy is nearly deterministic:
                    # max softmax prob over each state's valid candidates >= thresh.
                    det_count = 0
                    ent_norm_sum = 0.0
                    for row_i, R_i in enumerate(seq_lens):
                        logits_i = logits_b[row_i, :R_i]
                        probs_i = torch.softmax(logits_i, dim=0)
                        if float(probs_i.max().item()) >= det_prob_threshold:
                            det_count += 1
                        # Normalized entropy: raw / log(R). R==1 -> define as 0
                        # (a singleton state is trivially deterministic).
                        if R_i > 1:
                            ent_norm_sum += float(mb_entropies_t[row_i].item()) / float(np.log(R_i))
                    n_states = len(seq_lens)
                    kl_diag_writer.writerow([
                        kl_diag_mb_step,
                        update_count,
                        global_step,
                        epoch_i + 1,
                        mb_i + 1,
                        n_states,
                        int(seq_arr.min()) if n_states else 0,
                        f"{seq_arr.mean():.3f}" if n_states else "0",
                        f"{np.median(seq_arr):.1f}" if n_states else "0",
                        int(seq_arr.max()) if n_states else 0,
                        f"{float(mb_entropies_t.mean().item()):.5f}",
                        f"{(ent_norm_sum / n_states):.5f}" if n_states else "0",
                        f"{(det_count / n_states):.4f}" if n_states else "0",
                        f"{approx_kl:.6f}",
                        1 if is_trigger else 0,
                    ])
                kl_diag_mb_step += 1

                if is_trigger:
                    # Record WHERE and WHAT tripped the stop before breaking. The
                    # triggering minibatch's KL is the real signal; mean_kl below
                    # dilutes it across all completed steps and hides it.
                    early_stop = True
                    trigger_kl = approx_kl
                    stop_epoch = epoch_i + 1
                    stop_minibatch = mb_i + 1
                    break

        n_updates = max(n_kl_samples, 1)  # actual number of minibatch steps taken
        completed_steps = n_kl_samples    # minibatch steps actually run this update
        mean_kl = total_kl / n_kl_samples if n_kl_samples > 0 else 0.0
        elapsed = time.perf_counter() - t_start
        print(
            f"[Update {update_count}] "
            f"steps={global_step}  "
            f"episodes_in_batch={len(episode_records)}  "
            f"valid={n_valid}/{n_total}  "
            f"pg={total_pg_loss/n_updates:+.4f}  "
            f"vf={total_vf_loss/n_updates:.4f}  "
            f"ev={explained_var:+.3f}  "
            f"ret_std={ret_std:.4f}  "
            f"ent={total_ent/n_updates:.4f}  "
            f"ent_coef={ent_coef_now:.4f}  "
            f"kl={mean_kl:.4f}  "
            f"max_kl={max_kl:.4f}  "
            f"steps={completed_steps}/{planned_steps}  "
            f"elapsed={elapsed:.0f}s"
            + (
                f"  [KL stop @ epoch {stop_epoch}/{ppo_epochs} "
                f"mb {stop_minibatch}/{chunks_per_epoch} trigger_kl={trigger_kl:.4f}]"
                if early_stop else ""
            )
        )

        # Flush the diagnostics CSV once per update so a Ctrl-C'd run still
        # leaves complete, inspectable rows on disk.
        kl_diag_file.flush()

        if writer is not None:
            writer.add_scalar("train/pg_loss", total_pg_loss / n_updates, global_step)
            writer.add_scalar("train/vf_loss", total_vf_loss / n_updates, global_step)
            writer.add_scalar("train/entropy", total_ent / n_updates, global_step)
            writer.add_scalar("train/ent_coef", ent_coef_now, global_step)
            writer.add_scalar("train/approx_kl", mean_kl, global_step)
            # KL-stop diagnostics: max_kl exposes the spike the mean hides;
            # completed/planned and the fraction show how much of each update
            # survives; kl_stopped is a 0/1 rate. On a stop, trigger_kl and the
            # stop position pinpoint the offending minibatch.
            writer.add_scalar("train/max_kl", max_kl, global_step)
            writer.add_scalar("train/completed_steps", completed_steps, global_step)
            writer.add_scalar("train/planned_steps", planned_steps, global_step)
            writer.add_scalar(
                "train/completed_fraction",
                completed_steps / max(planned_steps, 1),
                global_step,
            )
            writer.add_scalar("train/kl_stopped", 1.0 if early_stop else 0.0, global_step)
            if early_stop:
                writer.add_scalar("train/trigger_kl", trigger_kl, global_step)
                writer.add_scalar("train/stop_epoch", stop_epoch, global_step)
                writer.add_scalar("train/stop_minibatch", stop_minibatch, global_step)
            if not (explained_var != explained_var):  # skip NaN
                writer.add_scalar("train/explained_variance", explained_var, global_step)
            writer.add_scalar("train/return_std", ret_std, global_step)
            writer.add_scalar("train/valid_fraction", n_valid / max(n_total, 1), global_step)

        # ---- Clear accumulation state for next cycle ----
        buffer.clear()
        episode_records.clear()
        accumulated_valid = 0

        # ---- Periodic evaluation ----
        if eval_paths and optimal_makespans and global_step >= next_eval_step:
            next_eval_step += eval_every
            last_eval_step = global_step
            ac.eval()
            metrics = evaluate(
                model=ac.model,
                instance_paths=eval_paths,
                max_resources=max_resources,
                time_limit_s=float(config["eval_time_limit_s"]),
                dominance=dominance,
                device=device,
                optimal_makespans=optimal_makespans,
            )

            is_best = metrics["mean_gap"] < best_mean_gap
            best_tag = "  [best]" if is_best else ""
            sep = "-" * 60
            print(f"\n{sep}")
            print(
                f"[Checkpoint] "
                f"steps={global_step}  "
                f"solved={metrics['solved_frac']*100:.1f}%  "
                f"gap={metrics['mean_gap']:.2f}%  "
                f"nodes={metrics['mean_nodes']:.0f}"
                f"{best_tag}"
            )

            log_entry = {"step": global_step, **metrics}
            eval_log.append(log_entry)
            eval_log_path.write_text(json.dumps(eval_log, indent=2))

            if writer is not None:
                writer.add_scalar("eval/solved_frac", metrics["solved_frac"], global_step)
                writer.add_scalar("eval/mean_gap", metrics["mean_gap"], global_step)
                writer.add_scalar("eval/mean_nodes", metrics["mean_nodes"], global_step)

            ckpt_path = checkpoint_dir / f"policy_ppo_step{global_step}.pt"
            save_policy_checkpoint(ac.model, str(ckpt_path), extra={"train_config": config, "eval_metrics": metrics, "value_norm": {"mean": ret_rms.mean, "std": ret_rms.std}})
            print(f"[Checkpoint] saved  → {ckpt_path}")

            if is_best:
                best_mean_gap = metrics["mean_gap"]
                save_policy_checkpoint(ac.model, str(best_model_path), extra={"train_config": config, "eval_metrics": metrics, "step": global_step, "value_norm": {"mean": ret_rms.mean, "std": ret_rms.std}})
                print(f"[Checkpoint] best   → {best_model_path}  (gap={best_mean_gap:.2f}%)")
            print(f"{sep}\n")

        # ---- Reset for the next episode ----
        if global_step < total_env_steps:
            current_instance_path, current_instance = next_instance()
            obs = env.reset(instance=current_instance)
            print(f"[Episode {episode_count+1}] start  {current_instance_path.name}\n")
        episode_steps = 0

    # ---- Final evaluation ----
    elapsed = time.perf_counter() - t_start
    if eval_paths and optimal_makespans and global_step != last_eval_step:
        ac.eval()
        metrics = evaluate(
            model=ac.model,
            instance_paths=eval_paths,
            max_resources=max_resources,
            time_limit_s=float(config["eval_time_limit_s"]),
            dominance=dominance,
            device=device,
            optimal_makespans=optimal_makespans,
        )
        is_best = metrics["mean_gap"] < best_mean_gap
        best_tag = "  [best]" if is_best else ""
        sep = "-" * 60
        print(f"\n{sep}")
        print(
            f"[Final Eval] "
            f"steps={global_step}  "
            f"solved={metrics['solved_frac']*100:.1f}%  "
            f"gap={metrics['mean_gap']:.2f}%  "
            f"nodes={metrics['mean_nodes']:.0f}"
            f"{best_tag}"
        )
        log_entry = {"step": global_step, "final": True, **metrics}
        eval_log.append(log_entry)
        eval_log_path.write_text(json.dumps(eval_log, indent=2))

        if writer is not None:
            writer.add_scalar("eval/solved_frac", metrics["solved_frac"], global_step)
            writer.add_scalar("eval/mean_gap", metrics["mean_gap"], global_step)
            writer.add_scalar("eval/mean_nodes", metrics["mean_nodes"], global_step)

        if is_best:
            best_mean_gap = metrics["mean_gap"]
            save_policy_checkpoint(ac.model, str(best_model_path), extra={"train_config": config, "eval_metrics": metrics, "step": global_step, "value_norm": {"mean": ret_rms.mean, "std": ret_rms.std}})
            print(f"[Final Eval] best   → {best_model_path}  (gap={best_mean_gap:.2f}%)")
        print(f"{sep}")

    # ---- Training summary ----
    print(f"\n{'='*80}")
    print(f"  Training complete")
    print(f"  steps={global_step:,}  episodes={episode_count:,}  updates={update_count:,}  elapsed={elapsed:.0f}s")
    if eval_log:
        best_entry = min(eval_log, key=lambda e: e["mean_gap"])
        print(f"  best gap     : {best_entry['mean_gap']:.2f}% at step {best_entry['step']:,}")
        print(f"  best solved  : {best_entry['solved_frac']*100:.1f}% at step {best_entry['step']:,}")
        print(f"  best model   → {best_model_path}")
    print(f"  eval log     → {eval_log_path}")
    print(f"{'='*80}")

    if writer is not None:
        writer.flush()
        writer.close()

    kl_diag_file.close()


if __name__ == "__main__":
    main()
