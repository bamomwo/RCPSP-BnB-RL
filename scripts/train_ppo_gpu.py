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
import json
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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
from rcpsp_bb_rl.ml.rl import BranchingEnv  # noqa: E402
from rcpsp_bb_rl.ml.rl.tree_return import (  # noqa: E402
    compute_episode_advantages_decoupled,
    make_cost_reward_fn,
    make_bonus_reward_fn,
)
from rcpsp_bb_rl.bnb.branching_order import make_order_fn  # noqa: E402
from rcpsp_bb_rl.bnb.solver import BnBSolver  # noqa: E402


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
    "ent_coef": 0.01,
    "vf_coef": 0.5,
    "lr": 3e-4,
    "max_grad_norm": 0.5,
    "target_kl": 0.02,
    "time_limit_s": 60.0,
    # Reward
    "alpha": 0.01,
    "beta1": 1.0,
    "beta2": 1.0,
    # Eval
    "eval_every_steps": 20_000,
    "eval_root": None,
    "eval_pattern": "*.RCP",
    "eval_time_limit_s": 60.0,
    "eval_optimal_json": None,
    # Output
    "save_path": "models/policy_ppo.pt",
    "checkpoint_dir": "models/checkpoints",
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
    alpha = float(config["alpha"])
    beta1 = float(config["beta1"])
    beta2 = float(config["beta2"])
    ent_coef = float(config["ent_coef"])
    vf_coef = float(config["vf_coef"])
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

    def next_instance() -> Path:
        nonlocal inst_idx, inst_order
        if inst_idx >= len(inst_order):
            inst_order = list(range(len(instance_paths)))
            random.shuffle(inst_order)
            inst_idx = 0
        path = instance_paths[inst_order[inst_idx]]
        inst_idx += 1
        return path

    # Start first episode
    current_instance_path = next_instance()
    obs = env.reset(instance=load_instance(current_instance_path))
    episode_steps = 0

    t_start = time.perf_counter()
    print(f"\n{'='*80}")
    print(f"  PPO Training (GPU-batched update)")
    print(f"  total_steps={total_env_steps:,}  backup=tree(cost_gamma={tree_gamma_cost},bonus_gamma={tree_gamma_bonus})  train_instances={len(instance_paths)}  eval_instances={len(eval_paths)}")
    print(f"{'='*80}")
    print(f"[Episode 1] start  {current_instance_path.name}\n")

    # === MAIN TRAINING LOOP ===
    while global_step < total_env_steps:

        # ---- Collect ONE complete episode (same as train_ppo.py) ----
        ac.eval()
        buffer.clear()
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

        # ---- Subtree-return backup (same as train_ppo.py) ----
        cost_reward_fn = make_cost_reward_fn(alpha=alpha)
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
            f"rewards=(G_root:{ep_return:+.2f}, G_cost:{g_cost:+.2f}, "
            f"G_first_incum:{g_first:+.2f}, G_incum_impro:{g_improve:+.2f})  "
            f"elapsed={elapsed:.0f}s"
        )
        print()

        # ---- Advantage computation (same as train_ppo.py) ----
        ta = compute_episode_advantages_decoupled(
            tree=episode_tree,
            node_ids=buffer.node_ids,
            values=buffer.values,
            cost_reward_fn=cost_reward_fn,
            bonus_reward_fn=bonus_reward_fn,
            gamma_cost=tree_gamma_cost,
            gamma_bonus=tree_gamma_bonus,
            keep_open=tree_keep_open,
        )
        raw_returns = torch.tensor(ta.returns, dtype=torch.float32)
        valid_mask = torch.tensor(ta.valid, dtype=torch.bool)

        valid_idx = valid_mask.nonzero().squeeze(-1).tolist()
        n_valid = len(valid_idx)
        n_total = len(buffer)

        if n_valid == 0:
            print(f"[Update skipped] no closed subtrees this episode "
                  f"(transitions={n_total})\n")
            if global_step < total_env_steps:
                current_instance_path = next_instance()
                obs = env.reset(instance=load_instance(current_instance_path))
                print(f"[Episode {episode_count+1}] start  {current_instance_path.name}\n")
            episode_steps = 0
            continue

        indices = np.array(valid_idx)

        # ---- Value normalisation (same as train_ppo.py) ----
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

        total_pg_loss = total_vf_loss = total_ent = total_kl = 0.0
        n_kl_samples = 0
        update_count += 1
        early_stop = False

        for _ in range(ppo_epochs):
            if early_stop:
                break
            # Shuffle within buckets: split into minibatch-sized chunks, shuffle
            # the chunk order (not individual items) to keep similar-R together.
            n_chunks = max(T // mb_size, 1)
            chunk_order = np.arange(n_chunks)
            np.random.shuffle(chunk_order)

            for ci in chunk_order:
                start = ci * mb_size
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

                # Value loss
                vf_loss = nn.functional.mse_loss(values_b, mb_returns)

                # Entropy bonus
                entropy_loss = -mb_entropies_t.mean()

                loss = pg_loss + vf_coef * vf_loss + ent_coef * entropy_loss

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

                if target_kl is not None and approx_kl > float(target_kl):
                    early_stop = True
                    break

        n_updates = ppo_epochs * n_chunks
        mean_kl = total_kl / n_kl_samples if n_kl_samples > 0 else 0.0
        elapsed = time.perf_counter() - t_start
        print(
            f"[Update {update_count}] "
            f"steps={global_step}  "
            f"episodes={episode_count}  "
            f"valid={n_valid}/{n_total}  "
            f"pg={total_pg_loss/n_updates:+.4f}  "
            f"vf={total_vf_loss/n_updates:.4f}  "
            f"ev={explained_var:+.3f}  "
            f"ret_std={ret_std:.4f}  "
            f"ent={total_ent/n_updates:.4f}  "
            f"kl={mean_kl:.4f}  "
            f"elapsed={elapsed:.0f}s"
            f"{'  [KL stop]' if early_stop else ''}"
        )

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
            current_instance_path = next_instance()
            obs = env.reset(instance=load_instance(current_instance_path))
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


if __name__ == "__main__":
    main()
