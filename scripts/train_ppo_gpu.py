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
    instance_name: str = ""


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
        self.depths: List[int] = []
        self.feasible_counts: List[int] = []

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
        depth: int = 0,
        feasible_count: int = 0,
    ) -> None:
        self.obs.append(obs)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.dones.append(done)
        self.terminateds.append(terminated)
        self.node_ids.append(node_id)
        self.parent_ids.append(parent_id)
        self.depths.append(depth)
        self.feasible_counts.append(feasible_count)

    def __len__(self) -> int:
        return len(self.values)

    def clear(self) -> None:
        self.__init__()


@dataclass
class StagedEpisode:
    """
    One episode's transitions, held outside the rollout buffer until the
    stratified subsample has chosen which of them to commit.

    Staging matters for correctness: the subtree-return backup must see EVERY
    transition of the episode (a node's return is defined by its whole subtree),
    so returns are computed here on the full episode and only then filtered.
    Sampling never changes what a kept transition's return is — it only chooses
    which states' gradients enter the batch average.

    Staging also bounds memory: a 120s time-limit episode produces tens of
    thousands of observation dicts, and holding 8 such episodes in the buffer
    at full length would dominate RAM. Only the capped subset is retained.
    """
    obs: List[Dict[str, torch.Tensor]] = field(default_factory=list)
    actions: List[int] = field(default_factory=list)
    log_probs: List[float] = field(default_factory=list)
    values: List[float] = field(default_factory=list)
    dones: List[bool] = field(default_factory=list)
    terminateds: List[bool] = field(default_factory=list)
    node_ids: List[Optional[int]] = field(default_factory=list)
    parent_ids: List[Optional[int]] = field(default_factory=list)
    depths: List[int] = field(default_factory=list)
    # Number of FEASIBLE candidates at each state (action_mask.sum()), not the
    # raw candidate count. What matters for the policy gradient is how many
    # actions were actually selectable — see subsample_episode.
    feasible_counts: List[int] = field(default_factory=list)
    # Transition indices at which the incumbent strictly improved (including the
    # first incumbent). Used for guaranteed inclusion — these carry the entire
    # bonus-channel signal and are far too rare to survive random subsampling.
    incumbent_steps: List[int] = field(default_factory=list)
    first_incumbent_step: Optional[int] = None

    def __len__(self) -> int:
        return len(self.values)


@dataclass
class SubsampleReport:
    """Per-episode accounting for the stratified subsample (logging only)."""
    n_valid: int
    n_forced_dropped: int   # states with <= 1 feasible candidate (no real choice)
    n_included: int
    n_kept: int
    cell_counts: Dict[Tuple[int, int], int] = field(default_factory=dict)

    def cells_str(self) -> str:
        if not self.cell_counts:
            return "-"
        return " ".join(
            f"t{t}d{d}:{n}" for (t, d), n in sorted(self.cell_counts.items()) if n
        )


def subsample_episode(
    episode: StagedEpisode,
    valid_flags: List[bool],
    *,
    cap: Optional[int],
    n_activities: int,
    time_bands: int,
    depth_bands: int,
    incumbent_window: int,
    rng: random.Random,
) -> Tuple[List[int], SubsampleReport]:
    """
    Choose which of an episode's transitions enter the PPO batch.

    Returns (kept_indices_sorted, report). With cap=None every valid transition
    is kept (legacy behaviour).

    Selection order:

      1. Drop states that offer no real choice, judged by the number of FEASIBLE
         candidates (action_mask.sum()) rather than the raw candidate count.
         Infeasible candidates are masked to -1e9 before the softmax, which
         underflows to probability exactly 0, so:
           - exactly one feasible candidate => p = 1, log_prob = 0, entropy = 0
             and grad log pi is identically zero. A 1-of-5 state is the same
             non-decision as a 1-of-1 state, just disguised.
           - zero feasible candidates => every logit is masked to the SAME
             value, so the softmax is uniform and the gradient is nonzero even
             though the solver skips every child and the ordering is irrelevant.
             These are worse than useless: pure noise with real magnitude.
         Both cases are removed by requiring feasible_count > 1. Dropping them
         from the value loss too is safe here because advantages come from the
         tree backup, so V(X) is only ever consumed as the baseline for the
         decision at X — and there is no decision at X.
      2. Guaranteed inclusion, off-budget: the pre-first-incumbent prefix (the
         opening dive — at most ~n_activities transitions, and the regime a
         short-horizon eval scores most heavily) plus a window either side of
         every incumbent improvement (where the bonus-channel advantage lives).
      3. Stratify the remainder over episode-progress x relative-depth cells and
         spend the leftover budget evenly across them, keeping all of an
         underfull cell and redistributing its surplus.

    Both stratification axes are expressed as FRACTIONS (position within the
    episode, depth / n_activities), so the sampler transfers unchanged to
    instance families with different activity counts.
    """
    valid_idx = [i for i, ok in enumerate(valid_flags) if ok]
    report = SubsampleReport(
        n_valid=len(valid_idx), n_forced_dropped=0, n_included=0, n_kept=0
    )
    if not valid_idx:
        return [], report

    # --- 1. Drop zero-gradient forced states (feasible candidates <= 1) ---
    candidates = [i for i in valid_idx if episode.feasible_counts[i] > 1]
    report.n_forced_dropped = len(valid_idx) - len(candidates)
    if not candidates:
        # Degenerate episode (every decision forced). Nothing to learn from.
        return [], report

    if cap is None or len(candidates) <= cap:
        report.n_kept = len(candidates)
        return candidates, report

    candidate_set = set(candidates)

    # --- 2. Guaranteed inclusion (off-budget) ----------------------------
    included: set = set()
    if episode.first_incumbent_step is not None:
        for i in range(0, min(episode.first_incumbent_step + 1, len(episode))):
            if i in candidate_set:
                included.add(i)
    for step in episode.incumbent_steps:
        lo = max(0, step - incumbent_window)
        hi = min(len(episode), step + incumbent_window + 1)
        for i in range(lo, hi):
            if i in candidate_set:
                included.add(i)

    # Inclusion alone can exceed the cap on episodes with many improvements.
    # Trim to the cap rather than letting one episode dominate the batch, but
    # keep the sample spread over the episode instead of truncating its tail.
    if len(included) >= cap:
        kept = sorted(rng.sample(sorted(included), cap))
        report.n_included = len(kept)
        report.n_kept = len(kept)
        return kept, report

    report.n_included = len(included)
    budget = cap - len(included)
    remaining = [i for i in candidates if i not in included]
    if not remaining:
        kept = sorted(included)
        report.n_kept = len(kept)
        return kept, report

    # --- 3. Stratify the remainder ---------------------------------------
    n_steps = max(1, len(episode))
    denom_depth = float(max(1, n_activities))
    n_time = max(1, int(time_bands))
    n_depth = max(1, int(depth_bands))

    cells: Dict[Tuple[int, int], List[int]] = {}
    for i in remaining:
        t_band = min(n_time - 1, int(i / n_steps * n_time))
        d_frac = episode.depths[i] / denom_depth
        d_band = min(n_depth - 1, int(max(0.0, d_frac) * n_depth))
        cells.setdefault((t_band, d_band), []).append(i)

    # Water-filling: cells smaller than their share are taken whole and their
    # unused budget is redistributed over the cells that still have surplus.
    # Without this, a budget/12 quota would leave shallow-early cells (which
    # hold only a handful of transitions) unable to absorb their share while
    # deep-late cells stay truncated.
    pending = dict(cells)
    chosen: List[int] = []
    while pending and budget > 0:
        share = budget // len(pending)
        if share == 0:
            # Fewer budget slots than cells: give the remainder to a random
            # subset of cells so no band is systematically starved.
            for key in rng.sample(sorted(pending), budget):
                chosen.append(rng.choice(pending[key]))
            budget = 0
            break
        exhausted = [key for key, items in pending.items() if len(items) <= share]
        if not exhausted:
            for key, items in pending.items():
                chosen.extend(rng.sample(items, share))
                budget -= share
            break
        for key in exhausted:
            items = pending.pop(key)
            chosen.extend(items)
            budget -= len(items)

    kept = sorted(included | set(chosen))
    report.n_kept = len(kept)
    for i in kept:
        t_band = min(n_time - 1, int(i / n_steps * n_time))
        d_band = min(
            n_depth - 1, int(max(0.0, episode.depths[i] / denom_depth) * n_depth)
        )
        key = (t_band, d_band)
        report.cell_counts[key] = report.cell_counts.get(key, 0) + 1
    return kept, report


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
    "pattern": "*.rcp",
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
    "target_mb_size": 4096,
    "clip_eps": 0.2,
    "tree_gamma": 1.0,
    "tree_gamma_cost": None,
    "tree_gamma_bonus": None,
    "tree_keep_open": False,
    "min_batch_size": 4096,
    # Effective-sample-size controls. A PPO batch must contain data from at
    # least min_episodes DISTINCT instances before an update fires, and no
    # single episode may contribute more than episode_transition_cap
    # transitions. Without the cap one long time-limit episode overflows
    # min_batch_size on its own, so every update was a gradient average over a
    # single instance (task-level sample size 1) — the dominant source of
    # update-to-update variance. The cap is spent by a stratified sampler
    # (see subsample_episode) rather than uniformly, because the raw episode is
    # >99% deep, post-incumbent transitions and the rare shallow/early states
    # are the ones the short-horizon eval actually scores.
    "min_episodes": 8,
    "episode_transition_cap": 4096,   # None -> keep every valid transition
    "stratify_time_bands": 3,         # episode-progress bands (early/mid/late)
    "stratify_depth_bands": 4,        # relative-depth bands (depth / n_activities)
    "incumbent_window": 25,           # transitions kept either side of an incumbent event
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
    "eval_pattern": "*.rcp",
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

    # --- Training hyperparams ---
    total_env_steps = int(config["total_env_steps"])
    ppo_epochs = int(config["ppo_epochs"])
    target_mb_size = int(config["target_mb_size"])
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
    min_episodes = int(config["min_episodes"])
    episode_transition_cap = (
        None if config["episode_transition_cap"] is None
        else int(config["episode_transition_cap"])
    )
    stratify_time_bands = int(config["stratify_time_bands"])
    stratify_depth_bands = int(config["stratify_depth_bands"])
    incumbent_window = int(config["incumbent_window"])
    subsample_rng = random.Random(int(config["seed"]) + 1)
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
    print(f"  PPO Training (GPU-batched update, min_batch_size={min_batch_size}, min_episodes={min_episodes})")
    print(f"  total_steps={total_env_steps:,}  backup=tree(cost_gamma={tree_gamma_cost},bonus_gamma={tree_gamma_bonus})  train_instances={len(instance_paths)}  eval_instances={len(eval_paths)}")
    print(f"  clip_eps={clip_eps}  ent_coef={ent_coef_start}->{ent_coef_end} (linear decay)")
    cap_desc = "off" if episode_transition_cap is None else str(episode_transition_cap)
    print(f"  episode_cap={cap_desc}  stratify={stratify_time_bands}x{stratify_depth_bands} (time x rel-depth)  incumbent_window={incumbent_window}")
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

        # ---- Collect ONE complete episode (into a staging area) ----
        # Transitions are staged rather than written to the rollout buffer:
        # the subtree backup below needs the FULL episode, but only the
        # stratified subsample is committed to the buffer afterwards.
        ac.eval()
        staged = StagedEpisode()
        episode_tree = None
        prev_best_ms: Optional[int] = None

        while True:
            with torch.no_grad():
                action_t, log_prob_t, _, value_t = ac.get_action_and_value(obs, device)

            action = int(action_t.item())
            step_out = env.step(action)

            staged.obs.append(obs)
            staged.actions.append(action)
            staged.log_probs.append(log_prob_t.item())
            staged.values.append(value_t.item())
            staged.dones.append(step_out.done)
            staged.terminateds.append(bool(step_out.info.get("terminated", False)))
            staged.node_ids.append(step_out.info.get("node_id"))
            staged.parent_ids.append(step_out.info.get("parent_id"))
            staged.depths.append(int(step_out.info.get("depth", 0)))
            # Feasible-candidate count, NOT the raw candidate count: masked
            # (infeasible) candidates get probability exactly 0, so only the
            # feasible count determines whether this state has a real choice.
            staged.feasible_counts.append(int(obs["action_mask"].sum().item()))

            # Incumbent tracking for guaranteed inclusion in the subsample:
            # record the transition index whenever best_makespan first appears
            # or strictly improves.
            step_best = step_out.info.get("best_makespan")
            if step_best is not None and (prev_best_ms is None or step_best < prev_best_ms):
                t_idx = len(staged) - 1
                staged.incumbent_steps.append(t_idx)
                if staged.first_incumbent_step is None:
                    staged.first_incumbent_step = t_idx
                prev_best_ms = step_best

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

        # ---- Compute advantages on the FULL episode ----
        # The subtree backup is run before subsampling: a node's return is
        # defined by its entire subtree, so filtering first would corrupt it.
        ta = compute_episode_advantages_decoupled(
            tree=episode_tree,
            node_ids=staged.node_ids,
            values=staged.values,
            cost_reward_fn=cost_reward_fn,
            bonus_reward_fn=bonus_reward_fn,
            gamma_cost=tree_gamma_cost,
            gamma_bonus=tree_gamma_bonus,
            keep_open=tree_keep_open,
        )

        # ---- Stratified subsample, then commit to the buffer ----
        kept_idx, sub_report = subsample_episode(
            staged,
            ta.valid,
            cap=episode_transition_cap,
            n_activities=len(current_instance.activities),
            time_bands=stratify_time_bands,
            depth_bands=stratify_depth_bands,
            incumbent_window=incumbent_window,
            rng=subsample_rng,
        )
        n_valid_ep = len(kept_idx)

        if n_valid_ep == 0:
            print(f"[Accumulate] no usable transitions (valid={sub_report.n_valid}, "
                  f"forced={sub_report.n_forced_dropped}) — skipping episode "
                  f"(accumulated={accumulated_valid})\n")
            if global_step < total_env_steps:
                current_instance_path, current_instance = next_instance()
                obs = env.reset(instance=current_instance)
                print(f"[Episode {episode_count+1}] start  {current_instance_path.name}\n")
            episode_steps = 0
            continue

        ep_start_idx = len(buffer)
        for i in kept_idx:
            buffer.add(
                obs=staged.obs[i],
                action=staged.actions[i],
                log_prob=staged.log_probs[i],
                value=staged.values[i],
                done=staged.dones[i],
                terminated=staged.terminateds[i],
                node_id=staged.node_ids[i],
                parent_id=staged.parent_ids[i],
                depth=staged.depths[i],
                feasible_count=staged.feasible_counts[i],
            )
        ep_end_idx = len(buffer)

        print(
            f"[Subsample] valid={sub_report.n_valid}  forced_dropped={sub_report.n_forced_dropped}  "
            f"included={sub_report.n_included}  kept={sub_report.n_kept}  "
            f"cells=[{sub_report.cells_str()}]"
        )

        # Cache the per-episode tree advantages for the kept transitions only
        # (avoids recomputing subtree returns at update time).
        episode_records.append(EpisodeRecord(
            tree=episode_tree,
            cost_reward_fn=cost_reward_fn,
            bonus_reward_fn=bonus_reward_fn,
            start_idx=ep_start_idx,
            end_idx=ep_end_idx,
            n_valid=n_valid_ep,
            returns=[ta.returns[i] for i in kept_idx],
            valid_flags=[True] * n_valid_ep,
            instance_name=current_instance_path.name,
        ))
        accumulated_valid += n_valid_ep
        staged = StagedEpisode()  # release the full episode

        if writer is not None:
            writer.add_scalar("subsample/valid", sub_report.n_valid, global_step)
            writer.add_scalar("subsample/kept", sub_report.n_kept, global_step)
            writer.add_scalar("subsample/included", sub_report.n_included, global_step)
            writer.add_scalar(
                "subsample/forced_dropped", sub_report.n_forced_dropped, global_step
            )

        # ---- Check if we have enough data for a PPO update ----
        # BOTH conditions must hold. The transition count alone let a single
        # long episode fire an update on its own, making every gradient an
        # average over one instance; requiring min_episodes distinct episodes
        # is what raises the effective (task-level) sample size. Postponing the
        # update keeps the batch strictly on-policy — no update has happened, so
        # every accumulated episode was collected under the same parameters.
        if accumulated_valid < min_batch_size or len(episode_records) < min_episodes:
            print(f"[Accumulate] valid={n_valid_ep}  accumulated={accumulated_valid}/{min_batch_size}  "
                  f"episodes={len(episode_records)}/{min_episodes} — collecting more\n")
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
        # Fix minibatch SIZE, not count: derive the chunk count per update so
        # each minibatch holds ~target_mb_size transitions regardless of the
        # (fluctuating) rollout size. Fixing the count instead let the size swing
        # ~5x with buffer size, so gradient-noise and KL-estimate variance swung
        # with it; pinning the size holds both roughly constant across updates.
        # round() (not //) centers the realized size on the target rather than
        # biasing it larger. array_split then guarantees exactly n_chunks
        # contiguous chunks whose sizes differ by at most 1 — no remainder tail,
        # no orphan minibatch of 1-10 transitions whose KL is pure noise (the
        # old `T // minibatches` + range-stepping failure mode). Contiguous
        # slices preserve the R-bucketing that minimizes padding waste.
        n_chunks = max(1, round(len(sorted_indices) / target_mb_size))
        mb_chunks = np.array_split(sorted_indices, n_chunks)

        # Linear entropy-coefficient decay (with floor) based on training
        # progress. progress in [0, 1] -> coef from ent_coef_start to ent_coef_end.
        progress = min(global_step / total_env_steps, 1.0)
        ent_coef_now = ent_coef_start + (ent_coef_end - ent_coef_start) * progress

        total_pg_loss = total_vf_loss = total_ent = total_kl = 0.0
        n_kl_samples = 0
        update_count += 1
        early_stop = False

        # ---- KL-stop diagnostics (logging only, no effect on the update) ----
        # n_chunks floats with buffer size (~target_mb_size per chunk), so
        # planned == n_chunks * epochs and varies from update to update.
        chunks_per_epoch = len(mb_chunks)
        planned_steps = chunks_per_epoch * ppo_epochs
        max_kl = 0.0            # largest per-minibatch KL this update
        trigger_kl = None       # KL of the minibatch that crossed target_kl
        stop_epoch = None       # 1-based epoch the stop fired in
        stop_minibatch = None   # 1-based minibatch-within-epoch the stop fired in

        for epoch_i in range(ppo_epochs):
            if early_stop:
                break
            # Shuffle the chunk ORDER each epoch (not the chunk contents, so the
            # R-bucketing within each chunk is preserved).
            chunk_order = list(range(len(mb_chunks)))
            np.random.shuffle(chunk_order)

            for mb_i, ci in enumerate(chunk_order):
                mb_idx = mb_chunks[ci]

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

                # Does THIS minibatch trip the early-stop?
                is_trigger = target_kl is not None and approx_kl > float(target_kl)

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
        n_distinct_instances = len({rec.instance_name for rec in episode_records})
        elapsed = time.perf_counter() - t_start
        print(
            f"[Update {update_count}] "
            f"steps={global_step}  "
            f"episodes_in_batch={len(episode_records)}  "
            f"instances={n_distinct_instances}  "
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
            # Effective-sample-size diagnostics: batch_instances is the quantity
            # the min_episodes gate exists to raise (it was ~1 before).
            writer.add_scalar("train/batch_episodes", len(episode_records), global_step)
            writer.add_scalar("train/batch_instances", n_distinct_instances, global_step)
            writer.add_scalar("train/batch_size", n_valid, global_step)

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


if __name__ == "__main__":
    main()
