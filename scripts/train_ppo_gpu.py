"""
GPU-optimized PPO fine-tuning for the RCPSP branching policy.

The policy samples one complete Plackett-Luce ranking at each branch point and
PPO scores that ranking as a compound action. The update phase is BATCHED:
observations are padded to a common sequence length and processed in one forward
pass per minibatch.

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
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

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
from rcpsp_bb_rl.ml.rl.ranking_policy import (  # noqa: E402
    RankingAction,
    make_ranking_action,
    ranking_log_probs_entropy,
)
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
    n_actor: int         # transitions eligible for the PPO actor loss
    n_critic: int        # transitions eligible for either critic head
    # Cached channel targets to avoid recomputing subtree returns at update time
    cost_returns: Optional[List[float]] = field(default=None)
    bonus_returns: Optional[List[float]] = field(default=None)
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
        """Returns (logits [R], values [2] in cost/bonus order)."""
        return self.model(candidate_feats, global_feats, action_mask, critic_feats)

    def get_ranking_and_value(
        self,
        obs: Dict[str, torch.Tensor],
        device: torch.device,
    ) -> Tuple[RankingAction, torch.Tensor]:
        """
        Sample a complete Plackett-Luce ranking during rollout collection.

        The ranking is one compound PPO action. All selections are made from
        this state before the complete activity order is sent to the DFS solver.
        """
        cand = obs["candidate_feats"].to(device)
        glob = obs["global_feats"].to(device)
        mask = obs["action_mask"].to(device)
        critic = obs.get("critic_feats")
        if critic is not None:
            critic = critic.to(device)

        logits, value = self.forward(cand, glob, mask, critic)
        ranking_action = make_ranking_action(logits, mask, sample=True)
        return ranking_action, value


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


def compute_ranking_log_probs_entropy(
    logits_batch: torch.Tensor,
    feasible_mask_batch: torch.Tensor,
    rankings: Sequence[Sequence[int]],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute complete-ranking likelihoods and autoregressive entropy.

    This wrapper keeps the training script's PPO helper boundary explicit while
    delegating the Plackett-Luce math to the shared rollout/replay implementation.
    """
    return ranking_log_probs_entropy(logits_batch, feasible_mask_batch, rankings)


# ---------------------------------------------------------------------------
# Rollout buffer
# ---------------------------------------------------------------------------

class RolloutBuffer:
    """Stores transitions from one rollout horizon."""

    def __init__(self) -> None:
        self.obs: List[Dict[str, torch.Tensor]] = []
        self.rankings: List[Tuple[int, ...]] = []
        self.log_probs: List[float] = []
        # Critic outputs are normalized (cost, bonus) channel values.
        self.values: List[Tuple[float, float]] = []
        self.dones: List[bool] = []
        self.terminateds: List[bool] = []
        self.node_ids: List[Optional[int]] = []
        self.parent_ids: List[Optional[int]] = []
        self.depths: List[int] = []
        self.feasible_counts: List[int] = []
        # A singleton has a supervised critic target but no policy gradient.
        # Keep the two loss eligibilities explicit instead of overloading the
        # old single "valid" mask.
        self.actor_eligible: List[bool] = []
        self.critic_eligible: List[bool] = []

    def add(
        self,
        obs: Dict[str, torch.Tensor],
        ranking: Sequence[int],
        log_prob: float,
        value: Sequence[float],
        done: bool,
        terminated: bool,
        node_id: Optional[int] = None,
        parent_id: Optional[int] = None,
        depth: int = 0,
        feasible_count: int = 0,
        actor_eligible: bool = False,
        critic_eligible: bool = False,
    ) -> None:
        self.obs.append(obs)
        self.rankings.append(tuple(int(index) for index in ranking))
        self.log_probs.append(log_prob)
        if len(value) != 2:
            raise ValueError(f"Expected two critic values (cost, bonus), got {value!r}")
        self.values.append((float(value[0]), float(value[1])))
        self.dones.append(done)
        self.terminateds.append(terminated)
        self.node_ids.append(node_id)
        self.parent_ids.append(parent_id)
        self.depths.append(depth)
        self.feasible_counts.append(feasible_count)
        self.actor_eligible.append(bool(actor_eligible))
        self.critic_eligible.append(bool(critic_eligible))

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
    rankings: List[Tuple[int, ...]] = field(default_factory=list)
    log_probs: List[float] = field(default_factory=list)
    values: List[Tuple[float, float]] = field(default_factory=list)
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
    """Per-episode accounting for actor and critic sampling (logging only)."""
    n_valid: int
    n_forced_dropped: int   # non-actor states (<= 1 feasible candidate)
    n_included: int
    n_kept: int             # actor samples
    n_critic_singletons: int = 0
    n_critic_singletons_kept: int = 0
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
    Choose the actor-eligible transitions of an episode for the PPO batch.

    Returns (kept_indices_sorted, report). With cap=None every
    actor-eligible transition is kept; forced/singleton states are handled by
    ``subsample_critic_singletons`` instead.

    Selection order:

      1. Keep only states that offer a real policy choice, judged by the number of FEASIBLE
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
         The actor therefore requires feasible_count > 1.  Singleton states
         are sampled separately for critic-only learning: although their policy
         gradient is exactly zero, their value targets are needed to support
         critic bootstrapping at forced pending frontier nodes.
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
            # Every remaining cell can supply the equal share.  Draw that
            # share first, then distribute any indivisible remainder over the
            # still-unselected transitions.  The old implementation stopped
            # after the floor division, silently under-filling the cap when
            # ``budget`` was not divisible by ``len(pending)``.
            selected_here: set[int] = set()
            for items in pending.values():
                sample = rng.sample(items, share)
                chosen.extend(sample)
                selected_here.update(sample)
            budget -= share * len(pending)
            if budget > 0:
                remainder_pool = [
                    i for items in pending.values() for i in items
                    if i not in selected_here
                ]
                chosen.extend(rng.sample(remainder_pool, budget))
                budget = 0
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


def subsample_critic_singletons(
    episode: StagedEpisode,
    valid_flags: List[bool],
    *,
    cap: Optional[int],
    n_activities: int,
    time_bands: int,
    depth_bands: int,
    rng: random.Random,
) -> List[int]:
    """Stratify valid forced decisions for critic-only supervision.

    A state with exactly one feasible candidate has no actor signal, but it
    still has a well-defined two-channel return and can occur as a non-exact
    pending frontier bootstrap state.  Keep it out of the actor budget while
    training the critic on a separately capped representative sample.
    """
    candidates = [
        i for i, is_valid in enumerate(valid_flags)
        if is_valid and episode.feasible_counts[i] == 1
    ]
    if not candidates or cap == 0:
        return []
    if cap is None or len(candidates) <= cap:
        return candidates

    n_steps = max(1, len(episode))
    denom_depth = float(max(1, n_activities))
    n_time = max(1, int(time_bands))
    n_depth = max(1, int(depth_bands))
    cells: Dict[Tuple[int, int], List[int]] = {}
    for i in candidates:
        t_band = min(n_time - 1, int(i / n_steps * n_time))
        d_band = min(
            n_depth - 1, int(max(0.0, episode.depths[i] / denom_depth) * n_depth)
        )
        cells.setdefault((t_band, d_band), []).append(i)

    # Same water-filling scheme as the actor sampler: retain sparse strata in
    # full and redistribute their unused quota to denser strata.
    pending = dict(cells)
    chosen: List[int] = []
    budget = cap
    while pending and budget > 0:
        share = budget // len(pending)
        if share == 0:
            for key in rng.sample(sorted(pending), budget):
                chosen.append(rng.choice(pending[key]))
            break
        exhausted = [key for key, items in pending.items() if len(items) <= share]
        if not exhausted:
            # Fill the equal per-cell allocation first, then spend any
            # indivisible remainder without reselecting an already chosen
            # singleton.  This keeps the returned indices unique and makes a
            # finite cap mean exactly that many critic samples whenever enough
            # valid singleton states exist.
            selected_here: set[int] = set()
            for items in pending.values():
                sample = rng.sample(items, share)
                chosen.extend(sample)
                selected_here.update(sample)
            budget -= share * len(pending)
            if budget > 0:
                remainder_pool = [
                    i for items in pending.values() for i in items
                    if i not in selected_here
                ]
                chosen.extend(rng.sample(remainder_pool, budget))
                budget = 0
            break
        for key in exhausted:
            items = pending.pop(key)
            chosen.extend(items)
            budget -= len(items)

    return sorted(chosen)


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

    def state_dict(self) -> Dict[str, float]:
        """Return all state needed to continue the running estimator exactly."""
        return {"mean": self.mean, "var": self.var, "count": self.count}

    def load_state_dict(self, state: Mapping[str, Any]) -> bool:
        """Restore a checkpointed value normalizer.

        Returns whether the checkpoint contained its historical sample count.
        Older checkpoints stored only mean/std, so they can restore the
        critic's output coordinates but cannot continue the running estimator
        with its original weighting.
        """
        try:
            mean = float(state["mean"])
            if "var" in state:
                var = float(state["var"])
            else:
                std = float(state["std"])
                var = std * std
            has_count = "count" in state
            count = float(state["count"]) if has_count else self.count
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("Invalid value_norm checkpoint state") from exc

        if not (np.isfinite(mean) and np.isfinite(var) and np.isfinite(count)):
            raise ValueError("value_norm checkpoint state must be finite")
        if var < 0.0 or count <= 0.0:
            raise ValueError("value_norm checkpoint state requires var >= 0 and count > 0")

        self.mean = mean
        self.var = var
        self.count = count
        return has_count


VALUE_NORM_EPS = 1e-8


def renormalize_value_head(
    model: BranchingTransformer,
    optimizer: optim.Optimizer,
    *,
    old_mean: Sequence[float],
    old_std: Sequence[float],
    new_mean: Sequence[float],
    new_std: Sequence[float],
) -> None:
    """Move the critic output from one normalized-return space to another.

    Each output row of the value head previously represented

        v_old = (v_raw - old_mean) / old_std.

    After the running return statistics change, PPO trains against

        v_new = (v_raw - new_mean) / new_std.

    The final linear layer applies this coordinate change independently to the
    cost and bonus rows, without changing either raw prediction at any state.
    Its Adam moments are scaled consistently as well, so the next optimizer
    step remains in the new output coordinate system.
    """
    old_mean = np.asarray(old_mean, dtype=np.float64).reshape(-1)
    old_std = np.asarray(old_std, dtype=np.float64).reshape(-1)
    new_mean = np.asarray(new_mean, dtype=np.float64).reshape(-1)
    new_std = np.asarray(new_std, dtype=np.float64).reshape(-1)
    if not (old_mean.size == old_std.size == new_mean.size == new_std.size == 2):
        raise ValueError("Two-channel value normalization requires exactly two statistics per head.")
    if np.any(old_std <= 0.0) or np.any(new_std <= 0.0):
        raise ValueError("Value-normalization standard deviations must be positive.")

    output_layer = model.value_head[-1]
    if not isinstance(output_layer, nn.Linear) or output_layer.out_features != 2:
        raise TypeError("Expected the critic's final value-head layer to be Linear(..., 2).")

    scale = torch.as_tensor(old_std / new_std, dtype=output_layer.weight.dtype,
                            device=output_layer.weight.device)
    offset = torch.as_tensor((old_mean - new_mean) / new_std,
                             dtype=output_layer.bias.dtype,
                             device=output_layer.bias.device)

    with torch.no_grad():
        output_layer.weight.mul_(scale[:, None])
        output_layer.bias.mul_(scale).add_(offset)

    # The final layer is used only by the value loss. Rescale Adam's moments
    # for its affine parameterization change: g_new = scale * g_old.
    for parameter in (output_layer.weight, output_layer.bias):
        state = optimizer.state.get(parameter)
        if not state:
            continue
        if "exp_avg" in state:
            state["exp_avg"].mul_(scale.reshape(-1, *([1] * (state["exp_avg"].ndim - 1))))
        if "exp_avg_sq" in state:
            sq_scale = scale * scale
            state["exp_avg_sq"].mul_(sq_scale.reshape(-1, *([1] * (state["exp_avg_sq"].ndim - 1))))
        if "max_exp_avg_sq" in state:
            sq_scale = scale * scale
            state["max_exp_avg_sq"].mul_(sq_scale.reshape(-1, *([1] * (state["max_exp_avg_sq"].ndim - 1))))


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


def load_oracle_node_counts(path: Path) -> Dict[Tuple[int, str], int]:
    """Load episode-and-instance keyed node counts for the oracle-alpha ablation."""
    lookup: Dict[Tuple[int, str], int] = {}
    with path.open() as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                key = (int(record["episode"]), str(record["instance"]))
                nodes = int(record["nodes"])
            except (TypeError, ValueError, KeyError, json.JSONDecodeError) as exc:
                raise ValueError(f"Invalid oracle record at {path}:{line_number}") from exc
            if nodes <= 0:
                raise ValueError(f"Invalid oracle node count at {path}:{line_number}: {nodes}")
            if key in lookup:
                raise ValueError(f"Duplicate oracle record for {key} in {path}")
            lookup[key] = nodes
    if not lookup:
        raise ValueError(f"No oracle records found in {path}")
    return lookup


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
    # Truncation handling: drop open ancestors (legacy) or complete them with
    # exact-zero cleanup plus a detached critic continuation estimate.
    "truncation_mode": "drop",
    "min_batch_size": 4096,
    # Effective-sample-size controls. A PPO batch must contain data from at
    # least min_episodes DISTINCT instances before an update fires, and no
    # single episode may contribute more than episode_transition_cap actor
    # transitions. The critic-only singleton sample has its own cap because it
    # is intentionally off the actor budget. Without the actor cap one long
    # time-limit episode overflows
    # min_batch_size on its own, so every update was a gradient average over a
    # single instance (task-level sample size 1) — the dominant source of
    # update-to-update variance. The cap is spent by a stratified sampler
    # (see subsample_episode) rather than uniformly, because the raw episode is
    # >99% deep, post-incumbent transitions and the rare shallow/early states
    # are the ones the short-horizon eval actually scores.
    "min_episodes": 8,
    "episode_transition_cap": 4096,   # None -> keep every actor-eligible transition
    "critic_singleton_transition_cap": 4096,  # None -> keep every valid singleton for critic only
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
    # Optional episode-and-instance keyed node-count lookup for the oracle-alpha
    # ablation. Missing records fall back to estimator_path.
    "oracle_node_counts_path": None,
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

    saved_value_norm: Optional[Dict[str, Any]] = None
    if config["bc_checkpoint"] is not None:
        print(f"Loading BC checkpoint: {config['bc_checkpoint']}")
        loaded_checkpoint = load_policy_checkpoint(
            config["bc_checkpoint"], device=device, dropout=float(config["dropout"]),
            critic_feature_dim=critic_dim, return_value_norm=True,
        )
        base_model, saved_value_norm = loaded_checkpoint
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
    truncation_mode = str(config.get("truncation_mode", "drop")).strip().lower()
    if truncation_mode not in {"drop", "critic_bootstrap"}:
        raise ValueError("truncation_mode must be 'drop' or 'critic_bootstrap'.")
    min_batch_size = int(config["min_batch_size"])
    min_episodes = int(config["min_episodes"])
    episode_transition_cap = (
        None if config["episode_transition_cap"] is None
        else int(config["episode_transition_cap"])
    )
    critic_singleton_transition_cap = (
        None if config.get("critic_singleton_transition_cap") is None
        else int(config["critic_singleton_transition_cap"])
    )
    stratify_time_bands = int(config["stratify_time_bands"])
    stratify_depth_bands = int(config["stratify_depth_bands"])
    incumbent_window = int(config["incumbent_window"])
    subsample_rng = random.Random(int(config["seed"]) + 1)
    alpha = float(config["alpha"])
    beta1 = float(config["beta1"])
    beta2 = float(config["beta2"])

    # --- Dynamic node-cost scaling ---
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
    oracle_path_raw = config.get("oracle_node_counts_path")
    oracle_path = None if oracle_path_raw is None else Path(oracle_path_raw)
    oracle_node_counts: Dict[Tuple[int, str], int] = {}
    if oracle_path is not None:
        if estimator is None:
            raise ValueError(
                "oracle_node_counts_path requires estimator_path for missing-record fallback."
            )
        oracle_node_counts = load_oracle_node_counts(oracle_path)
        print(f"Loaded oracle node counts: {len(oracle_node_counts)} records from {oracle_path}")
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
    cost_rms = RunningMeanStd()
    bonus_rms = RunningMeanStd()
    if saved_value_norm is not None:
        # Only a nested two-channel payload is a valid continuation of this
        # critic.  Older scalar PPO checkpoints must start fresh channel
        # statistics (and their scalar value head is shape-filtered on load).
        if (
            isinstance(saved_value_norm.get("cost"), Mapping)
            and isinstance(saved_value_norm.get("bonus"), Mapping)
        ):
            cost_has_count = cost_rms.load_state_dict(saved_value_norm["cost"])
            bonus_has_count = bonus_rms.load_state_dict(saved_value_norm["bonus"])
            print(
                "Restored PPO value normalization: "
                f"cost(mean={cost_rms.mean:.6g}, var={cost_rms.var:.6g}, count={cost_rms.count:.6g})  "
                f"bonus(mean={bonus_rms.mean:.6g}, var={bonus_rms.var:.6g}, count={bonus_rms.count:.6g})"
            )
            if not (cost_has_count and bonus_has_count):
                print("[warn] checkpoint channel value_norm has no count; restored means/variances with fresh counts.")
        else:
            print("[warn] checkpoint value_norm is scalar/legacy; starting fresh cost and bonus normalizers.")

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
    print(f"  truncation_mode={truncation_mode}")
    print(f"  clip_eps={clip_eps}  ent_coef={ent_coef_start}->{ent_coef_end} (linear decay)")
    print("  action_order=plackett_luce_full_ranking  ppo_action=complete_feasible_order")
    cap_desc = "off" if episode_transition_cap is None else str(episode_transition_cap)
    singleton_cap_desc = (
        "off" if critic_singleton_transition_cap is None
        else str(critic_singleton_transition_cap)
    )
    print(
        f"  episode_cap_actor={cap_desc}  critic_singleton_cap={singleton_cap_desc}  "
        f"stratify={stratify_time_bands}x{stratify_depth_bands} "
        f"(time x rel-depth)  incumbent_window={incumbent_window}"
    )
    vf_desc = f"huber(delta={huber_delta})" if vf_loss_type == "huber" else "mse"
    print(f"  vf_loss={vf_desc}  vf_coef={vf_coef}")
    if oracle_path is not None:
        print(f"  reward_scale=ORACLE   alpha=clip(c_target/N_ref, {alpha_min:g}, {alpha_max:g})  "
              f"c_target={c_target}  beta1={beta1}  beta2={beta2}")
        print(f"                        oracle={oracle_path}  fallback_estimator={estimator_path}")
    elif estimator is not None:
        print(f"  reward_scale=DYNAMIC  alpha=clip(c_target/N_est, {alpha_min:g}, {alpha_max:g})  "
              f"c_target={c_target}  beta1={beta1}  beta2={beta2}")
        print(f"                        estimator={estimator_path}")
    else:
        print(f"  reward_scale=STATIC   alpha={alpha}  beta1={beta1}  beta2={beta2}")
    print(f"{'='*80}")
    print(f"[Episode 1] start  {current_instance_path.name}\n")

    # === MAIN TRAINING LOOP ===
    # Multi-episode accumulation: collect until there are enough actor-eligible
    # transitions (>= min_batch_size) before performing a PPO update. Critic-only
    # singleton rows are retained alongside them but never satisfy this gate.
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
                ranking_action, value_t = ac.get_ranking_and_value(obs, device)

            action_order = ranking_action.solver_order_indices
            action = action_order[0]
            step_out = env.step(action, action_order_indices=action_order)

            staged.obs.append(obs)
            staged.rankings.append(tuple(ranking_action.feasible_order_indices))
            staged.log_probs.append(ranking_action.log_prob.item())
            staged.values.append((float(value_t[0].item()), float(value_t[1].item())))
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
        # Node-cost coefficient for this episode. Oracle records are keyed by
        # (episode number, instance name); missing records intentionally fall
        # back to the estimator so an ablation run can continue.
        oracle_nodes = oracle_node_counts.get((episode_count, current_instance_path.name))
        if oracle_nodes is not None:
            episode_n_est = float(oracle_nodes)
            episode_scale_source = "oracle"
        elif estimator is not None:
            episode_n_est = predict_difficulty(
                estimator, estimator_scaler, current_instance, device=device
            )
            episode_scale_source = "estimator_fallback" if oracle_path is not None else "estimator"
        else:
            episode_n_est = None
            episode_scale_source = "static"

        if episode_n_est is not None:
            episode_alpha = float(np.clip(c_target / episode_n_est, alpha_min, alpha_max))
        else:
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
            + (f"N_ref={episode_n_est:.0f}  source=oracle  alpha={episode_alpha:.2e}  "
               if episode_scale_source == "oracle" else
               f"N_est={episode_n_est:.0f}  source={episode_scale_source}  alpha={episode_alpha:.2e}  "
               if episode_n_est is not None else f"source=static  alpha={episode_alpha:.2e}  ")
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
                writer.add_scalar(
                    "episode/N_ref" if episode_scale_source == "oracle" else "episode/N_est",
                    episode_n_est,
                    global_step,
                )
            if oracle_path is not None:
                writer.add_scalar(
                    "episode/oracle_hit", 1.0 if episode_scale_source == "oracle" else 0.0,
                    global_step,
                )

        # ---- Compute subtree returns on the FULL episode ----
        # The subtree backup is run before subsampling: a node's return is
        # defined by its entire subtree, so filtering first would corrupt it.
        # Complete time-limit frontier nodes only in the explicit bootstrap
        # mode.  Exact-zero cases need no model call; genuinely unresolved
        # nodes use the existing critic as a detached continuation target.
        cost_boundary_values: Optional[Dict[int, float]] = None
        bonus_boundary_values: Optional[Dict[int, float]] = None
        if (
            truncation_mode == "critic_bootstrap"
            and episode_tree is not None
            and stats.done_reason == "time_limit"
        ):
            cost_boundary_values = {}
            bonus_boundary_values = {}
            frontier_nodes = episode_tree.get("frontier_nodes", [])
            final_incumbent = episode_tree.get("final_incumbent")
            final_frontier_lb = episode_tree.get("final_frontier_min_lb")
            for frontier_node in frontier_nodes:
                nid = int(frontier_node.node_id)
                if (
                    final_incumbent is not None
                    and frontier_node.lower_bound >= int(final_incumbent)
                ):
                    # Immediate reward (if any) remains in the normal backup;
                    # this boundary contributes no unresolved future work.
                    cost_boundary_values[nid] = 0.0
                    bonus_boundary_values[nid] = 0.0
                    continue

                # An empty ready set is pruned by the solver without an
                # expansion, so its unresolved continuation is exactly zero.
                # (A complete schedule is handled separately below.)
                if frontier_node.unscheduled and not frontier_node.ready:
                    cost_boundary_values[nid] = 0.0
                    bonus_boundary_values[nid] = 0.0
                    continue

                # Do not bootstrap a complete schedule: it may produce an
                # incumbent bonus whose effect on later DFS siblings cannot be
                # represented by an isolated local boundary value.  Such a
                # frontier remains open until a later, explicit terminal-target
                # design is introduced.
                if not frontier_node.unscheduled:
                    continue

                bootstrap_obs = env.observation_for_bootstrap(
                    frontier_node,
                    incumbent=(None if final_incumbent is None else int(final_incumbent)),
                    frontier_min_lb=(
                        None if final_frontier_lb is None else int(final_frontier_lb)
                    ),
                    stack_size=len(frontier_nodes),
                )
                # A non-terminal node whose ready activities are all infeasible
                # is still expanded once by the solver (and therefore incurs
                # exactly one node-cost charge) before producing no children.
                # A complete schedule has an empty candidate set, but is
                # intentionally left unresolved because popping it can create
                # an incumbent bonus that changes later DFS siblings.
                if (
                    bool(frontier_node.unscheduled)
                    and int(bootstrap_obs["action_mask"].sum().item()) == 0
                ):
                    cost_boundary_values[nid] = -float(episode_alpha)
                    bonus_boundary_values[nid] = 0.0
                    continue
                with torch.no_grad():
                    _, bootstrap_values = ac(
                        bootstrap_obs["candidate_feats"].to(device),
                        bootstrap_obs["global_feats"].to(device),
                        bootstrap_obs["action_mask"].to(device),
                        bootstrap_obs["critic_feats"].to(device),
                    )
                # Each value head is in its own normalized coordinate system;
                # convert each continuation to raw units before its channel's
                # discounted backup propagates it.
                cost_boundary_values[nid] = (
                    float(bootstrap_values[0].item()) * (cost_rms.std + VALUE_NORM_EPS)
                    + cost_rms.mean
                )
                bonus_boundary_values[nid] = (
                    float(bootstrap_values[1].item()) * (bonus_rms.std + VALUE_NORM_EPS)
                    + bonus_rms.mean
                )

        ta = compute_episode_advantages_decoupled(
            tree=episode_tree,
            node_ids=staged.node_ids,
            cost_reward_fn=cost_reward_fn,
            bonus_reward_fn=bonus_reward_fn,
            gamma_cost=tree_gamma_cost,
            gamma_bonus=tree_gamma_bonus,
            keep_open=False,
            cost_boundary_values=cost_boundary_values,
            bonus_boundary_values=bonus_boundary_values,
        )
        if ta.cost_returns is None or ta.bonus_returns is None:
            raise RuntimeError("Decoupled tree backup did not return channel targets.")

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
        singleton_idx = subsample_critic_singletons(
            staged,
            ta.valid,
            cap=critic_singleton_transition_cap,
            n_activities=len(current_instance.activities),
            time_bands=stratify_time_bands,
            depth_bands=stratify_depth_bands,
            rng=subsample_rng,
        )
        selected_idx = sorted(set(kept_idx) | set(singleton_idx))
        n_actor_ep = len(kept_idx)
        n_critic_ep = len(selected_idx)
        sub_report.n_critic_singletons = sum(
            1 for i, is_valid in enumerate(ta.valid)
            if is_valid and staged.feasible_counts[i] == 1
        )
        sub_report.n_critic_singletons_kept = len(singleton_idx)

        if n_critic_ep == 0:
            print(f"[Accumulate] no usable critic transitions (valid={sub_report.n_valid}, "
                  f"forced={sub_report.n_forced_dropped}) — skipping episode "
                  f"(accumulated={accumulated_valid})\n")
            if global_step < total_env_steps:
                current_instance_path, current_instance = next_instance()
                obs = env.reset(instance=current_instance)
                print(f"[Episode {episode_count+1}] start  {current_instance_path.name}\n")
            episode_steps = 0
            continue

        ep_start_idx = len(buffer)
        actor_idx_set = set(kept_idx)
        for i in selected_idx:
            is_actor = i in actor_idx_set
            buffer.add(
                obs=staged.obs[i],
                ranking=staged.rankings[i],
                log_prob=staged.log_probs[i],
                value=staged.values[i],
                done=staged.dones[i],
                terminated=staged.terminateds[i],
                node_id=staged.node_ids[i],
                parent_id=staged.parent_ids[i],
                depth=staged.depths[i],
                feasible_count=staged.feasible_counts[i],
                actor_eligible=is_actor,
                critic_eligible=True,
            )
        ep_end_idx = len(buffer)

        print(
            f"[Subsample] valid={sub_report.n_valid}  forced_dropped={sub_report.n_forced_dropped}  "
            f"included={sub_report.n_included}  kept_actor={sub_report.n_kept}  "
            f"singleton_critic={sub_report.n_critic_singletons_kept}/{sub_report.n_critic_singletons}  "
            f"cells=[{sub_report.cells_str()}]"
        )

        # Cache the two channel targets for every selected critic sample;
        # actor eligibility remains on the aligned rollout-buffer entries.
        episode_records.append(EpisodeRecord(
            tree=episode_tree,
            cost_reward_fn=cost_reward_fn,
            bonus_reward_fn=bonus_reward_fn,
            start_idx=ep_start_idx,
            end_idx=ep_end_idx,
            n_actor=n_actor_ep,
            n_critic=n_critic_ep,
            cost_returns=[ta.cost_returns[i] for i in selected_idx],
            bonus_returns=[ta.bonus_returns[i] for i in selected_idx],
            instance_name=current_instance_path.name,
        ))
        accumulated_valid += n_actor_ep
        staged = StagedEpisode()  # release the full episode

        if writer is not None:
            writer.add_scalar("subsample/valid", sub_report.n_valid, global_step)
            writer.add_scalar("subsample/kept_actor", sub_report.n_kept, global_step)
            writer.add_scalar("subsample/kept_critic_singletons", sub_report.n_critic_singletons_kept, global_step)
            writer.add_scalar("subsample/included", sub_report.n_included, global_step)
            writer.add_scalar(
                "subsample/forced_dropped", sub_report.n_forced_dropped, global_step
            )

        # ---- Check if we have enough data for a PPO update ----
        # BOTH actor conditions must hold. The transition count alone let a single
        # long episode fire an update on its own, making every gradient an
        # average over one instance; requiring min_episodes distinct episodes
        # is what raises the effective (task-level) sample size. Postponing the
        # update keeps the batch strictly on-policy — no update has happened, so
        # every accumulated episode was collected under the same parameters.
        actor_episodes = sum(rec.n_actor > 0 for rec in episode_records)
        if accumulated_valid < min_batch_size or actor_episodes < min_episodes:
            print(f"[Accumulate] actor={n_actor_ep}  critic={n_critic_ep}  "
                  f"accumulated_actor={accumulated_valid}/{min_batch_size}  "
                  f"actor_episodes={actor_episodes}/{min_episodes} — collecting more\n")
            if global_step < total_env_steps:
                current_instance_path, current_instance = next_instance()
                obs = env.reset(instance=current_instance)
                print(f"[Episode {episode_count+1}] start  {current_instance_path.name}\n")
            episode_steps = 0
            continue

        # ==================================================================
        # PPO UPDATE — we have accumulated >= min_batch_size valid transitions
        # ==================================================================

        # ---- Assemble channel returns and valid mask from all episodes ----
        all_cost_returns: List[float] = []
        all_bonus_returns: List[float] = []
        for rec in episode_records:
            if rec.cost_returns is None or rec.bonus_returns is None:
                raise RuntimeError("Episode record is missing decoupled critic targets.")
            all_cost_returns.extend(rec.cost_returns)
            all_bonus_returns.extend(rec.bonus_returns)

        raw_cost_returns = torch.tensor(all_cost_returns, dtype=torch.float32)
        raw_bonus_returns = torch.tensor(all_bonus_returns, dtype=torch.float32)
        n_total = len(buffer)
        if len(raw_cost_returns) != n_total or len(raw_bonus_returns) != n_total:
            raise RuntimeError("Rollout targets and buffer entries are misaligned.")
        actor_idx = [i for i, ok in enumerate(buffer.actor_eligible) if ok]
        critic_idx = [i for i, ok in enumerate(buffer.critic_eligible) if ok]
        n_actor = len(actor_idx)
        n_critic = len(critic_idx)
        if n_actor == 0 or n_critic == 0:
            raise RuntimeError("PPO batch must contain both actor and critic samples.")
        actor_indices = np.asarray(actor_idx, dtype=np.int64)
        critic_indices = np.asarray(critic_idx, dtype=np.int64)

        # ---- Per-channel value normalization ----
        # The actor's objective retains the raw-unit sum of channel residuals;
        # only the critic targets are normalized independently.  This avoids
        # letting normalizer scale alter the reward trade-off configured by
        # alpha, beta1, and beta2.
        old_means = np.array([cost_rms.mean, bonus_rms.mean], dtype=np.float64)
        old_stds = np.array(
            [cost_rms.std + VALUE_NORM_EPS, bonus_rms.std + VALUE_NORM_EPS],
            dtype=np.float64,
        )
        cost_rms.update(raw_cost_returns[critic_indices].numpy().astype(np.float64))
        bonus_rms.update(raw_bonus_returns[critic_indices].numpy().astype(np.float64))
        new_means = np.array([cost_rms.mean, bonus_rms.mean], dtype=np.float64)
        new_stds = np.array(
            [cost_rms.std + VALUE_NORM_EPS, bonus_rms.std + VALUE_NORM_EPS],
            dtype=np.float64,
        )

        # Preserve the critic's raw predictions while changing its output
        # coordinate system. Without this, returns would be normalized with
        # the new stats while the value loss still used old-coordinate outputs.
        renormalize_value_head(
            ac.model,
            optimizer,
            old_mean=old_means,
            old_std=old_stds,
            new_mean=new_means,
            new_std=new_stds,
        )

        cost_returns = (raw_cost_returns - new_means[0]) / new_stds[0]
        bonus_returns = (raw_bonus_returns - new_means[1]) / new_stds[1]
        values_old = torch.tensor([buffer.values[i] for i in critic_idx], dtype=torch.float32)
        old_means_t = torch.tensor(old_means, dtype=torch.float32)
        old_stds_t = torch.tensor(old_stds, dtype=torch.float32)
        values_raw = values_old * old_stds_t + old_means_t
        combined_returns_valid = raw_cost_returns[critic_indices] + raw_bonus_returns[critic_indices]
        combined_values_valid = values_raw[:, 0] + values_raw[:, 1]
        # Actor advantages are computed only where a genuine ranking choice
        # exists. Singleton critic-only rows have no valid log-probability
        # gradient and must not dilute actor normalization.
        actor_values_old = torch.tensor([buffer.values[i] for i in actor_idx], dtype=torch.float32)
        actor_values_raw = actor_values_old * old_stds_t + old_means_t
        actor_combined_returns = raw_cost_returns[actor_indices] + raw_bonus_returns[actor_indices]
        actor_advantages = actor_combined_returns - (actor_values_raw[:, 0] + actor_values_raw[:, 1])
        advantages = torch.zeros_like(raw_cost_returns)
        advantages[actor_indices] = actor_advantages

        # Critic diagnostics are evaluated in raw reward units on every
        # critic-supervised row (actor choices plus critic-only singletons).
        # Keeping the channel EVs separate is important: the dense cost head
        # can look healthy in the combined metric while the sparse incumbent
        # bonus head fails to learn useful variation.
        def _explained_variance(targets: torch.Tensor, predictions: torch.Tensor) -> float:
            target_var = targets.var()
            if target_var.item() == 0.0:
                # A constant target has no variance to explain. In particular,
                # bonus returns are often all zero in a batch, so NaN is the
                # informative/standard value rather than a critic failure.
                return float("nan")
            return (1.0 - (targets - predictions).var() / target_var).item()

        ev_cost = _explained_variance(
            raw_cost_returns[critic_indices], values_raw[:, 0]
        )
        ev_bonus = _explained_variance(
            raw_bonus_returns[critic_indices], values_raw[:, 1]
        )
        ev_combined = _explained_variance(
            combined_returns_valid, combined_values_valid
        )
        ret_std = combined_returns_valid.std().item()

        adv_valid = actor_advantages
        advantages[actor_indices] = (adv_valid - adv_valid.mean()) / (adv_valid.std() + 1e-8)

        # ---- BATCHED PPO UPDATE (the GPU-optimized part) ----
        # Sort actor rows by candidate-set size (R) for bucket batching.
        # Critic-only singleton rows are sorted and spread over the same actor
        # chunks, so every value target is trained once per epoch without
        # entering any actor statistic or policy-loss reduction.
        seq_lens_all = [buffer.obs[i]["candidate_feats"].shape[0] for i in actor_idx]
        sorted_order = np.argsort(seq_lens_all)
        sorted_actor_indices = actor_indices[sorted_order]

        ac.train()
        # Fix ACTOR minibatch SIZE, not count: derive the chunk count per update
        # so each actor minibatch holds ~target_mb_size transitions regardless
        # of the (fluctuating) rollout size. Critic-only singleton rows are
        # attached separately and do not alter actor gradient scale or KL.
        # round() (not //) centers the realized size on the target rather than
        # biasing it larger. array_split then guarantees exactly n_chunks
        # contiguous chunks whose sizes differ by at most 1 — no remainder tail,
        # no orphan minibatch of 1-10 transitions whose KL is pure noise (the
        # old `T // minibatches` + range-stepping failure mode). Contiguous
        # slices preserve the R-bucketing that minimizes padding waste.
        n_chunks = max(1, round(len(sorted_actor_indices) / target_mb_size))
        actor_mb_chunks = np.array_split(sorted_actor_indices, n_chunks)
        critic_only_idx = [
            i for i, is_actor in enumerate(buffer.actor_eligible)
            if buffer.critic_eligible[i] and not is_actor
        ]
        critic_only_sorted = np.asarray(
            sorted(critic_only_idx, key=lambda i: buffer.obs[i]["candidate_feats"].shape[0]),
            dtype=np.int64,
        )
        critic_only_mb_chunks = np.array_split(critic_only_sorted, n_chunks)

        # Linear entropy-coefficient decay (with floor) based on training
        # progress. progress in [0, 1] -> coef from ent_coef_start to ent_coef_end.
        progress = min(global_step / total_env_steps, 1.0)
        ent_coef_now = ent_coef_start + (ent_coef_end - ent_coef_start) * progress

        total_pg_loss = total_vf_loss = total_ent = total_ent_per_choice = 0.0
        total_kl = 0.0
        n_kl_samples = 0
        # PPO's ratio is defined only for actor-eligible complete rankings.
        # Accumulate numerator/denominator separately so clipfrac remains
        # sample-weighted if chunks differ in size or KL stopping truncates an
        # update part-way through its planned replay passes.
        total_clipped_actor_samples = 0
        total_replayed_actor_samples = 0
        update_count += 1
        early_stop = False
        initial_logprob_error = float("nan")
        mean_ranking_length = float(
            np.mean([len(buffer.rankings[i]) for i in actor_idx])
        )

        # ---- KL-stop diagnostics (logging only, no effect on the update) ----
        # n_chunks follows actor batch size (~target_mb_size per chunk), so
        # planned == n_chunks * epochs and varies from update to update.
        chunks_per_epoch = len(actor_mb_chunks)
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
            chunk_order = list(range(len(actor_mb_chunks)))
            np.random.shuffle(chunk_order)

            for mb_i, ci in enumerate(chunk_order):
                actor_mb_idx = actor_mb_chunks[ci]
                critic_only_mb_idx = critic_only_mb_chunks[ci]
                # Actor rows intentionally precede critic-only rows.  The
                # first slice is the sole input to PPO likelihood, entropy,
                # KL, and policy loss; the complete minibatch feeds the critic.
                mb_idx = np.concatenate((actor_mb_idx, critic_only_mb_idx))
                n_actor_mb = len(actor_mb_idx)

                mb_log_probs_old = torch.tensor(
                    [buffer.log_probs[i] for i in actor_mb_idx], dtype=torch.float32, device=device
                )
                mb_rankings = [buffer.rankings[i] for i in actor_mb_idx]
                mb_advantages = advantages[actor_mb_idx].to(device)
                mb_cost_returns = cost_returns[mb_idx].to(device)
                mb_bonus_returns = bonus_returns[mb_idx].to(device)

                # Batched forward pass
                mb_obs_list = [buffer.obs[i] for i in mb_idx]
                cand_b, glob_b, mask_b, critic_b, pad_b, _ = batch_observations(
                    mb_obs_list, device
                )

                logits_b, values_b = ac.model.forward_batch(
                    cand_b, glob_b, mask_b, critic_b, pad_b
                )

                # Complete Plackett-Luce action likelihood and entropy.
                (
                    mb_log_probs_new,
                    mb_entropies_t,
                    mb_entropy_per_choice_t,
                ) = compute_ranking_log_probs_entropy(
                    logits_b[:n_actor_mb], mask_b[:n_actor_mb], mb_rankings
                )

                # Before the first optimizer step, collection and batched replay
                # use identical weights. A mismatch here means PPO's ratio is not
                # the probability ratio of the action that DFS actually executed.
                if epoch_i == 0 and mb_i == 0:
                    initial_logprob_error = float(
                        (mb_log_probs_new - mb_log_probs_old).abs().max().item()
                    )
                    if not np.isfinite(initial_logprob_error) or initial_logprob_error > 1e-3:
                        raise RuntimeError(
                            "Rollout/replay ranking log-probability mismatch before update: "
                            f"max_abs_error={initial_logprob_error:.3e}"
                        )

                # Policy loss (clipped surrogate)
                log_ratio = mb_log_probs_new - mb_log_probs_old
                ratio = torch.exp(log_ratio)
                with torch.no_grad():
                    total_clipped_actor_samples += int(
                        ((ratio - 1.0).abs() > clip_eps).sum().item()
                    )
                    total_replayed_actor_samples += n_actor_mb
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss — each head predicts its own independently
                # normalized return. Average the channel losses so changing
                # from one head to two does not double the critic-loss scale.
                if vf_loss_type == "huber":
                    vf_cost = nn.functional.huber_loss(
                        values_b[:, 0], mb_cost_returns, delta=huber_delta
                    )
                    vf_bonus = nn.functional.huber_loss(
                        values_b[:, 1], mb_bonus_returns, delta=huber_delta
                    )
                else:
                    vf_cost = nn.functional.mse_loss(values_b[:, 0], mb_cost_returns)
                    vf_bonus = nn.functional.mse_loss(values_b[:, 1], mb_bonus_returns)
                vf_loss = 0.5 * (vf_cost + vf_bonus)

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
                total_ent_per_choice += mb_entropy_per_choice_t.mean().item()

                with torch.no_grad():
                    approx_kl = ((ratio - 1) - log_ratio).mean().item()
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
        clipfrac = (
            total_clipped_actor_samples / total_replayed_actor_samples
            if total_replayed_actor_samples > 0 else float("nan")
        )
        n_distinct_instances = len({rec.instance_name for rec in episode_records})
        elapsed = time.perf_counter() - t_start
        print(
            f"[Update {update_count}] "
            f"steps={global_step}  "
            f"episodes_in_batch={len(episode_records)}  "
            f"instances={n_distinct_instances}  "
            f"actor={n_actor}/{n_total}  "
            f"critic={n_critic}/{n_total}  "
            f"pg={total_pg_loss/n_updates:+.4f}  "
            f"vf={total_vf_loss/n_updates:.4f}  "
            f"ev_cost={ev_cost:+.3f}  "
            f"ev_bonus={ev_bonus:+.3f}  "
            f"ev_combined={ev_combined:+.3f}  "
            f"ret_std={ret_std:.4f}  "
            f"ent={total_ent/n_updates:.4f}  "
            f"ent_pos={total_ent_per_choice/n_updates:.4f}  "
            f"rank_len={mean_ranking_length:.2f}  "
            f"ent_coef={ent_coef_now:.4f}  "
            f"kl={mean_kl:.4f}  "
            f"clipfrac={clipfrac:.3f}  "
            f"max_kl={max_kl:.4f}  "
            f"lp_err={initial_logprob_error:.2e}  "
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
            writer.add_scalar(
                "train/entropy_per_choice",
                total_ent_per_choice / n_updates,
                global_step,
            )
            writer.add_scalar("train/ranking_length", mean_ranking_length, global_step)
            writer.add_scalar(
                "train/initial_logprob_error", initial_logprob_error, global_step
            )
            writer.add_scalar("train/ent_coef", ent_coef_now, global_step)
            writer.add_scalar("train/approx_kl", mean_kl, global_step)
            writer.add_scalar("train/clipfrac", clipfrac, global_step)
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
            # Keep the legacy combined tag for existing TensorBoard views and
            # add explicit channel diagnostics for the two-head critic.
            if not (ev_cost != ev_cost):  # skip NaN (constant target)
                writer.add_scalar("train/ev_cost", ev_cost, global_step)
            if not (ev_bonus != ev_bonus):
                writer.add_scalar("train/ev_bonus", ev_bonus, global_step)
            if not (ev_combined != ev_combined):
                writer.add_scalar("train/ev_combined", ev_combined, global_step)
                writer.add_scalar("train/explained_variance", ev_combined, global_step)
            writer.add_scalar("train/return_std", ret_std, global_step)
            writer.add_scalar("train/actor_fraction", n_actor / max(n_total, 1), global_step)
            writer.add_scalar("train/critic_fraction", n_critic / max(n_total, 1), global_step)
            # Effective-sample-size diagnostics: batch_instances is the quantity
            # the min_episodes gate exists to raise (it was ~1 before).
            writer.add_scalar("train/batch_episodes", len(episode_records), global_step)
            writer.add_scalar("train/batch_instances", n_distinct_instances, global_step)
            writer.add_scalar("train/actor_batch_size", n_actor, global_step)
            writer.add_scalar("train/critic_batch_size", n_critic, global_step)

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
            value_norm_state = {
                "cost": cost_rms.state_dict(),
                "bonus": bonus_rms.state_dict(),
            }
            save_policy_checkpoint(
                ac.model,
                str(ckpt_path),
                extra={"train_config": config, "eval_metrics": metrics, "value_norm": value_norm_state},
            )
            print(f"[Checkpoint] saved  → {ckpt_path}")

            if is_best:
                best_mean_gap = metrics["mean_gap"]
                save_policy_checkpoint(
                    ac.model,
                    str(best_model_path),
                    extra={"train_config": config, "eval_metrics": metrics, "step": global_step, "value_norm": value_norm_state},
                )
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
            value_norm_state = {
                "cost": cost_rms.state_dict(),
                "bonus": bonus_rms.state_dict(),
            }
            save_policy_checkpoint(
                ac.model,
                str(best_model_path),
                extra={"train_config": config, "eval_metrics": metrics, "step": global_step, "value_norm": value_norm_state},
            )
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
