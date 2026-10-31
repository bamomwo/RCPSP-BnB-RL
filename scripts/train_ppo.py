"""
PPO fine-tuning for the RCPSP branching policy (separate actor/critic, Tree-GAE).

The policy samples one activity at each branch point. The solver explores that
activity first, then the remaining activities in descending policy-logit order,
so the full DFS sibling order is governed by the actor. PPO scores only the
sampled activity; the tail is a deterministic function of the logits.

Credit assignment is a single-channel Tree-GAE backup (-1 per expanded node)
with every pending frontier node bootstrapped by the critic in log-node space.
The update phase is batched: observations are padded to a common sequence
length, sorted by candidate-set size to minimize padding waste, and processed
in one forward pass per minibatch. Evaluation uses the greedy policy ordering.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from rcpsp_bnb_rl.data.dataset import list_instance_paths  # noqa: E402
from rcpsp_bnb_rl.data.parsing import load_instance  # noqa: E402
from rcpsp_bnb_rl.ml.action_order import selected_first_policy_order  # noqa: E402
from rcpsp_bnb_rl.ml.models import (  # noqa: E402
    LOG_VALUE_MIN,
    BranchingTransformer,
    SeparateActorCritic,
    load_separate_actor_critic_checkpoint,
    clamp_log_value,
    log_value_to_raw,
    raw_return_to_log,
    save_separate_actor_critic_checkpoint,
)
from rcpsp_bnb_rl.ml.il.featurize import (  # noqa: E402
    global_feature_dim, candidate_feature_dim, critic_feature_dim,
    set_global_feature_dim, set_candidate_feature_dim,
    set_resource_feature_dim, set_interaction_feature_dim,
)
from rcpsp_bnb_rl.ml.rl import BranchingEnv  # noqa: E402
from rcpsp_bnb_rl.ml.rl.tree_return import compute_tree_gae  # noqa: E402
from rcpsp_bnb_rl.bnb.branching_order import make_order_fn  # noqa: E402
from rcpsp_bnb_rl.bnb.solver import BnBSolver  # noqa: E402

# ---------------------------------------------------------------------------
# Episode record for multi-episode accumulation
# ---------------------------------------------------------------------------

@dataclass
class EpisodeRecord:
    """Per-episode bookkeeping for multi-episode PPO batching."""
    start_idx: int       # index into buffer where this episode starts
    end_idx: int         # index into buffer where this episode ends (exclusive)
    n_kept: int          # transitions committed to the buffer
    n_actor_valid: int   # of which carry a policy gradient (feasible_count > 1)
    instance_name: str = ""


# ---------------------------------------------------------------------------
# Actor-Critic wrapper
# ---------------------------------------------------------------------------

class ActorCritic(nn.Module):
    """PPO interface for independent actor and critic transformers."""

    def __init__(self, model: SeparateActorCritic) -> None:
        super().__init__()
        self.model = model

    @property
    def policy_model(self) -> BranchingTransformer:
        return self.model.actor

    @property
    def critic_model(self) -> BranchingTransformer:
        return self.model.critic

    def save_checkpoint(self, path: str, extra: dict | None = None) -> None:
        save_separate_actor_critic_checkpoint(self.model, path, extra)

    def forward(
        self,
        candidate_feats: torch.Tensor,
        global_feats: torch.Tensor,
        action_mask: Optional[torch.Tensor] = None,
        critic_feats: Optional[torch.Tensor] = None,
        resource_feats: Optional[torch.Tensor] = None,
        candidate_resource_feats: Optional[torch.Tensor] = None,
        resource_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns (logits [R], value scalar). Unbatched."""
        return self.model(
            candidate_feats, global_feats, action_mask, critic_feats,
            resource_feats, candidate_resource_feats, resource_mask,
        )

    def get_action_and_value(
        self,
        obs: Dict[str, torch.Tensor],
        device: torch.device,
        action: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, List[int]]:
        """
        Sample or score one activity, returning action, log_prob, entropy, the
        critic's LOG-SPACE value (<= 0; see log_value_to_raw) and the complete
        DFS sibling order: the sampled activity first, then the remaining
        candidates by descending logit (infeasible ones last). Only the sampled
        activity is a PPO action; the tail is deterministic given the logits.
        """
        cand = obs["candidate_feats"].to(device)
        glob = obs["global_feats"].to(device)
        mask = obs["action_mask"].to(device)
        critic = obs.get("critic_feats")
        if critic is not None:
            critic = critic.to(device)

        kwargs = {"resource_feats": obs.get("resource_feats"),
                  "candidate_resource_feats": obs.get("candidate_resource_feats"),
                  "resource_mask": obs.get("resource_mask")}
        kwargs = {k: v.to(device) for k, v in kwargs.items() if v is not None}
        logits, value = self.model(cand, glob, mask, critic, **kwargs)
        masked_logits = logits.masked_fill(~mask, -1e9)
        dist = Categorical(logits=masked_logits)
        if action is None:
            action = dist.sample()
        order = selected_first_policy_order(masked_logits.tolist(), int(action.item()))
        return action, dist.log_prob(action), dist.entropy(), value, order


# ---------------------------------------------------------------------------
# Batched PPO helpers
# ---------------------------------------------------------------------------

def batch_observations(
    obs_list: List[Dict[str, torch.Tensor]],
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor], torch.Tensor, List[int], Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
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

    has_resources = "resource_feats" in obs_list[0]
    resource_batch = interaction_batch = resource_mask_batch = None
    if has_resources:
        resource_lens = [obs["resource_feats"].shape[0] for obs in obs_list]
        M_max = max(resource_lens)
        Fr = obs_list[0]["resource_feats"].shape[1]
        Fi = obs_list[0]["candidate_resource_feats"].shape[2]
        resource_batch = torch.zeros(B, M_max, Fr, device=device)
        interaction_batch = torch.zeros(B, R_max, M_max, Fi, device=device)
        resource_mask_batch = torch.zeros(B, M_max, dtype=torch.bool, device=device)

    for i, obs in enumerate(obs_list):
        R_i = seq_lens[i]
        cand_batch[i, :R_i] = obs["candidate_feats"]
        glob_batch[i] = obs["global_feats"]
        mask_batch[i, :R_i] = obs["action_mask"]
        pad_mask[i, :R_i] = True
        if critic_batch is not None and "critic_feats" in obs:
            critic_batch[i] = obs["critic_feats"]
        if has_resources:
            M_i = obs["resource_feats"].shape[0]
            resource_batch[i, :M_i] = obs["resource_feats"].to(device)
            interaction_batch[i, :R_i, :M_i] = obs["candidate_resource_feats"].to(device)
            resource_mask_batch[i, :M_i] = obs.get(
                "resource_mask", torch.ones(M_i, dtype=torch.bool)
            ).to(device)

    return cand_batch, glob_batch, mask_batch, critic_batch, pad_mask, seq_lens, resource_batch, interaction_batch, resource_mask_batch


def compute_log_probs_entropy(
    logits_batch: torch.Tensor,
    actions: torch.Tensor,
    feasible_mask_batch: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Replay categorical activity selections in a padded minibatch.

    The mask excludes both infeasible candidates and padding. No tail-order
    likelihood or entropy enters the PPO objective.
    """
    dist = Categorical(logits=logits_batch.masked_fill(~feasible_mask_batch, -1e9))
    return dist.log_prob(actions), dist.entropy()


# ---------------------------------------------------------------------------
# Rollout buffer
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
        self.actor_valid: List[bool] = []   # feasible_count > 1: policy-gradient eligible
        self.closed: List[bool] = []        # subtree closed: Y_log is exact, critic trains on it
        self.advantages: List[float] = []   # per-episode normalized actor advantages
        self.returns: List[float] = []      # log-space critic targets Y_log

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
        actor_valid: bool = False,
        closed: bool = False,
        advantage: float = 0.0,
        return_log: float = 0.0,
    ) -> None:
        self.obs.append(obs)
        self.actions.append(int(action))
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.dones.append(done)
        self.terminateds.append(terminated)
        self.node_ids.append(node_id)
        self.parent_ids.append(parent_id)
        self.depths.append(depth)
        self.feasible_counts.append(feasible_count)
        self.actor_valid.append(bool(actor_valid))
        self.closed.append(bool(closed))
        self.advantages.append(float(advantage))
        self.returns.append(float(return_log))

    def __len__(self) -> int:
        return len(self.values)

    def clear(self) -> None:
        self.__init__()


@dataclass
class StagedEpisode:
    """
    One episode's transitions, held outside the rollout buffer until the
    stratified subsample has chosen which of them to commit.

    Staging matters for correctness: the Tree-GAE backup must see EVERY
    transition of the episode (a node's residual needs its children's values),
    so targets are computed here on the full episode and only then filtered.
    Sampling never changes what a kept transition's target is — it only chooses
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
    # Scheduled REAL tasks / total real tasks at the state where the action was
    # taken. Older callers can omit this and use depth / n_activities instead.
    completion_ratios: List[float] = field(default_factory=list)
    # Number of FEASIBLE candidates at each state (action_mask.sum()), not the
    # raw candidate count. What matters for the policy gradient is how many
    # actions were actually selectable — see subsample_episode.
    feasible_counts: List[int] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.values)


@dataclass
class SubsampleReport:
    """Per-episode accounting for the stratified subsample (logging only)."""
    n_valid: int
    n_forced_available: int  # states with <= 1 feasible candidate (no real choice)
    n_forced_kept: int       # of those, kept for critic supervision only
    n_kept: int
    cell_counts: Dict[Tuple[int, int], int] = field(default_factory=dict)
    n_shallow_available: int = 0
    n_shallow_priority_target: int = 0  # sum of per-cell targets before capacity limits
    n_shallow_kept: int = 0
    n_shallow_added: int = 0  # retained choices from explicit shallow-priority additions
    n_stratified: int = 0    # choices from the ordinary time x depth fill

    def cells_str(self) -> str:
        if not self.cell_counts:
            return "-"
        return " ".join(
            f"t{t}d{d}:{n}" for (t, d), n in sorted(self.cell_counts.items()) if n
        )


def validate_shallow_subsampling_config(
    depth_threshold: Any, priority_per_cell: Any,
) -> Tuple[float, int]:
    """Validate the shallow cutoff and bounded priority-inclusion count."""
    if (
        isinstance(depth_threshold, bool)
        or not isinstance(depth_threshold, (int, float))
        or not math.isfinite(depth_threshold)
        or not 0.0 <= depth_threshold <= 1.0
    ):
        raise ValueError("shallow_depth_threshold must be a finite number in [0, 1].")
    if (
        isinstance(priority_per_cell, bool)
        or not isinstance(priority_per_cell, int)
        or priority_per_cell < 0
    ):
        raise ValueError("shallow_priority_per_cell must be a non-negative integer.")
    return float(depth_threshold), priority_per_cell


def _sample_stratified_cells(
    indices: List[int],
    budget: int,
    cell_of: Mapping[int, Tuple[int, int]],
    rng: random.Random,
) -> List[int]:
    """Fill a budget evenly across occupied cells, recycling scarce shares.

    Sampling is without replacement. When the budget is not divisible by the
    number of cells, distribute the final slots over randomly selected cells.
    """
    if budget <= 0:
        return []
    if len(indices) <= budget:
        return list(indices)
    pending: Dict[Tuple[int, int], List[int]] = {}
    for i in indices:
        pending.setdefault(cell_of[i], []).append(i)
    chosen: List[int] = []
    while pending and budget:
        share = budget // len(pending)
        exhausted = [key for key, items in pending.items() if len(items) <= share]
        if exhausted:
            for key in exhausted:
                items = pending.pop(key)
                chosen.extend(items)
                budget -= len(items)
            continue
        # Every cell has more than its equal share. Drawing share+1 from
        # random remainder cells uses the complete budget without biasing
        # the same time/depth cell on every episode.
        extra = set(rng.sample(sorted(pending), budget % len(pending)))
        for key, items in pending.items():
            count = share + int(key in extra)
            if count:
                chosen.extend(rng.sample(items, count))
        break
    return chosen


def subsample_episode(
    episode: StagedEpisode,
    valid_flags: List[bool],
    *,
    cap: Optional[int],
    forced_cap: int,
    n_activities: int,
    time_bands: int,
    depth_bands: int,
    rng: random.Random,
    shallow_depth_threshold: float = 0.5,
    shallow_priority_per_cell: int = 0,
) -> Tuple[List[int], SubsampleReport]:
    """
    Choose which of an episode's transitions enter the PPO batch.

    Returns (kept_indices_sorted, report). Selection never changes a
    transition's advantage or target; it only chooses whose gradients enter
    the batch average. Two pools are sampled independently:

      * Actor pool (FEASIBLE candidates > 1): states with a real choice. With
        cap=None every one is kept. Otherwise bounded shallow priorities are
        added per time x depth cell (shallow = real-task completion below
        shallow_depth_threshold), then the remaining capacity is filled by the
        episode-progress x relative-depth stratified sample, recycling shares
        from scarce cells.
      * Forced pool (feasible <= 1): no usable policy gradient (one feasible
        candidate gives p = 1 and zero gradient; zero feasible gives a uniform
        softmax whose gradient is pure noise), but a valid critic target. Up to
        forced_cap of them are kept with the same stratified fill, for value
        supervision only. The caller flags them with actor_valid=False.

    The grid retains solver depth / n_activities. Real-task completion is used
    only to identify shallow decisions; older callers can omit it and fall back
    to relative solver depth. All selections are without replacement.
    """
    shallow_depth_threshold, shallow_priority_per_cell = validate_shallow_subsampling_config(
        shallow_depth_threshold, shallow_priority_per_cell,
    )
    valid_idx = [i for i, ok in enumerate(valid_flags) if ok]
    report = SubsampleReport(
        n_valid=len(valid_idx), n_forced_available=0, n_forced_kept=0, n_kept=0,
    )
    if not valid_idx:
        return [], report

    actor_pool = [i for i in valid_idx if episode.feasible_counts[i] > 1]
    forced_pool = [i for i in valid_idx if episode.feasible_counts[i] <= 1]
    report.n_forced_available = len(forced_pool)

    # Both axes use normalized coordinates so they transfer across task sizes.
    n_steps = max(1, len(valid_flags), len(episode))
    denom_depth = float(max(1, n_activities))
    n_time = max(1, int(time_bands))
    n_depth = max(1, int(depth_bands))
    relative_depth = {
        i: episode.depths[i] / denom_depth if i < len(episode.depths) else 0.0
        for i in valid_idx
    }
    completion = {
        i: (
            episode.completion_ratios[i] if i < len(episode.completion_ratios)
            else relative_depth[i]
        ) for i in valid_idx
    }
    cell_of = {
        i: (
            min(n_time - 1, int(i / n_steps * n_time)),
            min(n_depth - 1, int(max(0.0, relative_depth[i]) * n_depth)),
        ) for i in valid_idx
    }

    selected: set[int] = set()
    shallow_added: set[int] = set()
    if actor_pool:
        shallow = {i for i in actor_pool if completion[i] < shallow_depth_threshold}
        report.n_shallow_available = len(shallow)
        shallow_cells: Dict[Tuple[int, int], List[int]] = {}
        for i in shallow:
            shallow_cells.setdefault(cell_of[i], []).append(i)
        report.n_shallow_priority_target = sum(
            min(len(items), shallow_priority_per_cell) for items in shallow_cells.values()
        )
        sample_size = len(actor_pool) if cap is None else min(len(actor_pool), max(0, cap))
        if sample_size == len(actor_pool):
            # An uncapped/small episode keeps every decision without consuming RNG.
            selected.update(actor_pool)
        else:
            if shallow_priority_per_cell:
                # Top up each occupied cell separately so one long shallow
                # segment cannot supply every addition.
                extra_candidates: List[int] = []
                for items in shallow_cells.values():
                    needed = min(len(items), shallow_priority_per_cell)
                    extra_candidates.extend(
                        items if len(items) <= needed else rng.sample(items, needed)
                    )
                shallow_added.update(_sample_stratified_cells(
                    extra_candidates, sample_size, cell_of, rng,
                ))
                selected.update(shallow_added)
            selected.update(_sample_stratified_cells(
                [i for i in actor_pool if i not in selected],
                sample_size - len(selected), cell_of, rng,
            ))
        report.n_shallow_kept = len(selected & shallow)
        report.n_shallow_added = len(shallow_added)
        report.n_stratified = len(selected) - len(shallow_added)

    forced_kept = _sample_stratified_cells(forced_pool, max(0, int(forced_cap)), cell_of, rng)
    report.n_forced_kept = len(forced_kept)
    selected.update(forced_kept)

    kept = sorted(selected)
    report.n_kept = len(kept)
    for i in kept:
        key = cell_of[i]
        report.cell_counts[key] = report.cell_counts.get(key, 0) + 1
    return kept, report


ADVANTAGE_SPACES = ("log", "raw")


def validate_advantage_space(space: Any) -> str:
    """"log": A = Y_log - V_log (scale-free across depths/instances); "raw": node units."""
    value = str(space).strip().lower()
    if value not in ADVANTAGE_SPACES:
        raise ValueError(f"advantage_space must be one of {ADVANTAGE_SPACES}.")
    return value


@torch.no_grad()
def predict_frontier_values(
    ac: ActorCritic,
    observations: Mapping[int, Dict[str, torch.Tensor]],
    device: torch.device,
    *,
    batch_size: int,
) -> Dict[int, float]:
    """Raw-unit critic estimates for pending frontier states (no actor pass).

    The critic emits log-space values; they are inverted with the clamped
    transform so the Tree-GAE backup receives finite node counts. Python
    floats are returned so the boundary cannot change during PPO epochs.
    """
    if batch_size <= 0:
        raise ValueError("Frontier prediction batch_size must be positive.")
    node_ids = sorted(observations, key=lambda nid: (len(observations[nid]["action_mask"]), nid))
    values: Dict[int, float] = {}
    was_training = ac.critic_model.training
    ac.critic_model.eval()
    try:
        for start in range(0, len(node_ids), batch_size):
            ids = node_ids[start:start + batch_size]
            cand, glob, mask, critic, pad, _, resources, interactions, resource_mask = batch_observations(
                [observations[nid] for nid in ids], device,
            )
            _, v_log = ac.critic_model.forward_batch(
                cand, glob, mask, critic, pad, resources, interactions, resource_mask,
            )
            raw = log_value_to_raw(v_log.to(dtype=torch.float64))
            if not torch.isfinite(raw).all().item():
                raise ValueError("Critic produced a non-finite frontier value.")
            values.update(zip(ids, raw.cpu().tolist()))
    finally:
        ac.critic_model.train(was_training)
    return values


@dataclass
class FrontierTargets:
    """Fixed boundary values for every pending node, in raw node units."""
    values: Dict[int, float] = field(default_factory=dict)
    metrics: Dict[str, float] = field(default_factory=dict)


def build_frontier_targets(
    env: BranchingEnv,
    ac: ActorCritic,
    device: torch.device,
    *,
    batch_size: int,
) -> FrontierTargets:
    """Boundary value of every pending node under the unit expansion reward.

    Prunable, dead-end and complete-schedule states cost nothing more (0);
    eligible states with no feasible action cost exactly one expansion (-1);
    everything else is the critic's raw estimate. Coverage is complete, so a
    parent's residual never lacks a child value.
    """
    frontier = env.bootstrap_frontier(min_completion=0.0)
    values = dict.fromkeys(frontier.zero_return_ids, 0.0)
    values.update(dict.fromkeys(frontier.expansion_only_ids, -1.0))
    critic_values = predict_frontier_values(
        ac, frontier.observations, device, batch_size=batch_size,
    )
    values.update(critic_values)
    metrics: Dict[str, float] = {
        "pending": len(frontier.completion_ratios),
        "exact_zero": len(frontier.zero_return_ids),
        "expansion_estimates": len(frontier.expansion_only_ids),
        "critic_estimates": len(critic_values),
        "critic_forced": sum(
            int(frontier.observations[nid]["action_mask"].sum()) == 1 for nid in critic_values
        ),
        "uncovered": len(frontier.completion_ratios) - len(values),
    }
    if critic_values:
        raw_values = np.asarray(list(critic_values.values()), dtype=np.float64)
        metrics.update(
            critic_raw_mean=float(raw_values.mean()),
            critic_raw_abs_max=float(np.abs(raw_values).max()),
        )
    return FrontierTargets(values=values, metrics=metrics)


@dataclass
class EpisodeTargets:
    """Per-transition Tree-GAE targets for one episode (unnormalized)."""
    advantages: List[float]   # actor advantage in the configured space
    returns_log: List[float]  # critic regression target Y_log = -ln(1 + |Y_raw|)
    values_raw: List[float]   # V_old in node units
    returns_raw: List[float]  # lambda-return Y_raw = V_old + A_raw
    valid: List[bool]         # transition's node is an expanded tree node
    closed: List[bool]        # subtree had no pending descendant: Y is exact


def compute_episode_targets(
    *,
    tree: Mapping[str, object],
    node_ids: List[Optional[int]],
    values_log: List[float],
    frontier_values: Mapping[int, float],
    gamma: float,
    lam: float,
    advantage_space: str,
) -> EpisodeTargets:
    """Run the Tree-GAE backup in raw units and express targets per transition.

    advantage_space="log" uses A = Y_log - V_log = ln((1+|V_raw|)/(1+|Y_raw|)),
    positive when the subtree closed in fewer nodes than predicted and O(1) at
    every depth. Both sides are clamped to [LOG_VALUE_MIN, 0] first, so two
    saturated quantities give A = 0 rather than a spurious penalty. "raw" keeps
    A_raw = Y_raw - V_raw in node units.
    """
    values_raw_t = log_value_to_raw(torch.tensor(values_log, dtype=torch.float64))
    values_raw = {
        nid: v for nid, v in zip(node_ids, values_raw_t.tolist()) if nid is not None
    }
    gae = compute_tree_gae(
        tree, values=values_raw, frontier_values=frontier_values, gamma=gamma, lam=lam,
    )
    tt = gae.for_transitions(node_ids)
    returns_raw = torch.tensor(tt.returns, dtype=torch.float64)
    v_raw = torch.tensor(tt.values, dtype=torch.float64)
    returns_log = raw_return_to_log(returns_raw)  # clamped to LOG_VALUE_MIN
    if advantage_space == "log":
        advantages = returns_log - clamp_log_value(torch.tensor(values_log, dtype=torch.float64))
    else:
        advantages = returns_raw - v_raw
    return EpisodeTargets(
        advantages=advantages.tolist(),
        returns_log=returns_log.tolist(),
        values_raw=tt.values,
        returns_raw=tt.returns,
        valid=tt.valid,
        closed=tt.closed,
    )


def normalize_episode_advantages(
    advantages: List[float], indices: List[int],
) -> Tuple[Dict[int, float], Dict[str, float]]:
    """Standardize advantages within one episode over its kept actor-valid indices.

    Every instance then contributes on the same scale to pg_loss. An episode
    with a single decision is centred only (it carries no scale information).
    """
    if not indices:
        return {}, {"adv_mean": float("nan"), "adv_std": float("nan"), "adv_abs_max": float("nan")}
    a = np.asarray([advantages[i] for i in indices], dtype=np.float64)
    mean = float(a.mean())
    std = float(a.std()) if a.size > 1 else 0.0
    normed = (a - mean) / (std + 1e-8) if (a.size > 1 and std > 0.0) else (a - mean)
    stats = {"adv_mean": mean, "adv_std": std, "adv_abs_max": float(np.abs(a).max())}
    return dict(zip(indices, normed.tolist())), stats


def _softplus_inverse(y: float) -> float:
    return math.log(math.expm1(y)) if y < 20.0 else y


def calibrate_value_head_bias(
    model: BranchingTransformer, *, current_mean_log_value: float, target_mean_log_value: float,
) -> float:
    """Shift the critic's output bias so its typical prediction matches the targets.

    The head emits -softplus(z). Moving the final Linear bias shifts z for every
    state by the same amount, so a fresh head (which predicts ~0 remaining
    nodes everywhere) can start near the batch-mean log target instead of
    climbing there one Huber-bounded step at a time. Returns the bias delta.
    """
    output_layer = model.value_head[-1]
    if not isinstance(output_layer, nn.Linear) or output_layer.out_features != 1:
        raise TypeError("Expected the critic's final value-head layer to be Linear(..., 1).")
    current = min(-1e-6, float(current_mean_log_value))
    target = min(-1e-6, float(target_mean_log_value))
    delta = _softplus_inverse(-target) - _softplus_inverse(-current)
    with torch.no_grad():
        output_layer.bias.add_(delta)
    return float(delta)


# ---------------------------------------------------------------------------
# Evaluation (same as train_ppo.py)
# ---------------------------------------------------------------------------

def evaluate(
    model: BranchingTransformer,
    instance_paths: List[Path],
    max_resources: int,
    resource_encoder: str,
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
            resource_encoder=resource_encoder,
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
        description="PPO fine-tuning for the RCPSP branching policy (separate actor/critic, Tree-GAE).",
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
    "resource_encoder": "legacy_flat",
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
    "tree_gae_lambda": 0.95,     # Tree-GAE decay; 1.0 recovers the Monte Carlo subtree return
    "advantage_space": "log",    # "log": Y_log - V_log (scale-free) | "raw": node units
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
    "forced_state_cap": 1024,         # forced (<= 1 feasible) states kept per episode, critic only
    "shallow_depth_threshold": 0.5,   # strict real-task completion fraction
    "shallow_priority_per_cell": 0,   # bounded shallow inclusion per grid cell; 0 disables
    "ent_coef_start": 0.01,
    "ent_coef_end": 0.001,
    "vf_coef": 0.5,
    "vf_loss_type": "huber",   # "mse" | "huber" — huber is robust to shallow-hard outliers
    "huber_delta": 1.0,        # in log-node units: 1.0 ~ one e-fold of tree size
    "actor_lr": 5e-5,            # conservative fine-tuning rate for the BC-initialized actor
    "critic_lr": 3e-4,
    "critic_warmup_updates": 16, # updates with the actor frozen while the critic calibrates
    "max_grad_norm": 0.5,
    "target_kl": 0.02,
    "time_limit_s": 60.0,
    # Eval
    "eval_every_steps": 20_000,
    "eval_every_updates": None,  # positive integer overrides eval_every_steps
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

    # Update-based evaluation takes precedence over the legacy step interval.
    eval_every_updates = config["eval_every_updates"]
    if eval_every_updates is not None and (
        isinstance(eval_every_updates, bool)
        or not isinstance(eval_every_updates, int)
        or eval_every_updates <= 0
    ):
        raise ValueError("eval_every_updates must be a positive integer or null.")
    eval_every_steps = (
        int(config["eval_every_steps"]) if eval_every_updates is None else None
    )
    if eval_every_steps is not None and eval_every_steps <= 0:
        raise ValueError("eval_every_steps must be positive when eval_every_updates is null.")

    # Validate the backup configuration before loading data/models or creating
    # output files.
    tree_gamma = float(config["tree_gamma"])
    tree_gae_lambda = float(config["tree_gae_lambda"])
    if not (0.0 < tree_gamma <= 1.0) or not (0.0 <= tree_gae_lambda <= 1.0):
        raise ValueError("tree_gamma must be in (0, 1] and tree_gae_lambda in [0, 1].")
    advantage_space = validate_advantage_space(config["advantage_space"])
    critic_warmup_updates = config["critic_warmup_updates"]
    if (
        isinstance(critic_warmup_updates, bool)
        or not isinstance(critic_warmup_updates, int)
        or critic_warmup_updates < 0
    ):
        raise ValueError("critic_warmup_updates must be a non-negative integer.")
    forced_state_cap = config["forced_state_cap"]
    if (
        isinstance(forced_state_cap, bool)
        or not isinstance(forced_state_cap, int)
        or forced_state_cap < 0
    ):
        raise ValueError("forced_state_cap must be a non-negative integer.")
    shallow_depth_threshold, shallow_priority_per_cell = validate_shallow_subsampling_config(
        config["shallow_depth_threshold"], config["shallow_priority_per_cell"],
    )

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
    resource_encoder = str(config.get("resource_encoder", "legacy_flat")).strip().lower()
    use_set_encoder = resource_encoder == "set"
    max_resources = int(config.get("max_resources", 4))
    global_dim = set_global_feature_dim() if use_set_encoder else global_feature_dim(max_resources)
    candidate_dim = set_candidate_feature_dim() if use_set_encoder else candidate_feature_dim(max_resources)
    critic_dim = critic_feature_dim()

    if config["bc_checkpoint"] is not None:
        print(f"Loading BC checkpoint: {config['bc_checkpoint']}")
        base_model = load_separate_actor_critic_checkpoint(
            config["bc_checkpoint"], device=device, dropout=float(config["dropout"]),
            critic_feature_dim=critic_dim,
        )
        if use_set_encoder != bool(getattr(base_model, "resource_set_enabled", False)):
            raise ValueError(
                "PPO resource_encoder does not match the BC checkpoint schema. "
                "Train BC with resource_encoder='set' before using the variable-resource PPO path, "
                "or select resource_encoder='legacy_flat' for the old checkpoint."
            )
    else:
        print("No BC checkpoint — initialising from scratch.")
        initial_model = BranchingTransformer(
            global_dim=global_dim,
            candidate_dim=candidate_dim,
            d_model=int(config["d_model"]),
            n_heads=int(config["n_heads"]),
            n_layers=int(config["n_layers"]),
            ffn_dim=int(config["ffn_dim"]),
            dropout=float(config["dropout"]),
            critic_feature_dim=critic_dim,
            resource_feature_dim=set_resource_feature_dim() if use_set_encoder else 0,
            interaction_feature_dim=set_interaction_feature_dim() if use_set_encoder else 0,
        )
        base_model = SeparateActorCritic.from_shared(initial_model, dropout=float(config["dropout"]))

    ac = ActorCritic(base_model).to(device)
    # Parameter-disjoint networks get independent optimizers so the critic can
    # be trained alone during warmup and at its own learning rate afterwards.
    actor_optimizer = optim.AdamW(ac.policy_model.parameters(), lr=float(config["actor_lr"]))
    critic_optimizer = optim.AdamW(ac.critic_model.parameters(), lr=float(config["critic_lr"]))
    print("Actor-critic architecture: separate")
    print(f"Model params: {sum(p.numel() for p in ac.parameters()):,}")
    print(
        f"  actor={sum(p.numel() for p in ac.policy_model.parameters()):,}  "
        f"critic={sum(p.numel() for p in ac.critic_model.parameters()):,}  shared=0"
    )

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
    min_batch_size = int(config["min_batch_size"])
    min_episodes = int(config["min_episodes"])
    episode_transition_cap = (
        None if config["episode_transition_cap"] is None
        else int(config["episode_transition_cap"])
    )
    stratify_time_bands = int(config["stratify_time_bands"])
    stratify_depth_bands = int(config["stratify_depth_bands"])
    subsample_rng = random.Random(int(config["seed"]) + 1)

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
    time_limit_s = float(config["time_limit_s"])
    dominance = str(config["dominance"])

    # --- Environment and state ---
    env = BranchingEnv(
        instance_source=instance_paths[0],
        max_resources=max_resources,
        resource_encoder=resource_encoder,
        time_limit_s=time_limit_s,
        dominance=dominance,
    )

    buffer = RolloutBuffer()
    global_step = 0
    update_count = 0
    episode_count = 0
    next_eval_step = eval_every_steps
    best_mean_gap = float("inf")
    eval_log: List[Dict] = []
    last_eval_update = -1

    inst_order = list(range(len(instance_paths)))
    random.shuffle(inst_order)
    inst_idx = 0

    def next_instance() -> Tuple[Path, Any]:
        """Return the next path and instance, loading once per episode reset."""
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
    print(f"  total_steps={total_env_steps:,}  backup=tree_gae(gamma={tree_gamma},lambda={tree_gae_lambda},adv_space={advantage_space})  train_instances={len(instance_paths)}  eval_instances={len(eval_paths)}")
    print(f"  clip_eps={clip_eps}  ent_coef={ent_coef_start}->{ent_coef_end} (linear decay)")
    if eval_every_updates is not None:
        print(f"  evaluation every {eval_every_updates} PPO update(s)")
    else:
        print(f"  evaluation every {eval_every_steps:,} environment steps (after a PPO update)")
    print("  action_order=sampled_first+policy_ranked_tail  ppo_action=single_activity")
    print(
        f"  actor_lr={float(config['actor_lr']):g}  critic_lr={float(config['critic_lr']):g}  "
        f"critic_warmup_updates={critic_warmup_updates} (actor frozen, eval skipped)"
    )
    print("  eval_action_order=greedy_policy_ranking")
    cap_desc = "off" if episode_transition_cap is None else str(episode_transition_cap)
    print(f"  episode_cap={cap_desc}  forced_state_cap={forced_state_cap}  stratify={stratify_time_bands}x{stratify_depth_bands} (time x rel-depth)")
    print(
        f"  shallow_depth_threshold={shallow_depth_threshold:g} (real-task completion, strict <)  "
        f"shallow_priority_per_cell={shallow_priority_per_cell}"
    )
    vf_desc = f"huber(delta={huber_delta})" if vf_loss_type == "huber" else "mse"
    print(f"  vf_loss={vf_desc}  vf_coef={vf_coef}")
    print("  reward=-1 per expanded node  critic=log-node space, trained on closed subtrees only  frontier_bootstrap=every pending node")
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

        while True:
            with torch.no_grad():
                action_t, log_prob_t, _, value_t, order = ac.get_action_and_value(obs, device)

            action = int(action_t.item())
            step_out = env.step(action, action_order_indices=order)

            staged.obs.append(obs)
            staged.actions.append(action)
            staged.log_probs.append(log_prob_t.item())
            staged.values.append(value_t.item())
            staged.dones.append(step_out.done)
            staged.terminateds.append(bool(step_out.info.get("terminated", False)))
            staged.node_ids.append(step_out.info.get("node_id"))
            staged.parent_ids.append(step_out.info.get("parent_id"))
            staged.depths.append(int(step_out.info.get("depth", 0)))
            staged.completion_ratios.append(float(step_out.info["completion_ratio"]))
            # Feasible-candidate count, NOT the raw candidate count: masked
            # (infeasible) candidates get probability exactly 0, so only the
            # feasible count determines whether this state has a real choice.
            staged.feasible_counts.append(int(obs["action_mask"].sum().item()))

            global_step += 1
            episode_steps += 1

            if step_out.done:
                episode_tree = env.search_tree()
                stats = env.episode_stats
                episode_count += 1
                break
            else:
                obs = step_out.observation

        # ---- Episode summary ----
        ep_return = -float(stats.nodes_expanded)  # unit expansion reward: G(root)
        elapsed = time.perf_counter() - t_start
        print(
            f"[Episode {episode_count}] Done  "
            f"Instance={current_instance_path.name}  "
            f"Reason={stats.done_reason}  "
            f"Steps={episode_steps}  "
            f"Nodes={stats.nodes_expanded}  "
            f"Best_Ms={stats.best_makespan}  "
            f"Inc_Improves={stats.incumbent_improvements}  "
            f"G_root={ep_return:+.0f}  "
            f"elapsed={elapsed:.0f}s"
        )
        print()

        if writer is not None:
            writer.add_scalar("episode/return", ep_return, global_step)
            writer.add_scalar("episode/nodes_expanded", stats.nodes_expanded, global_step)
            writer.add_scalar("episode/incumbent_improvements", stats.incumbent_improvements, global_step)
            writer.add_scalar("episode/steps", episode_steps, global_step)
            if stats.best_makespan is not None:
                writer.add_scalar("episode/best_makespan", stats.best_makespan, global_step)

        if episode_tree is None:
            # Invalid-action episodes leave no solver result; nothing to credit.
            print(f"[Accumulate] no search tree (reason={stats.done_reason}) — skipping episode\n")
            if global_step < total_env_steps:
                current_instance_path, current_instance = next_instance()
                obs = env.reset(instance=current_instance)
                print(f"[Episode {episode_count+1}] start  {current_instance_path.name}\n")
            episode_steps = 0
            continue

        # ---- Tree-GAE targets on the FULL episode ----
        # Every expanded node's residual needs its children's values, so the
        # backup runs before subsampling. The frontier supplies a boundary value
        # for every pending leaf; no expanded node is dropped for being open.
        frontier_targets = build_frontier_targets(env, ac, device, batch_size=target_mb_size)
        targets = compute_episode_targets(
            tree=episode_tree,
            node_ids=staged.node_ids,
            values_log=staged.values,
            frontier_values=frontier_targets.values,
            gamma=tree_gamma,
            lam=tree_gae_lambda,
            advantage_space=advantage_space,
        )

        # ---- Stratified subsample, then per-episode advantage normalization ----
        kept_idx, sub_report = subsample_episode(
            staged,
            targets.valid,
            cap=episode_transition_cap,
            forced_cap=forced_state_cap,
            n_activities=len(current_instance.activities),
            time_bands=stratify_time_bands,
            depth_bands=stratify_depth_bands,
            rng=subsample_rng,
            shallow_depth_threshold=shallow_depth_threshold,
            shallow_priority_per_cell=shallow_priority_per_cell,
        )
        actor_idx = [i for i in kept_idx if staged.feasible_counts[i] > 1]
        actor_set = set(actor_idx)
        normalized_adv, adv_stats = normalize_episode_advantages(targets.advantages, actor_idx)
        n_valid_ep = len(kept_idx)

        metrics = frontier_targets.metrics
        closed_kept = sum(targets.closed[i] for i in kept_idx)
        open_kept = n_valid_ep - closed_kept
        root_id = episode_tree.get("root_id")
        root_t = staged.node_ids.index(root_id) if root_id in staged.node_ids else None
        metrics.update(
            closed_kept=closed_kept, open_kept=open_kept,
            open_fraction=open_kept / max(1, n_valid_ep), **adv_stats,
        )
        target_clamped = sum(targets.returns_log[i] <= LOG_VALUE_MIN + 1e-6 for i in kept_idx)
        metrics["target_clamped_kept"] = target_clamped
        if root_t is not None:
            root_v, root_y = targets.values_raw[root_t], targets.returns_raw[root_t]
            metrics.update(
                root_value_raw=root_v,
                root_return_raw=root_y,
                root_advantage=targets.advantages[root_t],
                # |V|/|Y| at the root: ~1 means a calibrated critic. On exhausted
                # episodes Y is the exact node count, so this is predicted/actual.
                root_value_ratio=abs(root_v) / max(abs(root_y), 1.0),
            )
        print(
            f"[TreeGAE] pending={int(metrics.get('pending', 0))}  "
            f"exact_zero={int(metrics.get('exact_zero', 0))}  "
            f"expansion={int(metrics.get('expansion_estimates', 0))}  "
            f"critic={int(metrics.get('critic_estimates', 0))}  "
            f"open_kept={open_kept}/{n_valid_ep}  "
            f"target_clamped={target_clamped}  "
            f"root(V={metrics.get('root_value_raw', float('nan')):.0f}, "
            f"Y={metrics.get('root_return_raw', float('nan')):.0f}, "
            f"A={metrics.get('root_advantage', float('nan')):+.3f}, "
            f"ratio={metrics.get('root_value_ratio', float('nan')):.2f})  "
            f"adv(mean={adv_stats['adv_mean']:+.3f}, std={adv_stats['adv_std']:.3f}, "
            f"absmax={adv_stats['adv_abs_max']:.3f})"
        )
        if writer is not None:
            for name, value in metrics.items():
                if value == value:  # skip NaN
                    writer.add_scalar(f"tree_gae/{name}", value, global_step)

        if n_valid_ep == 0:
            print(f"[Accumulate] no usable transitions (valid={sub_report.n_valid}, "
                  f"forced={sub_report.n_forced_available}) — skipping episode "
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
                actor_valid=i in actor_set,
                closed=targets.closed[i],
                advantage=normalized_adv.get(i, 0.0),
                return_log=targets.returns_log[i],
            )
        ep_end_idx = len(buffer)

        print(
            f"[Subsample] valid={sub_report.n_valid}  kept={sub_report.n_kept}  "
            f"actor={len(actor_idx)}  forced_kept={sub_report.n_forced_kept}/{sub_report.n_forced_available}  "
            f"shallow_kept={sub_report.n_shallow_kept}  "
            f"shallow_priority_target={sub_report.n_shallow_priority_target}  "
            f"shallow_available={sub_report.n_shallow_available}  "
            f"shallow_fraction={sub_report.n_shallow_kept / max(1, len(actor_idx)):.3f}  "
            f"shallow_added={sub_report.n_shallow_added}  stratified={sub_report.n_stratified}  "
            f"cells=[{sub_report.cells_str()}]"
        )

        episode_records.append(EpisodeRecord(
            start_idx=ep_start_idx,
            end_idx=ep_end_idx,
            n_kept=n_valid_ep,
            n_actor_valid=len(actor_idx),
            instance_name=current_instance_path.name,
        ))
        accumulated_valid += n_valid_ep
        staged = StagedEpisode()  # release the full episode

        if writer is not None:
            writer.add_scalar("subsample/valid", sub_report.n_valid, global_step)
            writer.add_scalar("subsample/kept", sub_report.n_kept, global_step)
            writer.add_scalar("subsample/actor_valid", len(actor_idx), global_step)
            writer.add_scalar("subsample/forced_available", sub_report.n_forced_available, global_step)
            writer.add_scalar("subsample/forced_kept", sub_report.n_forced_kept, global_step)
            writer.add_scalar("subsample/shallow_available", sub_report.n_shallow_available, global_step)
            writer.add_scalar(
                "subsample/shallow_priority_target", sub_report.n_shallow_priority_target, global_step,
            )
            writer.add_scalar("subsample/shallow_kept", sub_report.n_shallow_kept, global_step)
            writer.add_scalar(
                "subsample/shallow_fraction", sub_report.n_shallow_kept / max(1, len(actor_idx)), global_step,
            )
            writer.add_scalar("subsample/shallow_added", sub_report.n_shallow_added, global_step)
            writer.add_scalar("subsample/stratified", sub_report.n_stratified, global_step)
            for (time_band, depth_band), count in sub_report.cell_counts.items():
                writer.add_scalar(f"subsample/cell/t{time_band}d{depth_band}", count, global_step)

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

        # ---- Assemble per-transition targets ----
        # Advantages were normalized per episode at commit time; critic targets
        # are in log-node space and need no running statistics.
        n_total = len(buffer)
        returns = torch.tensor(buffer.returns, dtype=torch.float32)
        advantages = torch.tensor(buffer.advantages, dtype=torch.float32)
        actor_valid = torch.tensor(buffer.actor_valid, dtype=torch.bool)
        # The critic regresses only on closed subtrees, whose targets are the
        # exact observed node counts. Open-spine targets contain the critic's
        # own frontier bootstraps and would make the regression self-referential.
        critic_valid = torch.tensor(buffer.closed, dtype=torch.bool)
        values_old = torch.tensor(buffer.values, dtype=torch.float32)
        n_actor = int(actor_valid.sum().item())
        n_critic = int(critic_valid.sum().item())
        indices = np.arange(n_total)

        # Critic diagnostics in log space over the closed (trained-on) transitions.
        returns_c = returns[critic_valid]
        values_c = values_old[critic_valid]
        var_returns = returns_c.var() if n_critic > 1 else torch.tensor(0.0)
        explained_var = (
            float("nan") if var_returns.item() == 0.0
            else (1.0 - (returns_c - values_c).var() / var_returns).item()
        )
        ret_std = returns_c.std().item() if n_critic > 1 else float("nan")

        # ---- BATCHED PPO UPDATE (the GPU-optimized part) ----
        # Sort valid indices by candidate-set size (R) for bucket batching.
        # Transitions with similar R end up in the same minibatch, minimizing
        # padding waste.
        seq_lens_all = [buffer.obs[i]["candidate_feats"].shape[0] for i in indices]
        sorted_order = np.argsort(seq_lens_all)
        sorted_indices = indices[sorted_order]

        # Critic warmup: the BC actor stays frozen (no policy pass, no KL stop)
        # until the critic has calibrated to the log-node targets.
        critic_warmup = update_count + 1 <= critic_warmup_updates
        ac.critic_model.train()
        ac.policy_model.train(not critic_warmup)
        if update_count == 0 and critic_warmup_updates > 0:
            bias_delta = calibrate_value_head_bias(
                ac.critic_model,
                current_mean_log_value=float(values_c.mean().item()),
                target_mean_log_value=float(returns_c.mean().item()),
            )
            print(
                f"[Warmup] calibrated critic output bias by {bias_delta:+.3f}: "
                f"mean V_log {values_old.mean().item():.3f} -> target mean {returns.mean().item():.3f}"
            )
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

        total_pg_loss = total_vf_loss = total_ent = 0.0
        total_kl = 0.0
        n_kl_samples = 0   # actor minibatch steps (0 during warmup)
        n_vf_samples = 0   # critic minibatch steps
        update_count += 1
        early_stop = False
        initial_logprob_error = float("nan")
        mean_feasible_count = float(np.mean(buffer.feasible_counts))

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
                # Policy terms are averaged over actor-valid states only; the
                # value loss uses every transition (forced states included).
                mb_actor = actor_valid[mb_idx].to(device=device, dtype=torch.float32)
                mb_n_actor = mb_actor.sum().clamp(min=1.0)
                mb_critic = critic_valid[mb_idx].to(device=device, dtype=torch.float32)
                mb_n_critic = mb_critic.sum().clamp(min=1.0)

                # Batched forward pass
                mb_obs_list = [buffer.obs[i] for i in mb_idx]
                cand_b, glob_b, mask_b, critic_b, pad_b, _, resource_b, interaction_b, resource_mask_b = batch_observations(
                    mb_obs_list, device
                )

                # ---- Critic: closed subtrees only, log-node space ----
                _, values_b = ac.critic_model.forward_batch(
                    cand_b, glob_b, mask_b, critic_feats=critic_b, pad_mask=pad_b,
                    resource_feats=resource_b, candidate_resource_feats=interaction_b,
                    resource_mask=resource_mask_b,
                )
                # huber_delta is in log units (1.0 ~ one e-fold of tree size):
                # errors within delta are quadratic, beyond it linear. Targets
                # are already clamped; the prediction is left unclamped here so
                # a saturated critic keeps a gradient back into range.
                if vf_loss_type == "huber":
                    vf_elem = nn.functional.huber_loss(
                        values_b, mb_returns, delta=huber_delta, reduction="none",
                    )
                else:
                    vf_elem = nn.functional.mse_loss(values_b, mb_returns, reduction="none")
                vf_loss = (vf_elem * mb_critic).sum() / mb_n_critic

                critic_optimizer.zero_grad()
                (vf_coef * vf_loss).backward()
                nn.utils.clip_grad_norm_(ac.critic_model.parameters(), max_grad_norm)
                critic_optimizer.step()
                total_vf_loss += vf_loss.item()
                n_vf_samples += 1

                if critic_warmup:
                    continue  # actor frozen: no policy pass, no KL accounting

                # ---- Actor: policy pass, losses masked to actor-valid states ----
                logits_b, _ = ac.policy_model.forward_batch(
                    cand_b, glob_b, mask_b, critic_feats=None, pad_mask=pad_b,
                    resource_feats=resource_b, candidate_resource_feats=interaction_b,
                    resource_mask=resource_mask_b,
                )

                # One categorical action per branching state; the tail is fixed.
                mb_log_probs_new, mb_entropies_t = compute_log_probs_entropy(
                    logits_b, mb_actions, mask_b
                )

                # Before the first actor step, collection and batched replay use
                # identical weights. A mismatch here means PPO's ratio is not
                # the probability ratio of the action that DFS actually executed.
                if epoch_i == 0 and mb_i == 0:
                    initial_logprob_error = float(
                        ((mb_log_probs_new - mb_log_probs_old).abs() * mb_actor).max().item()
                    )
                    if not np.isfinite(initial_logprob_error) or initial_logprob_error > 1e-3:
                        raise RuntimeError(
                            "Rollout/replay action log-probability mismatch before update: "
                            f"max_abs_error={initial_logprob_error:.3e}"
                        )

                # Policy loss (clipped surrogate)
                log_ratio = mb_log_probs_new - mb_log_probs_old
                ratio = torch.exp(log_ratio)
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps)
                pg_loss = (torch.max(pg_loss1, pg_loss2) * mb_actor).sum() / mb_n_actor

                # Entropy bonus
                entropy_loss = -(mb_entropies_t * mb_actor).sum() / mb_n_actor

                actor_optimizer.zero_grad()
                (pg_loss + ent_coef_now * entropy_loss).backward()
                nn.utils.clip_grad_norm_(ac.policy_model.parameters(), max_grad_norm)
                actor_optimizer.step()

                total_pg_loss += pg_loss.item()
                total_ent += (-entropy_loss.item())

                with torch.no_grad():
                    approx_kl = ((((ratio - 1) - log_ratio) * mb_actor).sum() / mb_n_actor).item()
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

        pg_avg = total_pg_loss / n_kl_samples if n_kl_samples else float("nan")
        ent_avg = total_ent / n_kl_samples if n_kl_samples else float("nan")
        vf_avg = total_vf_loss / max(n_vf_samples, 1)
        completed_steps = n_vf_samples    # minibatch steps actually run this update
        mean_kl = total_kl / n_kl_samples if n_kl_samples > 0 else 0.0
        n_distinct_instances = len({rec.instance_name for rec in episode_records})
        elapsed = time.perf_counter() - t_start
        print(
            f"[Update {update_count}] "
            f"phase={'critic_warmup' if critic_warmup else 'ppo'}  "
            f"steps={global_step}  "
            f"episodes_in_batch={len(episode_records)}  "
            f"instances={n_distinct_instances}  "
            f"kept={n_total}  actor={n_actor}  critic={n_critic}  "
            f"pg={pg_avg:+.4f}  "
            f"vf={vf_avg:.4f}  "
            f"ev={explained_var:+.3f}  "
            f"ret_std={ret_std:.4f}  "
            f"ent={ent_avg:.4f}  "
            f"feasible={mean_feasible_count:.2f}  "
            f"ent_coef={ent_coef_now:.4f}  "
            f"kl={mean_kl:.4f}  "
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
            writer.add_scalar("train/critic_warmup", 1.0 if critic_warmup else 0.0, global_step)
            writer.add_scalar("train/vf_loss", vf_avg, global_step)
            if not critic_warmup:
                writer.add_scalar("train/pg_loss", pg_avg, global_step)
                writer.add_scalar("train/entropy", ent_avg, global_step)
            writer.add_scalar("train/feasible_candidates", mean_feasible_count, global_step)
            writer.add_scalar(
                "train/initial_logprob_error", initial_logprob_error, global_step
            )
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
            writer.add_scalar("train/actor_fraction", n_actor / max(n_total, 1), global_step)
            # Effective-sample-size diagnostics: batch_instances is the quantity
            # the min_episodes gate exists to raise (it was ~1 before).
            writer.add_scalar("train/batch_episodes", len(episode_records), global_step)
            writer.add_scalar("train/batch_instances", n_distinct_instances, global_step)
            writer.add_scalar("train/batch_size", n_total, global_step)
            writer.add_scalar("train/batch_actor_size", n_actor, global_step)
            writer.add_scalar("train/batch_critic_size", n_critic, global_step)
            writer.add_scalar("train/critic_fraction", n_critic / max(n_total, 1), global_step)

        # ---- Clear accumulation state for next cycle ----
        buffer.clear()
        episode_records.clear()
        accumulated_valid = 0

        # ---- Periodic evaluation (only after a completed PPO update) ----
        eval_due = (
            update_count % eval_every_updates == 0
            if eval_every_updates is not None
            else global_step >= next_eval_step
        )
        # During warmup the actor is the unchanged BC policy: evaluate it once
        # as the baseline (update 1) and otherwise skip the redundant evals.
        if critic_warmup and update_count > 1:
            eval_due = False
        if eval_paths and optimal_makespans and eval_due:
            if eval_every_updates is None:
                # A long batch can cross several step thresholds. Evaluate its
                # policy once, then advance to the next future threshold.
                next_eval_step = (global_step // eval_every_steps + 1) * eval_every_steps
            last_eval_update = update_count
            ac.eval()
            metrics = evaluate(
                model=ac.policy_model,
                instance_paths=eval_paths,
                max_resources=max_resources,
                resource_encoder=resource_encoder,
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
                f"update={update_count}  "
                f"solved={metrics['solved_frac']*100:.1f}%  "
                f"gap={metrics['mean_gap']:.2f}%  "
                f"nodes={metrics['mean_nodes']:.0f}"
                f"{best_tag}"
            )

            log_entry = {"step": global_step, "update": update_count, **metrics}
            eval_log.append(log_entry)
            eval_log_path.write_text(json.dumps(eval_log, indent=2))

            if writer is not None:
                writer.add_scalar("eval/solved_frac", metrics["solved_frac"], global_step)
                writer.add_scalar("eval/mean_gap", metrics["mean_gap"], global_step)
                writer.add_scalar("eval/mean_nodes", metrics["mean_nodes"], global_step)

            ckpt_path = checkpoint_dir / f"policy_ppo_step{global_step}.pt"
            ac.save_checkpoint(str(ckpt_path), extra={"train_config": config, "eval_metrics": metrics, "step": global_step, "update": update_count})
            print(f"[Checkpoint] saved  → {ckpt_path}")

            if is_best:
                best_mean_gap = metrics["mean_gap"]
                ac.save_checkpoint(str(best_model_path), extra={"train_config": config, "eval_metrics": metrics, "step": global_step, "update": update_count})
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
    # Trailing collection can advance global_step without changing the policy.
    # Evaluate only if these weights have not already been evaluated.
    if eval_paths and optimal_makespans and update_count != last_eval_update:
        ac.eval()
        metrics = evaluate(
            model=ac.policy_model,
            instance_paths=eval_paths,
            max_resources=max_resources,
            resource_encoder=resource_encoder,
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
            f"update={update_count}  "
            f"solved={metrics['solved_frac']*100:.1f}%  "
            f"gap={metrics['mean_gap']:.2f}%  "
            f"nodes={metrics['mean_nodes']:.0f}"
            f"{best_tag}"
        )
        log_entry = {"step": global_step, "update": update_count, "final": True, **metrics}
        eval_log.append(log_entry)
        eval_log_path.write_text(json.dumps(eval_log, indent=2))

        if writer is not None:
            writer.add_scalar("eval/solved_frac", metrics["solved_frac"], global_step)
            writer.add_scalar("eval/mean_gap", metrics["mean_gap"], global_step)
            writer.add_scalar("eval/mean_nodes", metrics["mean_nodes"], global_step)

        if is_best:
            best_mean_gap = metrics["mean_gap"]
            ac.save_checkpoint(str(best_model_path), extra={"train_config": config, "eval_metrics": metrics, "step": global_step, "update": update_count})
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
