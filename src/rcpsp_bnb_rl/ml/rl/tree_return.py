"""Observed makespans, conservative bounds, and split-node preferences."""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
import math
from typing import TYPE_CHECKING, Dict, List, Mapping, Optional, Tuple

if TYPE_CHECKING:
    import torch


EXPLORED_STATUSES = frozenset({"expanded", "solution", "pruned"})


@dataclass(frozen=True)
class SubtreeOutcome:
    best_obs_cmax: Optional[float]
    subtree_closed: bool
    valid_lb: Optional[float]

    @property
    def has_schedule(self) -> bool:
        return self.best_obs_cmax is not None


@dataclass
class SubtreeOutcomes:
    outcomes: Dict[int, SubtreeOutcome]
    explored_children: Dict[int, List[int]]
    all_children: Dict[int, List[int]]

    @property
    def best_obs_cmax(self) -> Dict[int, Optional[float]]:
        return {nid: outcome.best_obs_cmax for nid, outcome in self.outcomes.items()}

    @property
    def subtree_closed(self) -> Dict[int, bool]:
        return {nid: outcome.subtree_closed for nid, outcome in self.outcomes.items()}

    @property
    def valid_lb(self) -> Dict[int, Optional[float]]:
        return {nid: outcome.valid_lb for nid, outcome in self.outcomes.items()}


@dataclass(frozen=True)
class PreferencePair:
    winner_activity: int
    loser_activity: int
    winner_node_id: int
    loser_node_id: int
    pair_type: str
    rel_gap: float
    weight: float


@dataclass
class PreferenceDiagnostics:
    n_split_nodes: int = 0
    n_active_split_nodes: int = 0
    n_certified: int = 0
    n_empirical: int = 0
    n_tie_break: int = 0
    n_masked_censored: int = 0
    n_ties: int = 0
    n_masked_unknown: int = 0
    n_masked_pending: int = 0
    n_masked_pruned: int = 0
    n_filtered_mode: int = 0
    split_nodes_by_depth: Dict[str, int] = field(default_factory=dict)
    active_split_nodes_by_depth: Dict[str, int] = field(default_factory=dict)


@dataclass
class PreferenceBatch:
    split_pairs: Dict[int, List[PreferencePair]]
    diagnostics: PreferenceDiagnostics
    outcomes: SubtreeOutcomes


def _number(value: object, name: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise ValueError(f"{name} must be finite and non-negative")
    return result


def _tree_structure(tree: Mapping[str, object]):
    by_id: Dict[int, Mapping[str, object]] = {}
    for node in tree.get("nodes", []):
        nid = int(node["id"])
        if nid in by_id:
            raise ValueError(f"Duplicate tree node {nid}")
        by_id[nid] = node
    children = {nid: [] for nid in by_id}
    parents: Dict[int, Optional[int]] = {}
    for nid, node in by_id.items():
        parent = node.get("parent_id")
        pid = None if parent is None else int(parent)
        parents[nid] = pid
        if pid is not None:
            if pid not in by_id:
                raise ValueError(f"Missing parent {pid} of tree node {nid}")
            children[pid].append(nid)
    # Leaf-to-root ordering avoids recursion and creation-id assumptions.
    remaining = {nid: len(cs) for nid, cs in children.items()}
    postorder = [nid for nid, count in remaining.items() if count == 0]
    for nid in postorder:
        pid = parents[nid]
        if pid is not None:
            remaining[pid] -= 1
            if remaining[pid] == 0:
                postorder.append(pid)
    if len(postorder) != len(by_id):
        raise ValueError("Search tree contains a cycle")
    return by_id, children, postorder


def compute_subtree_outcomes(tree: Mapping[str, object]) -> SubtreeOutcomes:
    """Back up feasible observations and bounds without costs or bootstrapping.

    Pending branches never produce labels. Missing dominance children prevent
    closure; a dominance-only dead end is entirely masked. Open internal nodes
    retain their static bound. Omitted feasibility children can close a subtree
    only with an exported cutoff bound.
    """
    by_id, children, postorder = _tree_structure(tree)
    outcomes: Dict[int, SubtreeOutcome] = {}
    explored_children: Dict[int, List[int]] = {}
    for nid in postorder:
        node = by_id[nid]
        status = str(node.get("status"))
        if status == "pending":
            continue
        if status not in EXPLORED_STATUSES:
            raise ValueError(f"Unknown status {status!r} for tree node {nid}")
        cs = children[nid]
        explored = [cid for cid in cs if cid in outcomes]
        if status == "expanded":
            explored_children[nid] = explored
        lb = None if node.get("lower_bound") is None else _number(
            node["lower_bound"], f"Lower bound of node {nid}",
        )
        if status == "solution":
            makespan = node.get("makespan")
            if makespan is None:
                makespan = node.get("realized_makespan")
            if makespan is None:
                raise ValueError(f"Solution node {nid} has no realized makespan")
            mk = _number(makespan, f"Makespan of node {nid}")
            outcomes[nid] = SubtreeOutcome(mk, True, mk)
            continue
        if status == "pruned":
            reason = node.get("prune_reason")
            is_bound = reason == "bound"
            outcomes[nid] = SubtreeOutcome(None, is_bound, lb if is_bound else None)
            continue

        observed = [outcomes[cid].best_obs_cmax for cid in explored
                    if outcomes[cid].best_obs_cmax is not None]
        best = min(observed) if observed else None
        dominance_count = int(node.get("dominance_pruned_children", 0))
        infeasible_count = int(node.get("infeasible_pruned_children", 0))
        infeasible_lb_raw = node.get("infeasible_child_lower_bound")
        infeasible_lb = None if infeasible_lb_raw is None else _number(
            infeasible_lb_raw, f"Feasibility cutoff of node {nid}",
        )
        closed = (
            not bool(node.get("unfinished_plunge", False)) and dominance_count == 0
            and len(explored) == len(cs)
            and all(outcomes[cid].subtree_closed for cid in explored)
            and all(outcomes[cid].valid_lb is not None for cid in explored)
            and (infeasible_count == 0 or infeasible_lb is not None)
            and bool(cs or infeasible_count)
        )
        if not cs and dominance_count:
            valid_lb = None
        elif not cs and not infeasible_count:
            valid_lb = None
        elif closed:
            child_bounds = [outcomes[cid].valid_lb for cid in explored]
            if infeasible_count:
                child_bounds.append(infeasible_lb)
            valid_lb = min(child_bounds)
            if lb is not None:
                valid_lb = max(lb, valid_lb)
        else:
            valid_lb = lb
        outcomes[nid] = SubtreeOutcome(best, closed, valid_lb)
    return SubtreeOutcomes(outcomes, explored_children, children)


def extract_valid_preference_pairs(
    tree: Mapping[str, object],
    outcomes: Optional[SubtreeOutcomes] = None,
    *,
    preference_mode: str = "all",
    pair_gap_alpha: float = 1.0,
    tie_break_weight: float = 0.5,
) -> PreferenceBatch:
    """Extract one directed preference per comparable unordered child pair.

    Strict schedule comparisons are empirical unless the loser's valid bound
    certifies them. Bound equality is a separate search-efficiency label.
    'certified_only' omits empirical and bound-equality labels.
    """
    if preference_mode not in {"all", "certified_only"}:
        raise ValueError("preference_mode must be 'all' or 'certified_only'")
    gap_alpha = _number(pair_gap_alpha, "pair_gap_alpha")
    tie_weight = _number(tie_break_weight, "tie_break_weight")
    backed = compute_subtree_outcomes(tree) if outcomes is None else outcomes
    by_id, _, _ = _tree_structure(tree)
    max_decision_depth = max(
        (int(by_id[nid].get("depth", 0)) for nid in backed.explored_children), default=-1,
    )
    n_depths = max(int(tree.get("n_activities", 0)), max_decision_depth + 1)
    diagnostics = PreferenceDiagnostics(
        split_nodes_by_depth={f"d{depth}": 0 for depth in range(n_depths)},
        active_split_nodes_by_depth={f"d{depth}": 0 for depth in range(n_depths)},
    )
    split_pairs: Dict[int, List[PreferencePair]] = {}
    for parent, explored in backed.explored_children.items():
        if len(explored) < 2:
            continue
        diagnostics.n_split_nodes += 1
        depth = int(by_id[parent].get("depth", 0))
        depth_key = f"d{depth}"
        diagnostics.split_nodes_by_depth[depth_key] += 1
        pairs: List[PreferencePair] = []
        for left, right in combinations(backed.all_children[parent], 2):
            if left not in backed.outcomes or right not in backed.outcomes:
                diagnostics.n_masked_pending += 1
                continue
            a, b = backed.outcomes[left], backed.outcomes[right]
            if (a.best_obs_cmax is None and a.valid_lb is None) or (b.best_obs_cmax is None and b.valid_lb is None):
                diagnostics.n_masked_unknown += 1
                continue
            if a.best_obs_cmax is None and b.best_obs_cmax is None:
                diagnostics.n_masked_pruned += 1
                continue
            if a.best_obs_cmax is None or (
                b.best_obs_cmax is not None and b.best_obs_cmax < a.best_obs_cmax
            ):
                left, right, a, b = right, left, b, a
            winner_mk = a.best_obs_cmax
            if b.best_obs_cmax is not None:
                if winner_mk == b.best_obs_cmax:
                    diagnostics.n_ties += 1
                    continue
                kind = "certified" if b.valid_lb is not None and winner_mk < b.valid_lb else "empirical"
                gap = (b.best_obs_cmax - winner_mk) / max(1.0, winner_mk)
            else:
                if winner_mk > b.valid_lb:
                    diagnostics.n_masked_censored += 1
                    continue
                kind = "tie_break_bound" if winner_mk == b.valid_lb else "certified"
                gap = (b.valid_lb - winner_mk) / max(1.0, winner_mk)
            if preference_mode == "certified_only" and kind != "certified":
                diagnostics.n_filtered_mode += 1
                continue
            winner_activity = by_id[left].get("activity")
            loser_activity = by_id[right].get("activity")
            if winner_activity is None or loser_activity is None:
                raise ValueError(f"Compared children of node {parent} need activity ids")
            weight = tie_weight if kind == "tie_break_bound" else 1.0 + gap_alpha * gap
            if weight == 0.0:
                diagnostics.n_filtered_mode += 1
                continue
            pairs.append(PreferencePair(
                int(winner_activity), int(loser_activity), left, right, kind, gap, weight,
            ))
            if kind == "certified":
                diagnostics.n_certified += 1
            elif kind == "empirical":
                diagnostics.n_empirical += 1
            else:
                diagnostics.n_tie_break += 1
        if pairs:
            split_pairs[parent] = pairs
            diagnostics.n_active_split_nodes += 1
            diagnostics.active_split_nodes_by_depth[depth_key] += 1
    return PreferenceBatch(split_pairs, diagnostics, backed)


def ranknet_pairwise_loss(
    logits: "torch.Tensor",
    winner_slots: "torch.Tensor",
    loser_slots: "torch.Tensor",
    pair_weights: "torch.Tensor",
    pair_mask: "torch.Tensor",
) -> Tuple["torch.Tensor", int]:
    """Average weighted RankNet over pairs per node, then over active nodes.

    Pair tensors have shape [B, P]; logits have shape [B, R]. Masked indices
    may be -1. No active pairs returns a differentiable zero.
    """
    import torch
    import torch.nn.functional as F

    if logits.ndim != 2 or winner_slots.ndim != 2:
        raise ValueError("logits and pair tensors must be matrices")
    shape = winner_slots.shape
    if any(t.shape != shape for t in (loser_slots, pair_weights, pair_mask)) or shape[0] != logits.shape[0]:
        raise ValueError("Pair tensor shapes must match the logits batch")
    active_mask = pair_mask.bool()
    n_pairs = int(active_mask.sum().item())
    if n_pairs == 0:
        return logits.sum() * 0.0, 0
    if bool(((winner_slots[active_mask] < 0) | (winner_slots[active_mask] >= logits.shape[1])).any()) or bool(((loser_slots[active_mask] < 0) | (loser_slots[active_mask] >= logits.shape[1])).any()):
        raise ValueError("Active pair index is outside the candidate logits")
    if not bool(torch.isfinite(pair_weights[active_mask]).all()) or bool((pair_weights[active_mask] < 0).any()):
        raise ValueError("Active pair weights must be finite and non-negative")
    safe_win = winner_slots.masked_fill(~active_mask, 0)
    safe_lose = loser_slots.masked_fill(~active_mask, 0)
    differences = logits.gather(1, safe_win) - logits.gather(1, safe_lose)
    weights = pair_weights.masked_fill(~active_mask, 0.0)
    losses = F.softplus(-differences) * weights
    counts = active_mask.sum(dim=1)
    active_nodes = counts > 0
    per_node = losses.sum(dim=1) / counts.clamp_min(1)
    return per_node[active_nodes].mean(), n_pairs


def categorical_policy_kl(
    old_logits: "torch.Tensor",
    new_logits: "torch.Tensor",
    action_mask: Optional["torch.Tensor"] = None,
) -> "torch.Tensor":
    """Exact KL(pi_old || pi_new), one scalar per state; old is detached."""
    import torch

    if old_logits.shape != new_logits.shape or old_logits.ndim != 2:
        raise ValueError("Old and new logits must have matching [B, R] shapes")
    mask = torch.ones_like(old_logits, dtype=torch.bool) if action_mask is None else action_mask.bool()
    if mask.shape != old_logits.shape or not bool(mask.any(dim=1).all()):
        raise ValueError("Each policy state needs at least one feasible candidate")
    old_logp = old_logits.detach().masked_fill(~mask, -torch.inf).log_softmax(dim=-1)
    new_logp = new_logits.masked_fill(~mask, -torch.inf).log_softmax(dim=-1)
    difference = old_logp.masked_fill(~mask, 0.0) - new_logp.masked_fill(~mask, 0.0)
    return (old_logp.exp() * difference).sum(dim=-1)
