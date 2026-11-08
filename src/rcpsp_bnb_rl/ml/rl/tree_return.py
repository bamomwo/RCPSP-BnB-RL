"""
Tree backups for the B&B branching policy.

Single-channel reward: every expanded node costs r(v) = -1.0; pruned and
solution leaves cost nothing, so G(root) = -(nodes expanded).

Tree-GAE (post-order lambda-return backup, raw node units):
    delta(v) = r(v) + gamma * sum_c V(c) - V(v)
    A(v)     = delta(v) + gamma * sum_c lambda_c * A(c)
    Y(v)     = V(v) + A(v)
with V = 0 for pruned/solution leaves and V = supplied boundary value for
pending (truncated) leaves. lambda_c = 1 when child c's subtree is closed
(its return is observed exactly, so Y(c) = G(c) with no critic blending) and
lambda_c = lambda on open branches. Every expanded node receives a target
whether or not its subtree closed. At lambda = 1 the recursion telescopes to
the Monte Carlo return everywhere: A(v) = G(v) - V(v).

``compute_subtree_returns`` is the plain Monte Carlo backup G(v), kept as the
reference that the lambda = 1 Tree-GAE must reproduce.

``compute_multidive_returns`` is the multi-dive primal backup: explored nodes
only, terminal makespan/LB gap to the root anchor, step cost counted up to the
first best leaf. It needs no frontier values at all.
"""

from __future__ import annotations

import bisect
import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

if TYPE_CHECKING:  # torch is only needed by split_ranking_loss
    import torch

# A per-node reward function: maps one node dict to its immediate reward r(n).
RewardFn = Callable[[Mapping[str, object]], float]


@dataclass
class TreeReturns:
    """Subtree returns and separate solver-closure/target-availability flags."""
    G: Dict[int, float]
    reward: Dict[int, float]
    closed: Dict[int, bool]
    available: Dict[int, bool] = field(default_factory=dict)
    bootstrapped: Dict[int, bool] = field(default_factory=dict)
    bootstrap_contributions: Dict[int, float] = field(default_factory=dict)


def compute_subtree_returns(
    tree: Mapping[str, object],
    reward_fn: RewardFn,
    gamma: float = 1.0,
    *,
    frontier_values: Optional[Mapping[int, float]] = None,
) -> TreeReturns:
    """
    Post-order backup of per-node rewards over the search tree.
    Processes nodes by decreasing depth (iterative, no recursion). O(N log N).

    ``frontier_values`` contains raw remaining-subtree returns for pending
    leaves, added once without an extra expansion reward (unit discounting
    only). ``closed`` describes observed solver closure; ``available`` also
    accepts subtrees whose entire unresolved frontier has a supplied value.
    """
    nodes: List[Mapping[str, object]] = list(tree.get("nodes", []))  # type: ignore[arg-type]
    boundary = dict(frontier_values or {})
    if boundary and gamma != 1.0:
        raise ValueError("Frontier bootstrapping requires gamma=1.0.")
    if boundary:
        nodes_by_id = {int(n["id"]): n for n in nodes}  # type: ignore[index]
        parent_ids = {
            int(n["parent_id"]) for n in nodes if n.get("parent_id") is not None
        }
        for nid, value in boundary.items():
            node = nodes_by_id.get(nid)
            if node is None:
                raise ValueError(f"Frontier node {nid!r} does not exist in the tree.")
            if node.get("status") != "pending" or nid in parent_ids:
                raise ValueError(f"Frontier node {nid!r} must be a pending leaf.")
            if not math.isfinite(value):
                raise ValueError(f"Frontier value for node {nid!r} must be finite.")

    # Seed: each node's G starts at its own immediate reward; closed starts True
    # unless the node itself is an unexplored frontier node.
    G: Dict[int, float] = {}
    reward: Dict[int, float] = {}
    closed: Dict[int, bool] = {}
    available: Dict[int, bool] = {}
    bootstrapped: Dict[int, bool] = {}
    bootstrap_contributions: Dict[int, float] = {}
    parent_of: Dict[int, Optional[int]] = {}
    for n in nodes:
        nid = int(n["id"])  # type: ignore[index]
        r = float(reward_fn(n))
        reward[nid] = r
        contribution = float(boundary.get(nid, 0.0))
        G[nid] = r + contribution
        closed[nid] = (n.get("status") != "pending")
        available[nid] = closed[nid] or nid in boundary
        bootstrapped[nid] = nid in boundary
        bootstrap_contributions[nid] = contribution
        pid = n.get("parent_id")
        parent_of[nid] = None if pid is None else int(pid)  # type: ignore[arg-type]

    # Fold children into parents, deepest first.
    by_depth_desc = sorted(nodes, key=lambda n: int(n["depth"]), reverse=True)  # type: ignore[index]
    for n in by_depth_desc:
        nid = int(n["id"])  # type: ignore[index]
        pid = parent_of[nid]
        if pid is None:
            continue  # root: nothing to fold into
        G[pid] += gamma * G[nid]
        # A parent's subtree is closed only if every child subtree is closed
        # (and the parent itself is not pending, already in its seed).
        closed[pid] = closed[pid] and closed[nid]
        available[pid] = available[pid] and available[nid]
        bootstrapped[pid] = bootstrapped[pid] or bootstrapped[nid]
        bootstrap_contributions[pid] += gamma * bootstrap_contributions[nid]

    return TreeReturns(
        G=G, reward=reward, closed=closed, available=available,
        bootstrapped=bootstrapped,
        bootstrap_contributions=bootstrap_contributions,
    )


# ---------------------------------------------------------------------------
# Single-channel Tree-GAE
# ---------------------------------------------------------------------------

EXPANSION_REWARD = -1.0


def expansion_reward(node: Mapping[str, object]) -> float:
    """Unit cost per expanded node; pruned, solution and pending nodes are free."""
    return EXPANSION_REWARD if node.get("status") == "expanded" else 0.0


@dataclass
class TransitionTargets:
    """Tree-GAE quantities aligned with a transition buffer (raw node units)."""
    advantages: List[float]
    returns: List[float]
    values: List[float]
    valid: List[bool]   # transition's node is an expanded node of the tree
    closed: List[bool]  # subtree had no pending descendant: return is exact


@dataclass
class TreeGAE:
    """Per-node Tree-GAE quantities, all in raw (node-count) units."""
    values: Dict[int, float]       # V_old(v) for every node, boundaries included
    deltas: Dict[int, float]       # 1-step tree TD residual, expanded nodes only
    advantages: Dict[int, float]   # A(v), expanded nodes only
    returns: Dict[int, float]      # Y(v) = V(v) + A(v), expanded nodes only
    closed: Dict[int, bool]        # no pending descendant: Y(v) is the exact G(v)

    def for_transitions(self, node_ids: Sequence[Optional[int]]) -> TransitionTargets:
        """Align per-node quantities with the transitions that branched at them."""
        T = len(node_ids)
        out = TransitionTargets(
            advantages=[0.0] * T, returns=[0.0] * T, values=[0.0] * T,
            valid=[False] * T, closed=[False] * T,
        )
        for t, nid in enumerate(node_ids):
            if nid is None or nid not in self.advantages:
                continue
            out.advantages[t] = self.advantages[nid]
            out.returns[t] = self.returns[nid]
            out.values[t] = self.values[nid]
            out.valid[t] = True
            out.closed[t] = self.closed[nid]
        return out


def compute_tree_gae(
    tree: Mapping[str, object],
    *,
    values: Mapping[int, float],
    frontier_values: Optional[Mapping[int, float]] = None,
    gamma: float = 1.0,
    lam: float = 0.95,
) -> TreeGAE:
    """
    Post-order Tree-GAE over the search tree (iterative, deepest first).

    ``values`` holds the raw critic estimate V_old(v) of every EXPANDED node,
    evaluated when the node was popped. ``frontier_values`` holds the raw
    boundary value of every PENDING leaf (critic bootstrap, or an exact
    0 / -1 for prunable / expansion-only states). Pruned and solution leaves
    are worth 0. A missing value is an error: Tree-GAE needs full coverage of
    every child to form a parent's residual.

    Closed children are folded with lambda = 1 so their observed return enters
    the parent exactly; critic error only reaches a parent through open branches.
    """
    if not math.isfinite(gamma) or not 0.0 < gamma <= 1.0:
        raise ValueError("gamma must be in (0, 1].")
    if not math.isfinite(lam) or not 0.0 <= lam <= 1.0:
        raise ValueError("lam must be in [0, 1].")

    nodes: List[Mapping[str, object]] = list(tree.get("nodes", []))  # type: ignore[arg-type]
    boundary = dict(frontier_values or {})

    V: Dict[int, float] = {}
    parent_of: Dict[int, Optional[int]] = {}
    status_of: Dict[int, str] = {}
    for n in nodes:
        nid = int(n["id"])  # type: ignore[index]
        status = str(n.get("status"))
        if status == "expanded":
            if nid not in values:
                raise ValueError(f"Expanded node {nid!r} has no critic value.")
            v = float(values[nid])
        elif status == "pending":
            if nid not in boundary:
                raise ValueError(f"Pending node {nid!r} has no boundary value.")
            v = float(boundary[nid])
        else:
            v = 0.0
        if not math.isfinite(v):
            raise ValueError(f"Value for node {nid!r} must be finite.")
        V[nid] = v
        status_of[nid] = status
        pid = n.get("parent_id")
        parent_of[nid] = None if pid is None else int(pid)  # type: ignore[arg-type]
    for nid in boundary:
        if status_of.get(nid) != "pending":
            raise ValueError(f"Frontier node {nid!r} must be a pending leaf.")

    child_value_sum: Dict[int, float] = {nid: 0.0 for nid in V}
    child_adv_sum: Dict[int, float] = {nid: 0.0 for nid in V}
    closed: Dict[int, bool] = {nid: status_of[nid] != "pending" for nid in V}
    deltas: Dict[int, float] = {}
    advantages: Dict[int, float] = {}
    returns: Dict[int, float] = {}

    # Deepest first: every child is folded before its parent is visited.
    by_depth_desc = sorted(nodes, key=lambda n: int(n["depth"]), reverse=True)  # type: ignore[index]
    for n in by_depth_desc:
        nid = int(n["id"])  # type: ignore[index]
        if status_of[nid] == "expanded":
            delta = EXPANSION_REWARD + gamma * child_value_sum[nid] - V[nid]
            adv = delta + gamma * child_adv_sum[nid]
            deltas[nid] = delta
            advantages[nid] = adv
            returns[nid] = V[nid] + adv
        else:
            adv = 0.0  # leaves and pending nodes carry no branching decision
        pid = parent_of[nid]
        if pid is None:
            continue
        # closed[nid] is final here: every descendant is deeper and already folded.
        child_value_sum[pid] += V[nid]
        child_adv_sum[pid] += (1.0 if closed[nid] else lam) * adv
        closed[pid] = closed[pid] and closed[nid]

    return TreeGAE(
        values=V, deltas=deltas, advantages=advantages, returns=returns, closed=closed,
    )


# ---------------------------------------------------------------------------
# Multi-dive primal backup (explored nodes only)
# ---------------------------------------------------------------------------

EXPLORED_STATUSES = frozenset({"expanded", "solution", "pruned"})


@dataclass
class MultiDiveReturns:
    """Bottom-up multi-dive quantities over EXPLORED nodes (pending ignored).

    Every quantity is keyed by node id and defined for every explored node.
    ``split_children`` lists, for each expanded node with >= 2 explored
    children, those children in creation order (policy rank order), so the
    split-node ranking loss can score them by ``q_sub``.
    """
    best_term_r: Dict[int, float]     # best terminal score inside the subtree
    best_leaf_step: Dict[int, int]    # visit order at which that score was first reached
    steps_to_best: Dict[int, int]     # expanded nodes in the subtree visited on/before it
    q_sub: Dict[int, float]           # best_term_r - alpha_step * steps_to_best
    explored_children: Dict[int, List[int]]  # expanded node -> explored children
    split_children: Dict[int, List[int]]     # subset with >= 2 explored children
    alpha_step: float
    # Expanded nodes with no explored child, scored by their LB ceiling:
    n_dead_end: int = 0               # no children at all (every child dominance-pruned)
    n_cut: int = 0                    # pending children only (plunge cut by the budget/clock)

    def targets_for_transitions(
        self, node_ids: Sequence[Optional[int]],
    ) -> Tuple[List[float], List[bool]]:
        """Critic target Y(v) = q_sub(v) aligned with a transition buffer."""
        targets = [0.0] * len(node_ids)
        valid = [False] * len(node_ids)
        for t, nid in enumerate(node_ids):
            if nid is not None and nid in self.explored_children:
                targets[t] = self.q_sub[nid]
                valid[t] = True
        return targets, valid


def terminal_gap_score(value: float, lb_root: float) -> float:
    """-(value - LB_root) / LB_root, clipped at 0 for values below the anchor."""
    anchor = max(1.0, float(lb_root))
    return -max(0.0, (float(value) - float(lb_root)) / anchor)


def compute_multidive_returns(
    tree: Mapping[str, object],
    *,
    lb_root: float,
    n_activities: int,
    alpha: float = 0.01,
) -> MultiDiveReturns:
    """
    Post-order multi-dive backup over the explored tree (deepest first).

    Node dicts need ``id, parent_id, depth, status, visit_order, lower_bound``
    and ``makespan`` on solution leaves. Pending nodes are dropped: they cost
    nothing and are never bootstrapped. Leaves score their terminal gap to the
    root anchor; an expanded node takes the best score among its explored
    children and pays ``alpha / n_activities`` per expanded node of its subtree
    visited no later than the first leaf that achieved that score, so effort
    spent after the best leaf (budget-share artefacts of DFS order) is free.
    An expanded node with no explored child is a failed dive. If it still has
    pending children the plunge was cut mid-dive (``n_cut``; dive-boundary
    stopping prevents this) and it falls back to its lower-bound ceiling. If it
    has no children at all, every child was dominance-pruned (``n_dead_end``):
    it is scored at the incumbent in force when it was visited, like an
    LB-prune at the bound, so a dominated subtree never outranks a real
    schedule. Its LB ceiling is used only when no incumbent existed yet.
    """
    if not math.isfinite(alpha) or alpha < 0.0:
        raise ValueError("alpha must be a finite non-negative number.")
    if n_activities <= 0:
        raise ValueError("n_activities must be positive.")
    alpha_step = float(alpha) / float(n_activities)

    nodes: List[Mapping[str, object]] = [
        n for n in tree.get("nodes", []) if str(n.get("status")) in EXPLORED_STATUSES  # type: ignore[union-attr]
    ]
    by_id: Dict[int, Mapping[str, object]] = {}
    visit: Dict[int, int] = {}
    for n in nodes:
        nid = int(n["id"])  # type: ignore[index]
        vo = n.get("visit_order")
        if vo is None:
            raise ValueError(f"Explored node {nid!r} has no visit_order.")
        by_id[nid] = n
        visit[nid] = int(vo)

    explored_children: Dict[int, List[int]] = {}
    has_pending_child: set[int] = set()
    for n in tree.get("nodes", []):  # type: ignore[union-attr]
        pid = n.get("parent_id")
        if pid is None or int(pid) not in by_id:
            continue
        if str(n.get("status")) in EXPLORED_STATUSES:
            explored_children.setdefault(int(pid), []).append(int(n["id"]))  # type: ignore[index]
        else:
            has_pending_child.add(int(pid))
    for cs in explored_children.values():
        cs.sort()  # creation order == policy rank order

    best_term_r: Dict[int, float] = {}
    best_leaf_step: Dict[int, int] = {}
    steps_to_best: Dict[int, int] = {}
    q_sub: Dict[int, float] = {}
    # Sorted visit orders of expanded nodes inside each subtree (freed on fold).
    subtree_visits: Dict[int, List[int]] = {}
    n_dead_end = n_cut = 0

    # Incumbent in force at a given visit: prefix-min makespan over earlier solutions.
    sol_visits = sorted(
        (visit[int(n["id"])], float(n["makespan"]))  # type: ignore[index, arg-type]
        for n in nodes if str(n.get("status")) == "solution" and n.get("makespan") is not None
    )
    sol_orders = [v for v, _ in sol_visits]
    prefix_min: List[float] = []
    for _, mk in sol_visits:
        prefix_min.append(mk if not prefix_min else min(prefix_min[-1], mk))

    def incumbent_before(visit_order: int) -> Optional[float]:
        k = bisect.bisect_left(sol_orders, visit_order)
        return prefix_min[k - 1] if k > 0 else None

    by_depth_desc = sorted(nodes, key=lambda n: int(n["depth"]), reverse=True)  # type: ignore[index]
    for n in by_depth_desc:
        nid = int(n["id"])  # type: ignore[index]
        status = str(n.get("status"))
        if status == "solution":
            mk = n.get("makespan")
            if mk is None:
                raise ValueError(f"Solution node {nid!r} has no makespan.")
            best_term_r[nid] = terminal_gap_score(float(mk), lb_root)
            best_leaf_step[nid] = visit[nid]
            steps_to_best[nid] = 0
            q_sub[nid] = best_term_r[nid]
            subtree_visits[nid] = []
            continue
        if status == "pruned":
            best_term_r[nid] = terminal_gap_score(float(n["lower_bound"]), lb_root)  # type: ignore[index]
            best_leaf_step[nid] = visit[nid]
            steps_to_best[nid] = 0
            q_sub[nid] = best_term_r[nid]
            subtree_visits[nid] = []
            continue

        children = explored_children.get(nid, [])
        explored_children.setdefault(nid, children)
        merged: List[int] = [visit[nid]]
        for c in children:
            merged.extend(subtree_visits.pop(c))
        merged.sort()
        subtree_visits[nid] = merged
        if children:
            best = max(best_term_r[c] for c in children)
            step = min(best_leaf_step[c] for c in children if best_term_r[c] == best)
        else:
            lb_v = float(n["lower_bound"])  # type: ignore[index]
            if nid in has_pending_child:
                n_cut += 1
                best = terminal_gap_score(lb_v, lb_root)
            else:
                n_dead_end += 1
                inc = incumbent_before(visit[nid])
                best = terminal_gap_score(lb_v if inc is None else max(inc, lb_v), lb_root)
            step = visit[nid]
        best_term_r[nid] = best
        best_leaf_step[nid] = step
        steps_to_best[nid] = bisect.bisect_right(merged, step)
        q_sub[nid] = best - alpha_step * steps_to_best[nid]

    split_children = {v: cs for v, cs in explored_children.items() if len(cs) >= 2}
    return MultiDiveReturns(
        best_term_r=best_term_r, best_leaf_step=best_leaf_step,
        steps_to_best=steps_to_best, q_sub=q_sub,
        explored_children=explored_children, split_children=split_children,
        alpha_step=alpha_step, n_dead_end=n_dead_end, n_cut=n_cut,
    )


# ---------------------------------------------------------------------------
# Split-node hindsight ranking loss
# ---------------------------------------------------------------------------

def split_ranking_loss(
    logits: "torch.Tensor",
    child_slots: "torch.Tensor",
    child_scores: "torch.Tensor",
    child_mask: "torch.Tensor",
    row_weight: Optional["torch.Tensor"] = None,
) -> Tuple["torch.Tensor", int]:
    """
    Pairwise hindsight ranking loss over explored children of split nodes.

    logits       [B, R]   current policy logits of the node (padded)
    child_slots  [B, M]   candidate index of each explored child (padded)
    child_scores [B, M]   Q_sub of each explored child
    child_mask   [B, M]   True where a child slot is real
    row_weight   [B]      optional per-node weight (e.g. 0 for non-actor nodes)

    For every ordered pair (i, j) of a node's explored children with
    score_i > score_j the loss is (score_i - score_j) * softplus(logit_j -
    logit_i), so the gradient pushes the better child's logit above the worse
    one's, proportionally to how much better it did. Returns the mean over
    valid pairs and the number of valid pairs.
    """
    import torch
    import torch.nn.functional as F

    B, M = child_slots.shape
    if B == 0 or M < 2:
        return logits.new_zeros(()), 0
    safe_slots = child_slots.clamp(min=0)
    child_logits = torch.gather(logits, 1, safe_slots)           # [B, M]
    diff_score = child_scores.unsqueeze(2) - child_scores.unsqueeze(1)   # [B, M, M] s_i - s_j
    diff_logit = child_logits.unsqueeze(2) - child_logits.unsqueeze(1)   # [B, M, M] l_i - l_j
    pair_mask = child_mask.unsqueeze(2) & child_mask.unsqueeze(1) & (diff_score > 0)
    if row_weight is not None:
        pair_mask = pair_mask & (row_weight > 0).view(B, 1, 1)
    n_pairs = int(pair_mask.sum().item())
    if n_pairs == 0:
        return logits.new_zeros(()), 0
    pair_loss = diff_score * F.softplus(-diff_logit)
    if row_weight is not None:
        pair_loss = pair_loss * row_weight.view(B, 1, 1)
    return (pair_loss * pair_mask).sum() / n_pairs, n_pairs
