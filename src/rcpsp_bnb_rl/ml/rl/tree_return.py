"""
Tree backups for the B&B branching policy.

Single-channel reward: every expanded node costs r(v) = -1.0; pruned and
solution leaves cost nothing, so G(root) = -(nodes expanded).

Tree-GAE (post-order lambda-return backup, raw node units):
    delta(v) = r(v) + gamma * sum_c V(c) - V(v)
    A(v)     = delta(v) + gamma * lambda * sum_c A(c)
    Y(v)     = V(v) + A(v)
with V = 0 for pruned/solution leaves and V = supplied boundary value for
pending (truncated) leaves. Every expanded node receives a target whether or
not its subtree closed; ``closed`` is kept as a diagnostic only. At lambda = 1
the recursion telescopes to the Monte Carlo return: A(v) = G(v) - V(v).

``compute_subtree_returns`` is the plain Monte Carlo backup G(v), kept as the
reference that the lambda = 1 Tree-GAE must reproduce.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Mapping, Optional, Sequence

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
    closed: List[bool]  # diagnostic: subtree had no pending descendant


@dataclass
class TreeGAE:
    """Per-node Tree-GAE quantities, all in raw (node-count) units."""
    values: Dict[int, float]       # V_old(v) for every node, boundaries included
    deltas: Dict[int, float]       # 1-step tree TD residual, expanded nodes only
    advantages: Dict[int, float]   # A(v), expanded nodes only
    returns: Dict[int, float]      # Y(v) = V(v) + A(v), expanded nodes only
    closed: Dict[int, bool]        # diagnostic: no pending descendant

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
            adv = delta + gamma * lam * child_adv_sum[nid]
            deltas[nid] = delta
            advantages[nid] = adv
            returns[nid] = V[nid] + adv
        else:
            adv = 0.0  # leaves and pending nodes carry no branching decision
        pid = parent_of[nid]
        if pid is None:
            continue
        child_value_sum[pid] += V[nid]
        child_adv_sum[pid] += adv
        closed[pid] = closed[pid] and closed[nid]

    return TreeGAE(
        values=V, deltas=deltas, advantages=advantages, returns=returns, closed=closed,
    )
