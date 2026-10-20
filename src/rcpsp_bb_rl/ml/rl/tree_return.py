"""
Closure-based (subtree) return backup for the B&B branching policy.

Backup rule: G(X) = r(X) + gamma * sum_children G(c)

Each decision at node X is credited with its own subtree's return — no sibling
leak. The critic baseline V(X) handles inherited-luck (inherited incumbent).

Two-channel (decoupled) backup: cost channel (-alpha per expanded node,
discountable) and incumbent-bonus channel (undiscounted). Total return = sum of
both channels.

Truncation: nodes whose subtree contains "pending" descendants remain open
(closed=False). Optional frontier values can make their training targets
available without claiming that the solver actually explored those subtrees.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Mapping, Optional

# A per-node reward function: maps one node dict to its immediate reward r(n).
RewardFn = Callable[[Mapping[str, object]], float]


def validate_incumbent_bonus_config(
    *, beta1: float, beta2: float, gamma_bonus: float,
) -> None:
    """Validate the independently tunable incumbent-reward channels.

    ``beta1`` controls first-incumbent quality and ``beta2`` controls later
    relative makespan improvements. The bonus backup remains undiscounted so
    each incumbent event is credited consistently through its ancestors.
    """
    if not math.isfinite(beta1) or not math.isfinite(beta2) or beta1 < 0 or beta2 < 0:
        raise ValueError("beta1 and beta2 must be finite and non-negative.")
    if gamma_bonus != 1.0:
        raise ValueError(
            "tree_gamma_bonus must be 1.0 so incumbent rewards "
            "remain undiscounted."
        )


def make_cost_reward_fn(*, alpha: float) -> RewardFn:
    """Cost channel: -alpha per expanded node, zero for every other status.

    An expanded node is a branching decision that creates further search work.
    Pruned, solution, and pending nodes are deliberately free: they may incur a
    small bookkeeping cost, but are not charged by the search-expansion
    objective. This keeps the reward's effort definition aligned with
    ``BnBSolver.nodes_expanded`` used by the estimator and oracle lookup.
    """
    def reward_fn(node: Mapping[str, object]) -> float:
        return -alpha if node.get("status") == "expanded" else 0.0

    return reward_fn


def make_bonus_reward_fn(
    tree: Optional[Mapping[str, object]],
    *,
    beta1: float = 0.0,
    beta2: float = 0.0,
    root_lb: Optional[float] = None,
) -> RewardFn:
    """
    Incumbent-bonus channel: positive bonuses on incumbent nodes, zero elsewhere.

    Components:
      1. First incumbent: beta1 * (root_lb / first_incumbent_makespan)
      2. Each later improvement: beta2 * (prev - new) / prev

    This channel is kept undiscounted (gamma_bonus=1.0) since an incumbent is a
    path quantity — every ancestor contributed equally to reaching it.
    """
    nodes: List[Mapping[str, object]] = list(tree.get("nodes", [])) if tree else []  # type: ignore[arg-type]

    # Incumbent nodes in chronological (ascending node-id) order. The env marks
    # is_incumbent exactly on solution nodes that strictly improved the best
    # makespan, so this sequence is the incumbent history.
    inc_nodes = sorted(
        (n for n in nodes if n.get("is_incumbent")),
        key=lambda n: int(n["id"]),  # type: ignore[index]
    )

    bonus: Dict[int, float] = {}
    prev_mk: Optional[float] = None
    for i, n in enumerate(inc_nodes):
        nid = int(n["id"])  # type: ignore[index]
        mk = n.get("makespan")
        if mk is None:
            # Defensive: an incumbent node should always carry a makespan.
            continue
        mk = float(mk)
        if i == 0:
            # Component 1: strength of the FIRST incumbent.
            if beta1 and root_lb is not None and mk > 0:
                bonus[nid] = bonus.get(nid, 0.0) + beta1 * (float(root_lb) / mk)
        else:
            # Component 2: relative improvement over the previous incumbent.
            if beta2 and prev_mk is not None and prev_mk > 0:
                bonus[nid] = bonus.get(nid, 0.0) + beta2 * ((prev_mk - mk) / prev_mk)
        prev_mk = mk

    def reward_fn(node: Mapping[str, object]) -> float:
        return bonus.get(int(node["id"]), 0.0)  # type: ignore[index]

    return reward_fn


def make_node_reward_fn(
    tree: Optional[Mapping[str, object]],
    *,
    alpha: float,
    beta1: float = 0.0,
    beta2: float = 0.0,
    root_lb: Optional[float] = None,
) -> RewardFn:
    """Combined reward: expanded nodes get -alpha plus any incumbent bonus."""
    cost_fn = make_cost_reward_fn(alpha=alpha)
    bonus_fn = make_bonus_reward_fn(tree, beta1=beta1, beta2=beta2, root_lb=root_lb)

    def reward_fn(node: Mapping[str, object]) -> float:
        return cost_fn(node) + bonus_fn(node)

    return reward_fn


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
    leaves. Each is added once, without an extra expansion reward. Only unit
    discounting is supported for these combined cost-and-bonus estimates.
    ``closed`` always describes observed solver closure; ``available`` also
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


@dataclass
class TreeAdvantages:
    """Raw subtree-return targets aligned with the transition buffer."""
    returns: List[float]
    valid: List[bool]
    closed: List[bool] = field(default_factory=list)
    bootstrapped: List[bool] = field(default_factory=list)
    bootstrap_contributions: List[float] = field(default_factory=list)


def compute_tree_advantages(
    *,
    trees: List[Optional[Mapping[str, object]]],
    episode_index: List[int],
    node_ids: List[Optional[int]],
    reward_fn: RewardFn,
    gamma: float = 1.0,
    keep_open: bool = False,
) -> TreeAdvantages:
    """
    Assign each transition the subtree return G(node) of the node it branched at.
    Open subtrees are invalid unless keep_open=True.
    """
    # One backup per distinct episode, cached so we don't recompute per transition.
    backups: Dict[int, Optional[TreeReturns]] = {}
    for ei in set(episode_index):
        tree = trees[ei] if 0 <= ei < len(trees) else None
        backups[ei] = (
            compute_subtree_returns(tree, reward_fn, gamma) if tree is not None else None
        )

    T = len(node_ids)
    returns: List[float] = [0.0] * T
    valid: List[bool] = [False] * T
    closed: List[bool] = [False] * T

    for t in range(T):
        res = backups.get(episode_index[t])
        nid = node_ids[t]
        if res is None or nid is None or nid not in res.G:
            continue  # no tree / unknown node -> leave invalid
        g = res.G[nid]
        returns[t] = g
        closed[t] = bool(res.closed.get(nid, False))
        valid[t] = closed[t] or keep_open

    return TreeAdvantages(
        returns=returns, valid=valid, closed=closed,
        bootstrapped=[False] * T, bootstrap_contributions=[0.0] * T,
    )


def compute_episode_advantages(
    *,
    tree: Optional[Mapping[str, object]],
    node_ids: List[Optional[int]],
    reward_fn: RewardFn,
    gamma: float = 1.0,
    keep_open: bool = False,
) -> TreeAdvantages:
    """Single-episode wrapper: all transitions share one tree, episode_index=0."""
    return compute_tree_advantages(
        trees=[tree],
        episode_index=[0] * len(node_ids),
        node_ids=node_ids,
        reward_fn=reward_fn,
        gamma=gamma,
        keep_open=keep_open,
    )


def compute_episode_advantages_decoupled(
    *,
    tree: Optional[Mapping[str, object]],
    node_ids: List[Optional[int]],
    cost_reward_fn: RewardFn,
    bonus_reward_fn: RewardFn,
    gamma_cost: float = 1.0,
    gamma_bonus: float = 1.0,
    keep_open: bool = False,
    frontier_values: Optional[Mapping[int, float]] = None,
) -> TreeAdvantages:
    """
    Decoupled two-channel backup: cost and bonus backed up separately with
    independent gammas, then summed. G(X) = G_cost(X) + G_bonus(X).

    Optional frontier values predict the combined remaining return and require
    both gammas to equal one and ``keep_open=False``. A target is valid only
    after every pending descendant has a supplied boundary value. These
    estimates never change the observed closure flags.
    """
    T = len(node_ids)
    returns: List[float] = [0.0] * T
    valid: List[bool] = [False] * T
    closed: List[bool] = [False] * T
    bootstrapped: List[bool] = [False] * T
    bootstrap_contributions: List[float] = [0.0] * T
    result = TreeAdvantages(
        returns=returns, valid=valid, closed=closed,
        bootstrapped=bootstrapped,
        bootstrap_contributions=bootstrap_contributions,
    )
    if frontier_values:
        if keep_open:
            raise ValueError("Frontier bootstrapping requires keep_open=False.")
        if gamma_cost != 1.0 or gamma_bonus != 1.0:
            raise ValueError(
                "Frontier bootstrapping requires gamma_cost=gamma_bonus=1.0."
            )

    if tree is None:
        if frontier_values:
            raise ValueError("Frontier values require an observed tree.")
        # No tree (e.g. invalid-action episode) -> nothing usable.
        return result

    # Put each combined boundary value in one channel only. The bonus backup
    # still contributes every observed incumbent reward exactly once.
    cost_res = compute_subtree_returns(
        tree, cost_reward_fn, gamma_cost, frontier_values=frontier_values,
    )
    bonus_res = compute_subtree_returns(tree, bonus_reward_fn, gamma_bonus)

    for t in range(T):
        nid = node_ids[t]
        if nid is None or nid not in cost_res.G:
            continue  # unknown node -> leave invalid
        g = cost_res.G[nid] + bonus_res.G[nid]
        returns[t] = g
        closed[t] = bool(cost_res.closed.get(nid, False))
        valid[t] = bool(cost_res.available.get(nid, False)) or keep_open
        bootstrapped[t] = valid[t] and cost_res.bootstrapped.get(nid, False)
        bootstrap_contributions[t] = cost_res.bootstrap_contributions.get(nid, 0.0)

    return result
