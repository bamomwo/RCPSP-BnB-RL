"""
Closure-based (subtree) return backup for the B&B branching policy.

Backup rule: G(X) = r(X) + gamma * sum_children G(c)

Each decision at node X is credited with its own subtree's return — no sibling
leak. The critic baseline V(X) handles inherited-luck (inherited incumbent).

Two-channel (decoupled) backup: cost channel (-alpha per node, discountable) and
incumbent-bonus channel (undiscounted). Total return = sum of both channels.

Truncation: nodes whose subtree contains "pending" descendants are flagged as
open (closed=False). The training loop decides what to do with them.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Mapping, Optional

# A per-node reward function: maps one node dict to its immediate reward r(n).
RewardFn = Callable[[Mapping[str, object]], float]


def make_cost_reward_fn(*, alpha: float) -> RewardFn:
    """Cost channel: -alpha per node. Summed over subtree = -alpha * |subtree|."""
    def reward_fn(node: Mapping[str, object]) -> float:
        return -alpha

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
    """Combined single-channel reward: r(n) = -alpha + bonus(n). Use only with a single gamma."""
    cost_fn = make_cost_reward_fn(alpha=alpha)
    bonus_fn = make_bonus_reward_fn(tree, beta1=beta1, beta2=beta2, root_lb=root_lb)

    def reward_fn(node: Mapping[str, object]) -> float:
        return cost_fn(node) + bonus_fn(node)

    return reward_fn


@dataclass
class TreeReturns:
    """Result of subtree backup: G (returns), reward (per-node), closed (fully explored flag)."""
    G: Dict[int, float]
    reward: Dict[int, float]
    closed: Dict[int, bool]


def compute_subtree_returns(
    tree: Mapping[str, object],
    reward_fn: RewardFn,
    gamma: float = 1.0,
) -> TreeReturns:
    """
    Post-order backup of per-node rewards over the search tree.
    Processes nodes by decreasing depth (iterative, no recursion). O(N log N).
    """
    nodes: List[Mapping[str, object]] = list(tree.get("nodes", []))  # type: ignore[arg-type]

    # Seed: each node's G starts at its own immediate reward; closed starts True
    # unless the node itself is an unexplored frontier node.
    G: Dict[int, float] = {}
    reward: Dict[int, float] = {}
    closed: Dict[int, bool] = {}
    parent_of: Dict[int, Optional[int]] = {}
    for n in nodes:
        nid = int(n["id"])  # type: ignore[index]
        r = float(reward_fn(n))
        reward[nid] = r
        G[nid] = r
        closed[nid] = (n.get("status") != "pending")
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

    return TreeReturns(G=G, reward=reward, closed=closed)


@dataclass
class TreeAdvantages:
    """Per-transition training signal from subtree backups. Aligned with the transition buffer."""
    advantages: List[float]
    returns: List[float]
    valid: List[bool]


def compute_tree_advantages(
    *,
    trees: List[Optional[Mapping[str, object]]],
    episode_index: List[int],
    node_ids: List[Optional[int]],
    values: List[float],
    reward_fn: RewardFn,
    gamma: float = 1.0,
    keep_open: bool = False,
) -> TreeAdvantages:
    """
    Assign each transition the subtree return G(node) of the node it branched at.
    Advantage = G(node) - V(node). Open subtrees are invalid unless keep_open=True.
    """
    # One backup per distinct episode, cached so we don't recompute per transition.
    backups: Dict[int, Optional[TreeReturns]] = {}
    for ei in set(episode_index):
        tree = trees[ei] if 0 <= ei < len(trees) else None
        backups[ei] = (
            compute_subtree_returns(tree, reward_fn, gamma) if tree is not None else None
        )

    T = len(node_ids)
    advantages: List[float] = [0.0] * T
    returns: List[float] = [0.0] * T
    valid: List[bool] = [False] * T

    for t in range(T):
        res = backups.get(episode_index[t])
        nid = node_ids[t]
        if res is None or nid is None or nid not in res.G:
            continue  # no tree / unknown node -> leave invalid
        g = res.G[nid]
        returns[t] = g
        advantages[t] = g - values[t]
        valid[t] = bool(res.closed.get(nid, False)) or keep_open

    return TreeAdvantages(advantages=advantages, returns=returns, valid=valid)


def compute_episode_advantages(
    *,
    tree: Optional[Mapping[str, object]],
    node_ids: List[Optional[int]],
    values: List[float],
    reward_fn: RewardFn,
    gamma: float = 1.0,
    keep_open: bool = False,
) -> TreeAdvantages:
    """Single-episode wrapper: all transitions share one tree, episode_index=0."""
    return compute_tree_advantages(
        trees=[tree],
        episode_index=[0] * len(node_ids),
        node_ids=node_ids,
        values=values,
        reward_fn=reward_fn,
        gamma=gamma,
        keep_open=keep_open,
    )


def compute_episode_advantages_decoupled(
    *,
    tree: Optional[Mapping[str, object]],
    node_ids: List[Optional[int]],
    values: List[float],
    cost_reward_fn: RewardFn,
    bonus_reward_fn: RewardFn,
    gamma_cost: float = 1.0,
    gamma_bonus: float = 1.0,
    keep_open: bool = False,
) -> TreeAdvantages:
    """
    Decoupled two-channel backup: cost and bonus backed up separately with
    independent gammas, then summed. G(X) = G_cost(X) + G_bonus(X).
    """
    T = len(node_ids)
    advantages: List[float] = [0.0] * T
    returns: List[float] = [0.0] * T
    valid: List[bool] = [False] * T

    if tree is None:
        # No tree (e.g. invalid-action episode) -> nothing usable.
        return TreeAdvantages(advantages=advantages, returns=returns, valid=valid)

    cost_res = compute_subtree_returns(tree, cost_reward_fn, gamma_cost)
    bonus_res = compute_subtree_returns(tree, bonus_reward_fn, gamma_bonus)

    for t in range(T):
        nid = node_ids[t]
        if nid is None or nid not in cost_res.G:
            continue  # unknown node -> leave invalid
        g = cost_res.G[nid] + bonus_res.G[nid]
        returns[t] = g
        advantages[t] = g - values[t]
        valid[t] = bool(cost_res.closed.get(nid, False)) or keep_open

    return TreeAdvantages(advantages=advantages, returns=returns, valid=valid)
