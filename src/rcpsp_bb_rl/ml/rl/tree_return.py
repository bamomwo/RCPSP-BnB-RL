"""
Closure-based (subtree) return backup for the B&B branching policy.

This is the credit-assignment backbone ("Machine 2") for the tree-structured
MDP. It REPLACES linear GAE: instead of walking backward over a flat step
sequence, it sums each node's immediate reward over its own subtree, so the
return assigned to a branching decision at node X is exactly the realized
consequence of X's subtree — no sideways leak from siblings (Obstacle 1), and
the inherited-luck confound (Obstacle 2) is handled downstream by subtracting
the critic baseline V(X) to form the advantage.

Backup rule (post-order):

    G(X) = r(X) + gamma * sum over children c of X of G(c)

with gamma = 1.0 by default (undiscounted Monte-Carlo subtree return — faithful
to the concept). gamma < 1 down-weights deeper nodes if variance near the root
becomes a problem; it is exposed as a knob, not baked in.

This module is deliberately REWARD-AGNOSTIC: it takes a per-node reward callable
`reward_fn(node) -> float`. The actual reward terms are designed separately; the
backbone only needs "what is each node worth" to produce returns.

Truncation: on a time-limited episode some subtrees never closed (they contain
"pending" frontier nodes), so their G is a partial sum. This module does NOT
decide what to do about that — it reports a `closed` flag per node (True iff the
subtree contains no pending descendant). The training loop applies the
truncation policy (e.g. drop open subtrees, or bootstrap their ancestors with
V). Keeping that choice out of the pure backup keeps this unit-testable.

Tree input shape (matches BranchingEnv.search_tree()):

    {
      "nodes": [ {id, parent_id, depth, status, is_incumbent, makespan}, ... ],
      "edges": [ (parent_id, child_id), ... ],   # unused here; parent_id suffices
      "root_id": <int or None>,
    }

status in {"pending", "expanded", "pruned", "solution"}.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Mapping, Optional

# A per-node reward function: maps one node dict to its immediate reward r(n).
RewardFn = Callable[[Mapping[str, object]], float]


def make_node_reward_fn(
    tree: Optional[Mapping[str, object]],
    *,
    alpha: float,
    beta1: float = 0.0,
    beta2: float = 0.0,
    root_lb: Optional[float] = None,
) -> RewardFn:
    """
    Build the per-node immediate reward r(n) for one episode's search tree
    (Machine 1). This stays SEPARATE from the backup (Machine 2): it just
    decides what each node is worth; compute_subtree_returns sums it.

    Three components (all placed on a single node so propagation credits exactly
    the decisions on that node's root->n path):

      1. Node cost (DENSE) — every node:
             r(n) -= alpha
         => G(X) contribution -alpha * |subtree(X)|. This is the true objective
            (prove optimality in fewest nodes). Earliness of a good incumbent is
            paid HERE automatically (good incumbent -> more pruning -> smaller
            subtree -> less negative cost), so the bonuses below do NOT include
            any node-count normalisation.

      2. Strong first incumbent (ONE-TIME) — the first incumbent node:
             r(first) += beta1 * (root_lb / first_incumbent_makespan)
         Rewards the QUALITY of the first complete solution (1.0 if it is already
         provably optimal, i.e. equals the root lower bound). No earliness term.

      3. Incumbent improvement (SPARSE) — each LATER improving incumbent node:
             r(inc) += beta2 * (prev_incumbent - new_incumbent) / prev_incumbent
         Rewards the relative size of each makespan improvement after the first.

    The tree must be env.search_tree() output: nodes carry `is_incumbent`
    (solution nodes that strictly improved the best makespan, in ascending
    node-id order) and `makespan` (set on solution nodes). root_lb is the root
    node's lower bound; if None, the first-incumbent strength term is skipped.

    Returns a reward_fn(node) -> float closure with the per-node bonuses
    precomputed (O(1) per lookup), so it plugs straight into
    compute_subtree_returns / compute_episode_advantages.
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
            # Component 2: strength of the FIRST incumbent.
            if beta1 and root_lb is not None and mk > 0:
                bonus[nid] = bonus.get(nid, 0.0) + beta1 * (float(root_lb) / mk)
        else:
            # Component 3: relative improvement over the previous incumbent.
            if beta2 and prev_mk is not None and prev_mk > 0:
                bonus[nid] = bonus.get(nid, 0.0) + beta2 * ((prev_mk - mk) / prev_mk)
        prev_mk = mk

    def reward_fn(node: Mapping[str, object]) -> float:
        r = -alpha
        nid = int(node["id"])  # type: ignore[index]
        if nid in bonus:
            r += bonus[nid]
        return r

    return reward_fn


@dataclass
class TreeReturns:
    """
    Result of the subtree backup.

    G       : node_id -> subtree return G(node) (post-order sum of rewards).
    reward  : node_id -> immediate per-node reward r(node) (kept for debugging
              and sanity checks).
    closed  : node_id -> True iff the node's subtree is fully explored, i.e. it
              contains no "pending" (frontier) descendant and the node itself is
              not pending. A decision's return G(node) is only complete when
              closed[node] is True.
    """
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

    Implementation note: we fold children into parents in order of DECREASING
    depth. Because a child's depth is always parent.depth + 1, processing the
    deepest nodes first guarantees every node's children are fully accumulated
    before that node is folded into its own parent. This is an O(N log N)
    iterative backup (the log N is the depth sort) — no recursion, so it is safe
    on the deep / large trees B&B produces (10k+ nodes).

    Returns a TreeReturns with G, reward, and closed maps keyed by node id.
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
    """
    Per-transition training signal produced from per-episode subtree backups.

    All three lists are aligned with the flat transition buffer (index t):
      advantages[t] : G(node_t) - V(s_t)   (the PPO advantage)
      returns[t]    : G(node_t)            (the value-head target)
      valid[t]      : whether transition t should be used in the update. Under
                      the "drop open subtrees" truncation policy this is False
                      for any decision whose subtree never closed (its G is a
                      partial sum), and True otherwise.
    """
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
    Tree analogue of GAE: assign each decision transition the subtree return of
    the node it was made at, using one post-order backup per episode.

    This REPLACES compute_gae. It does not bootstrap across a flat step sequence;
    a decision's return is exactly the realized consequence of its own subtree.

    Parameters
    ----------
    trees         : per-episode search trees (env.search_tree() output), indexed
                    by the episode_index values. An entry may be None if a tree
                    was unavailable for that episode (e.g. invalid_action with no
                    materialised result) — its transitions are marked invalid.
    episode_index : per transition, which episode it belongs to (index into
                    `trees`).
    node_ids      : per transition, the decision node id (from the buffer).
    values        : per transition, the critic value V(s_t).
    reward_fn     : per-node immediate reward r(n). REWARD-AGNOSTIC backbone.
    gamma         : subtree discount (1.0 = undiscounted).
    keep_open     : truncation policy. False (default) drops transitions whose
                    decision-node subtree never closed (partial G). True keeps
                    them with their partial G (no bootstrap) — exposed for
                    experimentation, not recommended as a default.

    Returns
    -------
    TreeAdvantages with advantages, returns, valid aligned to the transitions.
    """
    # One backup per distinct episode, cached so we don't recompute per
    # transition.
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
    """
    Single-episode convenience wrapper around compute_tree_advantages.

    Design A uses one episode per rollout, so all transitions share one tree and
    no episode-index bookkeeping is needed. This forwards a constant episode
    index of 0 and a one-element trees list.
    """
    return compute_tree_advantages(
        trees=[tree],
        episode_index=[0] * len(node_ids),
        node_ids=node_ids,
        values=values,
        reward_fn=reward_fn,
        gamma=gamma,
        keep_open=keep_open,
    )
