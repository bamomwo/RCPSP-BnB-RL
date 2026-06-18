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

Two-channel (decoupled) backup: the per-node reward splits into a COST channel
(-alpha per node, a LOCAL quantity that may be depth-discounted) and an
INCUMBENT-BONUS channel (progress bonuses on incumbent nodes, a PATH quantity
kept undiscounted so it reaches every ancestor at full strength). Because the
backup is LINEAR in the reward, the total return is just the sum of the two
channels' returns, each backed up with its own gamma — see
compute_episode_advantages_decoupled. At equal gammas this is identical to the
single-channel path.

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


def make_cost_reward_fn(*, alpha: float) -> RewardFn:
    """
    The COST channel of the per-node reward (Machine 1): a flat search cost
    charged once per node.

        c(n) = -alpha   for every node

    Summed over a subtree this gives -alpha * |subtree(X)|, the true objective
    (prove optimality in the fewest nodes). This is the channel that may be
    depth-discounted (gamma_cost < 1): the cost of exploring a subtree is a
    LOCAL quantity incurred at various depths below X, so "how far below X" is a
    meaningful distance and discounting it is coherent (a locality knob).
    """
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
    The INCUMBENT-BONUS channel of the per-node reward (Machine 1): positive
    progress bonuses placed on incumbent nodes, and ZERO everywhere else. This
    channel carries NO -alpha (that is the cost channel's job).

    Two components (each placed on a single incumbent node so propagation credits
    exactly the decisions on that node's root->n path):

      1. Strong first incumbent (ONE-TIME) — the first incumbent node:
             b(first) = beta1 * (root_lb / first_incumbent_makespan)
         Rewards the QUALITY of the first complete solution (1.0 if it is already
         provably optimal, i.e. equals the root lower bound). No earliness term.

      2. Incumbent improvement (SPARSE) — each LATER improving incumbent node:
             b(inc) = beta2 * (prev_incumbent - new_incumbent) / prev_incumbent
         Rewards the relative size of each makespan improvement after the first.

    Why this channel is kept UNDISCOUNTED (gamma_bonus = 1.0): an incumbent is a
    PATH quantity. The specific schedule at a leaf L is the joint product of the
    entire chain of decisions root->L; every decision on the path was equally
    necessary to construct it (pruning never discards the optimum, so X's child
    ORDER does not change WHICH solutions live below X, only how cheaply they are
    reached). There is no meaningful "distance" over which to decay it, so the
    bonus should be felt at FULL strength by every ancestor decision.

    The tree must be env.search_tree() output: nodes carry `is_incumbent`
    (solution nodes that strictly improved the best makespan, in ascending
    node-id order) and `makespan` (set on solution nodes). root_lb is the root
    node's lower bound; if None, the first-incumbent strength term is skipped.

    Returns a reward_fn(node) -> float closure with the per-node bonuses
    precomputed (O(1) per lookup).
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
    """
    Combined single-channel reward r(n) = cost(n) + bonus(n) = -alpha + bonus.

    This is the SINGLE-GAMMA path: cost and bonus are glued into one number, so
    one discount in compute_subtree_returns hits both. Use this only when you
    want both channels discounted identically (e.g. gamma = 1.0, where it is
    exactly equivalent to the decoupled backup). For independent discounts —
    cost discountable, bonus undiscounted — build the two channels separately
    (make_cost_reward_fn / make_bonus_reward_fn) and use
    compute_episode_advantages_decoupled.
    """
    cost_fn = make_cost_reward_fn(alpha=alpha)
    bonus_fn = make_bonus_reward_fn(tree, beta1=beta1, beta2=beta2, root_lb=root_lb)

    def reward_fn(node: Mapping[str, object]) -> float:
        return cost_fn(node) + bonus_fn(node)

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
    Decoupled (two-channel) subtree backup: cost and incumbent-bonus rewards are
    backed up SEPARATELY, each with its own discount, then summed.

    The backup G(X) = r(X) + gamma * sum_children G(c) is LINEAR in r, so
    splitting r(n) = cost(n) + bonus(n) splits the return exactly:

        G(X) = G_cost(X; gamma_cost)  +  G_bonus(X; gamma_bonus)

    with NO loss — it is just regrouping the same sum. Decoupling lets the two
    channels carry different discounts, which is the whole point:

      - COST is a LOCAL quantity (search incurred at various depths below X), so
        "distance below X" is meaningful and gamma_cost < 1 is a coherent
        locality / variance knob. The cost sum is also where almost all the
        return variance lives (thousands of -alpha nodes), so this is exactly
        where a discount earns its keep.

      - The INCUMBENT BONUS is a PATH quantity: a leaf's schedule is the joint
        product of the whole root->leaf decision chain, so every ancestor was
        equally necessary and there is no distance to decay over. gamma_bonus is
        kept at 1.0 so each bonus reaches every ancestor decision at FULL
        strength (low variance, high meaning — not what we want to shrink).

    At gamma_cost == gamma_bonus this reduces EXACTLY to the single-channel
    compute_episode_advantages with reward_fn = cost + bonus (the linearity makes
    the two backups' sum identical to one backup over the summed reward).

    `closed` is a property of tree STRUCTURE, not of the rewards, so it is
    identical across channels — we take it from the cost backup. A transition is
    valid iff its decision-node subtree closed (or keep_open=True).

    Parameters
    ----------
    tree           : the episode search tree (env.search_tree() output), or None.
    node_ids       : per transition, the decision node id (from the buffer).
    values         : per transition, the critic value V(s_t).
    cost_reward_fn : per-node cost c(n) (e.g. make_cost_reward_fn).
    bonus_reward_fn: per-node bonus b(n) (e.g. make_bonus_reward_fn).
    gamma_cost     : discount for the cost channel (1.0 = undiscounted).
    gamma_bonus    : discount for the bonus channel (1.0 = undiscounted; the
                     recommended default — see above).
    keep_open      : truncation policy. False (default) drops transitions whose
                     decision-node subtree never closed (partial G).

    Returns
    -------
    TreeAdvantages with advantages = G_cost + G_bonus - V, returns = G_cost +
    G_bonus, valid aligned to the transitions.
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
        # `closed` is structural and channel-independent; use the cost backup's.
        valid[t] = bool(cost_res.closed.get(nid, False)) or keep_open

    return TreeAdvantages(advantages=advantages, returns=returns, valid=valid)
