from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch

from rcpsp_bb_rl.bnb.dominance import normalize_dominance_spec
from rcpsp_bb_rl.bnb.lower_bounds import DEFAULT_LOWER_BOUND_ID, lower_bound
from rcpsp_bb_rl.bnb.scheduling import build_profile, earliest_feasible_start
from rcpsp_bb_rl.bnb.solver import BBNode, BnBSolver, ScheduleEntry, SolverResult, StepContext, current_makespan
from rcpsp_bb_rl.data.parsing import RCPSPInstance, load_instance
from rcpsp_bb_rl.ml.il.featurize import (
    NodeContext,
    candidate_features,
    critic_features,
    global_features,
)


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class StepOutput:
    observation: Dict[str, torch.Tensor]
    reward: float
    done: bool
    info: Dict


@dataclass
class RewardConfig:
    """
    Coefficients for the env's per-step DIAGNOSTIC reward.

    IMPORTANT — this per-step reward no longer trains anything. Under the
    closure-based (subtree) credit assignment (Design A; see
    ml/rl/tree_return.py), the PPO returns and advantages are produced by a
    post-episode tree backup over the finished search tree, NOT from these
    per-step rewards. The trainer stores step_out.reward in the rollout buffer
    but never reads it in the update; it is consumed only by per-episode logging
    (episode_reward, the rwd_std diagnostic, and reward_breakdown).

    The actual training reward lives in tree_return.py:
      - make_cost_reward_fn  : -alpha per node (the cost channel), and
      - make_bonus_reward_fn : the beta1/beta2 incumbent bonuses (bonus channel),
    backed up separately and summed. Those functions read alpha/beta1/beta2 from
    the SAME config keys, so this dataclass and the tree backup stay in step, but
    the formula below is a per-advance approximation kept purely for monitoring —
    it does not have to match the tree return and is not used to compute it.

    Per-step diagnostic reward at each branching advance:

        r  = -alpha * nodes_delta                                   (always)

        if the FIRST incumbent appeared on this advance:
            quality = clamp(root_lb / first_inc, 0, 1)
            r += beta1 * quality / (1 + log1p(nodes_to_first_incumbent))

        if an EXISTING incumbent improved on this advance:
            r += beta2 * (old_inc - new_inc) / nodes_since_last_incumbent

    where
        nodes_delta                = nodes expanded during this advance
                                     (~1 per step during search; larger only on
                                     the terminal advance)
        root_lb                    = root critical-path lower bound (cp_lb)
        nodes_to_first_incumbent   = nodes expanded when the first incumbent was
                                     found
        nodes_since_last_incumbent = nodes expanded since the previous incumbent

    Coefficients (shared with the tree backup's reward channels):
        alpha  — per-node search cost. Sets the search-effort scale.
        beta1  — first-incumbent strength bonus weight.
        beta2  — incumbent-improvement bonus weight.
    """
    alpha: float = 0.01
    beta1: float = 1.0
    beta2: float = 1.0


@dataclass
class EpisodeStats:
    """Tracks per-episode statistics for logging in train_ppo.py."""
    nodes_expanded: int = 0
    nodes_pruned: int = 0
    dominance_pruned: int = 0
    incumbent_improvements: int = 0
    first_incumbent_node: Optional[int] = None
    first_incumbent_makespan: Optional[int] = None
    last_incumbent_node: Optional[int] = None
    last_incumbent_makespan: Optional[int] = None
    best_makespan: Optional[int] = None
    final_gap: Optional[float] = None
    done_reason: str = "unknown"
    total_reward: float = 0.0
    reward_breakdown: Dict[str, float] = field(default_factory=lambda: {
        "cost": 0.0,
        "first_incumbent": 0.0,
        "incumbent_improvement": 0.0,
    })


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

class BranchingEnv:
    """
    RL environment for the serial B&B branching policy.

    Each episode solves one RCPSP instance. Each step is one branching
    decision — the agent picks which ready activity to schedule next.

    The environment wraps BnBSolver directly, so the B&B mechanics
    (dominance, LB pruning, child generation) are identical to evaluation.
    The solver is paused at each branching decision via a callback; the
    agent provides the ordering, then the solver resumes.

    Observations use NodeContext featurisation — identical to the IL
    pipeline — so a BC-pretrained BranchingTransformer warm-starts
    without any adaptation.
    """

    def __init__(
        self,
        instance_source: RCPSPInstance | Path | str,
        max_resources: int = 4,
        time_limit_s: float = 60.0,
        reward_cfg: Optional[RewardConfig] = None,
        dominance: object = "set_based",
        lb_spec: object = DEFAULT_LOWER_BOUND_ID,
    ) -> None:
        self.instance_source = instance_source
        self.max_resources = max_resources
        self.time_limit_s = time_limit_s
        self.reward_cfg = reward_cfg or RewardConfig()
        self.dominance_spec = normalize_dominance_spec(dominance)
        self.lb_spec = lb_spec

        # Set at reset()
        self.instance: Optional[RCPSPInstance] = None
        self._episode_stats: EpisodeStats = EpisodeStats()
        self._n_activities: int = 0
        self._cp_lb: int = 0

        # Step-level state written by the callback, read by step()
        self._pending_node: Optional[BBNode] = None
        self._pending_ctx: Optional[StepContext] = None
        self._pending_incumbent: Optional[int] = None

        # Synchronisation between step() and the solver thread
        self._solver_gen = None
        self._done: bool = True
        self._done_reason: str = "unknown"
        self._steps: int = 0
        # Full search tree (SolverResult) of the most recent episode, captured
        # when the solver finishes. Consumed by the closure-based (subtree)
        # return backup, which needs every node — including pruned and frontier
        # nodes the agent never branched on — not just the decision nodes.
        self._result: Optional[SolverResult] = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_instance(self) -> RCPSPInstance:
        if isinstance(self.instance_source, RCPSPInstance):
            return self.instance_source
        return load_instance(Path(self.instance_source))

    def _observe(
        self,
        node: BBNode,
        incumbent: Optional[int],
        ctx_step: Optional[StepContext] = None,
    ) -> Dict[str, torch.Tensor]:
        horizon = sum(a.duration for a in self.instance.activities.values())
        profile = build_profile(
            self.instance.activities,
            self.instance.resource_caps,
            node.scheduled,
            horizon=horizon,
        )
        from rcpsp_bb_rl.bnb.precedence import build_predecessors
        predecessors = build_predecessors(self.instance)
        ready_sorted = sorted(node.ready)
        earliest_starts: Dict[int, Optional[int]] = {
            rid: earliest_feasible_start(
                self.instance, predecessors, node.scheduled,
                rid, incumbent=incumbent, profile=profile,
            )
            for rid in ready_sorted
        }
        ctx = NodeContext(
            instance=self.instance,
            scheduled=node.scheduled,
            unscheduled=node.unscheduled,
            ready=node.ready,
            lower_bound=node.lower_bound,
            incumbent=incumbent,
            earliest_starts=earliest_starts,
        )
        glob = torch.tensor(
            global_features(ctx, self.max_resources, depth=node.depth),
            dtype=torch.float32,
        )
        cand = torch.tensor(
            [candidate_features(ctx, rid, self.max_resources) for rid in ready_sorted],
            dtype=torch.float32,
        )
        mask = torch.tensor(
            [earliest_starts.get(rid) is not None for rid in ready_sorted],
            dtype=torch.bool,
        )
        critic = torch.tensor(
            self._critic_features(node, incumbent, ctx_step),
            dtype=torch.float32,
        )
        return {
            "global_feats": glob,
            "candidate_feats": cand,
            "ready_ids": torch.tensor(ready_sorted, dtype=torch.long),
            "action_mask": mask,
            "critic_feats": critic,
        }

    def _critic_features(
        self,
        node: BBNode,
        incumbent: Optional[int],
        ctx_step: Optional[StepContext],
    ) -> List[float]:
        """
        Build the critic-only runtime feature vector for the node the agent is
        about to act on. Pulls search-lifetime quantities from the solver's
        StepContext (nodes expanded, elapsed time, frontier size, dual bound).
        When no StepContext is available (should not happen on the RL path),
        falls back to neutral values so the vector is still well-formed.
        """
        if ctx_step is not None:
            return critic_features(
                incumbent=incumbent,
                node_lb=node.lower_bound,
                cp_lb=self._cp_lb,
                frontier_min_lb=ctx_step.frontier_min_lb,
                depth=node.depth,
                n_unscheduled=len(node.unscheduled),
                n_ready=len(node.ready),
                num_activities=self._n_activities,
                stagnation_depth=ctx_step.stagnation_depth,
            )
        return critic_features(
            incumbent=incumbent,
            node_lb=node.lower_bound,
            cp_lb=self._cp_lb,
            frontier_min_lb=None,
            depth=node.depth,
            n_unscheduled=len(node.unscheduled),
            n_ready=len(node.ready),
            num_activities=self._n_activities,
            stagnation_depth=0,
        )

    def _compute_reward(
        self,
        *,
        pre_inc: Optional[int],
        post_inc: Optional[int],
        node_lb: Optional[int],
        nodes_delta: int,
        nodes_since_last_incumbent: int,
    ) -> Tuple[float, Dict[str, float]]:
        """
        Per-step DIAGNOSTIC reward for the advance triggered by the action just
        taken. This is logging-only: the trainer stores it but never uses it to
        compute returns or advantages — those come from the post-episode subtree
        backup (ml/rl/tree_return.py). See RewardConfig for the full note.

        The three terms below mirror the tree backup's reward channels
        (cost = -alpha per node; beta1 first-incumbent strength; beta2 incumbent
        improvement) so the per-episode logs track the same quantities the policy
        is actually trained on — but this is a per-advance approximation and is
        NOT required to equal the tree return.

        All quantities reflect the advance triggered by the action just taken:
            pre_inc  : incumbent at the branching decision the agent acted on
            post_inc : incumbent at the next branching decision (or termination)

        Three terms:
          - r_cost = -alpha * nodes_delta
              Per-node search cost. nodes_delta is ~1 during search (one node
              expanded per step) and larger only on the terminal advance.
          - r_first_incumbent = beta1 * (root_lb / first_inc)
                                 / (1 + log1p(nodes_to_first_incumbent))
              One-time, emitted when the FIRST incumbent appears. Rewards a
              strong first incumbent (root_lb / first_inc -> 1 when the first
              incumbent is near the root bound) found with little search
              (divided by log of nodes spent reaching it).
          - r_incumbent_improvement = beta2 * (old_inc - new_inc)
                                       / nodes_since_last_incumbent
              Emitted whenever an EXISTING incumbent improves. Rewards makespan
              reduction per unit of search effort between incumbents.

        node_lb is unused (no gap-based cost) but kept in the signature for
        call-site stability.
        """
        import math

        _ = node_lb  # retained for call-site stability; unused
        cfg = self.reward_cfg
        reward = 0.0
        breakdown: Dict[str, float] = {
            "cost": 0.0, "first_incumbent": 0.0, "incumbent_improvement": 0.0,
        }

        # 1. Per-node search cost — each expanded node pays -alpha once.
        width = float(max(0, int(nodes_delta)))
        r_cost = -cfg.alpha * width
        reward += r_cost
        breakdown["cost"] = r_cost

        # 2. First-incumbent bonus — one-time, when the first feasible schedule
        #    appears (pre_inc is None and post_inc is not None). Strong-and-cheap
        #    first incumbents score high; weak or expensively-found ones score low.
        r_first = 0.0
        if pre_inc is None and post_inc is not None and post_inc > 0:
            root_lb = float(self._cp_lb)
            quality = max(0.0, min(1.0, root_lb / float(post_inc)))
            nodes_to_first = max(0, int(nodes_since_last_incumbent))
            r_first = cfg.beta1 * quality / (1.0 + math.log1p(nodes_to_first))
            reward += r_first
        breakdown["first_incumbent"] = r_first

        # 3. Incumbent-improvement bonus — when an EXISTING incumbent improves.
        #    Makespan reduction normalized by the search effort it took.
        r_improve = 0.0
        if (
            pre_inc is not None
            and post_inc is not None
            and post_inc < pre_inc
        ):
            seg = float(max(1, int(nodes_since_last_incumbent)))
            r_improve = cfg.beta2 * float(pre_inc - post_inc) / seg
            reward += r_improve
        breakdown["incumbent_improvement"] = r_improve

        return reward, breakdown

    # ------------------------------------------------------------------
    # Generator-based solver coroutine
    # ------------------------------------------------------------------

    def _run_solver(self, instance: RCPSPInstance):
        """
        Generator that drives BnBSolver step by step.

        BnBSolver is synchronous, so we run it in a daemon thread and
        communicate via two single-slot queues:
          to_env   : solver → env  ("branch", node, incumbent, ctx) or ("done", result)
          to_solver: env → solver  (chosen ordering list)

        The generator yields ("branch", node, incumbent, ctx) each time the
        solver needs a branching decision, and ("done", result) when finished.
        The caller sends back the chosen ordering via generator.send().
        """
        import queue
        import threading

        solver = BnBSolver(instance=instance)
        to_solver: queue.Queue = queue.Queue(maxsize=1)
        to_env: queue.Queue = queue.Queue(maxsize=1)

        def _order_fn(node: BBNode, incumbent: Optional[int], step_ctx: StepContext) -> List[int]:
            to_env.put(("branch", node, incumbent, step_ctx))
            return to_solver.get()

        def _run():
            try:
                result = solver.solve(
                    order_ready_fn=_order_fn,
                    lb_spec=self.lb_spec,
                    dominance=self.dominance_spec,
                    time_limit_s=self.time_limit_s,
                )
                to_env.put(("done", result))
            except Exception as exc:
                to_env.put(("error", exc))

        threading.Thread(target=_run, daemon=True).start()

        while True:
            msg = to_env.get()
            if msg[0] == "branch":
                _, node, incumbent, step_ctx = msg
                ordering = yield ("branch", node, incumbent, step_ctx)
                to_solver.put(ordering)
            elif msg[0] == "done":
                yield ("done", msg[1])
                return
            else:
                raise msg[1]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(
        self,
        instance: Optional[RCPSPInstance | Path | str] = None,
    ) -> Dict[str, torch.Tensor]:
        """Start a fresh episode."""
        if instance is not None:
            if isinstance(instance, (Path, str)):
                self.instance_source = instance
                self.instance = load_instance(Path(instance))
            else:
                self.instance_source = instance
                self.instance = instance
        else:
            self.instance = self._load_instance()

        self._episode_stats = EpisodeStats()
        self._steps = 0
        self._done = False
        self._done_reason = "unknown"
        self._result = None
        # nodes_expanded value at the last incumbent (or 0 = episode start);
        # used to size the incumbent efficiency bonus by segment length.
        self._last_incumbent_nodes = 0
        self._n_activities = len(self.instance.activities)
        self._cp_lb = int(lower_bound(
            self.instance,
            set(self.instance.activities.keys()),
            {},
            lb_id=self.lb_spec,
        ))

        self._solver_gen = self._run_solver(self.instance)
        msg = next(self._solver_gen)

        if msg[0] == "done":
            # Trivial instance — solved without any branching decision
            _, result = msg
            self._done = True
            self._done_reason = "search_exhausted"
            self._episode_stats.done_reason = self._done_reason
            self._episode_stats.best_makespan = result.best_makespan
            raise RuntimeError("Instance solved at root — no branching decisions needed.")

        _, node, incumbent, step_ctx = msg
        self._pending_node = node
        self._pending_ctx = step_ctx
        self._pending_incumbent = incumbent

        return self._observe(node, incumbent, step_ctx)

    def step(self, action_index: int) -> StepOutput:
        """
        Branch on the activity at position action_index in sorted(node.ready).

        The chosen activity is placed first in the ordering passed to the
        solver; the solver explores it first (DFS/LIFO push order).
        """
        if self._done or self._solver_gen is None:
            raise RuntimeError("Call reset() before step().")

        node = self._pending_node
        ctx = self._pending_ctx
        incumbent = self._pending_incumbent
        ready_sorted = sorted(node.ready)
        info: Dict = {}

        if action_index < 0 or action_index >= len(ready_sorted):
            info["done_reason"] = "invalid_action"
            info["terminated"] = True
            info["node_id"] = node.node_id
            info["parent_id"] = node.parent_id
            info["depth"] = node.depth
            self._episode_stats.done_reason = "invalid_action"
            self._done = True
            return StepOutput({}, 0.0, True, info)

        chosen = ready_sorted[action_index]
        # Put chosen first; solver pushes in reversed order so chosen is
        # explored first (LIFO stack).
        ordering = [chosen] + [a for a in ready_sorted if a != chosen]

        # ---- Snapshot pre-action state ----
        # ctx was created when the solver paused for THIS branching decision,
        # so ctx.incumbent_after / ctx.nodes_expanded reflect the moment the
        # agent is about to act. node.lower_bound is the LOCAL lower bound of
        # the node being branched on (not the frontier minimum).
        pre_inc: Optional[int] = ctx.incumbent_after
        pre_burden: int = ctx.proof_burden
        pre_frontier_min_lb: Optional[int] = ctx.frontier_min_lb
        pre_nodes: int = ctx.nodes_expanded
        node_lb: Optional[int] = node.lower_bound
        # Path-local stagnation of the node being branched on. Carried in the
        # step info and used as a critic-only feature; not part of any reward.
        pre_stagnation_depth: int = ctx.stagnation_depth

        # Resume the solver with the chosen ordering.
        try:
            msg = self._solver_gen.send(ordering)
        except StopIteration:
            msg = ("done_implicit", None)

        self._steps += 1

        # ---- Snapshot post-action state (after the advance triggered by
        #      the chosen ordering) ----
        done = False
        done_reason = "running"
        next_ctx: Optional[StepContext] = None
        result_obj = None

        if msg[0] == "branch":
            _, next_node, next_incumbent, next_ctx = msg
            post_inc: Optional[int] = next_ctx.incumbent_after
            post_burden: int = next_ctx.proof_burden
            post_nodes: int = next_ctx.nodes_expanded
            advance_lb_pruned = next_ctx.lb_pruned
            advance_dom_pruned = next_ctx.dom_pruned
        elif msg[0] == "done":
            _, result_obj = msg
            done = True
            done_reason = result_obj.done_reason
            post_inc = result_obj.best_makespan
            post_burden = result_obj.final_proof_burden
            post_nodes = result_obj.nodes_expanded
            advance_lb_pruned = 0
            advance_dom_pruned = 0
        else:  # done_implicit — generator already exhausted
            done = True
            done_reason = "search_exhausted"
            post_inc = pre_inc
            post_burden = 0
            post_nodes = pre_nodes
            advance_lb_pruned = 0
            advance_dom_pruned = 0

        nodes_delta = max(0, post_nodes - pre_nodes)

        # ---- Update episode stats ----
        if msg[0] == "branch":
            self._episode_stats.nodes_expanded = post_nodes
            self._episode_stats.nodes_pruned += advance_lb_pruned
            self._episode_stats.dominance_pruned += advance_dom_pruned
        elif result_obj is not None:
            self._episode_stats.nodes_expanded = result_obj.nodes_expanded
            self._episode_stats.nodes_pruned = result_obj.nodes_pruned
            self._episode_stats.dominance_pruned = result_obj.dominance_pruned_children
            self._episode_stats.best_makespan = result_obj.best_makespan

        # Incumbent-improvement tracking: pre_inc -> post_inc during this advance.
        if pre_inc is None and post_inc is not None:
            if self._episode_stats.first_incumbent_node is None:
                self._episode_stats.first_incumbent_node = post_nodes
                self._episode_stats.first_incumbent_makespan = post_inc
            self._episode_stats.last_incumbent_node = post_nodes
            self._episode_stats.last_incumbent_makespan = post_inc
        elif pre_inc is not None and post_inc is not None and post_inc < pre_inc:
            self._episode_stats.incumbent_improvements += 1
            self._episode_stats.last_incumbent_node = post_nodes
            self._episode_stats.last_incumbent_makespan = post_inc

        # ---- Compute per-step diagnostic reward (logging only) ----
        # Segment length: nodes expanded since the previous incumbent. Sized
        # before we advance _last_incumbent_nodes so an improving advance is
        # credited against the segment that produced it.
        nodes_since_last_incumbent = max(1, post_nodes - self._last_incumbent_nodes)
        # Advance the segment anchor on ANY new incumbent, including the first
        # (so the 2nd incumbent's segment is measured from the 1st, not from
        # episode start). This is broader than the bonus condition inside
        # _compute_reward, which fires only on improvement of an EXISTING
        # incumbent.
        new_incumbent = post_inc is not None and (pre_inc is None or post_inc < pre_inc)

        reward, breakdown = self._compute_reward(
            pre_inc=pre_inc,
            post_inc=post_inc,
            node_lb=node_lb,
            nodes_delta=nodes_delta,
            nodes_since_last_incumbent=nodes_since_last_incumbent,
        )

        if new_incumbent:
            self._last_incumbent_nodes = post_nodes

        for key, value in breakdown.items():
            self._episode_stats.reward_breakdown[key] = (
                self._episode_stats.reward_breakdown.get(key, 0.0) + value
            )

        if done:
            self._done = True
            self._done_reason = done_reason
            # Capture the full search tree for the closure-based return backup.
            # result_obj is set only on a clean "done" message; on done_implicit
            # (generator already exhausted) it stays None.
            self._result = result_obj
            self._episode_stats.done_reason = done_reason
            self._episode_stats.total_reward += reward
            if self._episode_stats.best_makespan is not None:
                instance_lb = lower_bound(
                    self.instance,
                    set(self.instance.activities.keys()),
                    {},
                    lb_id=self.lb_spec,
                )
                self._episode_stats.final_gap = float(
                    self._episode_stats.best_makespan - instance_lb
                )

        info.update({
            "done_reason": done_reason,
            "terminated": done and done_reason != "time_limit",
            "steps": self._steps,
            # ---- Tree identity of the branched node ----
            # The decision at this step was made AT this node; record its
            # identity so each transition can be reattached to the search tree
            # for the closure-based (subtree) return backup. parent_id lets us
            # walk the tree bottom-up; depth is a convenience for diagnostics.
            "node_id": node.node_id,
            "parent_id": node.parent_id,
            "depth": node.depth,
            "nodes_expanded": post_nodes,
            "nodes_delta": nodes_delta,
            "best_makespan": post_inc,
            "action_task": chosen,
            "lb_pruned": advance_lb_pruned,
            "dom_pruned": advance_dom_pruned,
            "proof_burden_before": pre_burden,
            "proof_burden_after": post_burden,
            "frontier_min_lb": pre_frontier_min_lb,
            "stagnation_depth": pre_stagnation_depth,
            "nodes_since_last_incumbent": nodes_since_last_incumbent,
            "gap": (
                max(0.0, float(pre_inc - pre_frontier_min_lb) / float(pre_inc))
                if (pre_inc is not None and pre_inc > 0 and pre_frontier_min_lb is not None)
                else 0.0
            ),
            "reward_breakdown": breakdown,
        })

        if not done and msg[0] == "branch":
            self._pending_node = next_node
            self._pending_ctx = next_ctx
            self._pending_incumbent = next_incumbent
            obs = self._observe(next_node, next_incumbent, next_ctx)
        else:
            obs = {}

        return StepOutput(obs, reward, done, info)

    @property
    def episode_stats(self) -> EpisodeStats:
        return self._episode_stats

    @property
    def result(self) -> Optional[SolverResult]:
        """The full SolverResult of the finished episode (None until done)."""
        return self._result

    def search_tree(self) -> Optional[Dict[str, object]]:
        """
        The complete search tree of the finished episode, in a flat,
        backup-ready form for the closure-based (subtree) return.

        Returns None until the episode is done. Otherwise a dict with:
          - "nodes": list of per-node dicts, each:
                {id, parent_id, depth, status, is_incumbent, makespan}
            status is one of "pending" | "expanded" | "pruned" | "solution".
            is_incumbent marks the solution nodes that strictly improved the
            best makespan, in chronological (node-id) order — these are where
            incumbent bonuses are earned. makespan is set only on solution
            nodes (else None).
          - "edges": list of (parent_id, child_id) tuples.
          - "root_id": the root node id (or None for an empty tree).

        This includes EVERY node the solver created — expanded, pruned, and
        unexplored frontier ("pending") nodes — not just the decision nodes the
        agent branched on. The subtree sum G(X) needs all of them.
        """
        res = self._result
        if res is None:
            return None

        # Mark incumbent-improving solution nodes. Solution nodes are produced
        # in chronological order as node ids increase, so a single forward pass
        # over ascending makespan reproduces the incumbent history.
        best: Optional[int] = None
        incumbent_ids = set()
        sol_makespan: Dict[int, int] = {}
        for n in sorted(res.nodes, key=lambda nn: nn.node_id):
            if n.status == "solution":
                mk = current_makespan(n.scheduled)
                sol_makespan[n.node_id] = mk
                if best is None or mk < best:
                    best = mk
                    incumbent_ids.add(n.node_id)

        nodes = [
            {
                "id": n.node_id,
                "parent_id": n.parent_id,
                "depth": n.depth,
                "status": n.status,
                "is_incumbent": n.node_id in incumbent_ids,
                "makespan": sol_makespan.get(n.node_id),
            }
            for n in res.nodes
        ]
        root_id = res.nodes[0].node_id if res.nodes else None
        return {
            "nodes": nodes,
            "edges": list(res.edges),
            "root_id": root_id,
            # Root lower bound (critical-path LB at episode start). Used by the
            # first-incumbent strength bonus: beta1 * (root_lb / first_inc).
            "root_lb": float(self._cp_lb),
        }
