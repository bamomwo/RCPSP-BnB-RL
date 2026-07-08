from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

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
    instance_static_features,
)


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class StepOutput:
    observation: Dict[str, torch.Tensor]
    done: bool
    info: Dict


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


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

class BranchingEnv:
    """
    RL environment for the serial B&B branching policy.

    Each episode solves one RCPSP instance. Each step is one branching
    decision — the agent picks which ready activity to schedule next.
    """

    def __init__(
        self,
        instance_source: RCPSPInstance | Path | str,
        max_resources: int = 4,
        time_limit_s: float = 60.0,
        dominance: object = "set_based",
        lb_spec: object = DEFAULT_LOWER_BOUND_ID,
    ) -> None:
        self.instance_source = instance_source
        self.max_resources = max_resources
        self.time_limit_s = time_limit_s
        self.dominance_spec = normalize_dominance_spec(dominance)
        self.lb_spec = lb_spec

        # Set at reset()
        self.instance: Optional[RCPSPInstance] = None
        self._episode_stats: EpisodeStats = EpisodeStats()
        self._n_activities: int = 0
        self._cp_lb: int = 0
        self._static_feats: Optional[List[float]] = None

        # Step-level state written by the callback, read by step()
        self._pending_node: Optional[BBNode] = None
        self._pending_ctx: Optional[StepContext] = None
        self._pending_incumbent: Optional[int] = None

        self._solver_gen = None
        self._done: bool = True
        self._done_reason: str = "unknown"
        self._steps: int = 0
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
        """Build critic-only feature vector for the current node."""
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
                static_feats=self._static_feats,
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
            static_feats=self._static_feats,
        )

    # ------------------------------------------------------------------
    # Generator-based solver coroutine
    # ------------------------------------------------------------------

    def _run_solver(self, instance: RCPSPInstance):
        """
        Generator driving BnBSolver via a daemon thread. Yields ("branch", ...)
        at each branching decision; caller sends back the chosen ordering.
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
        self._n_activities = len(self.instance.activities)
        self._cp_lb = int(lower_bound(
            self.instance,
            set(self.instance.activities.keys()),
            {},
            lb_id=self.lb_spec,
        ))
        # Instance-static difficulty features (constant for the whole episode).
        # Computed once here and reused for every node's critic vector.
        self._static_feats = instance_static_features(self.instance)

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
            return StepOutput({}, True, info)

        chosen = ready_sorted[action_index]
        ordering = [chosen] + [a for a in ready_sorted if a != chosen]

        pre_inc: Optional[int] = ctx.incumbent_after
        pre_burden: int = ctx.proof_burden
        pre_frontier_min_lb: Optional[int] = ctx.frontier_min_lb
        pre_nodes: int = ctx.nodes_expanded
        pre_stagnation_depth: int = ctx.stagnation_depth

        # Resume the solver with the chosen ordering.
        try:
            msg = self._solver_gen.send(ordering)
        except StopIteration:
            msg = ("done_implicit", None)

        self._steps += 1

        # ---- Post-action state ----
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

        # Incumbent-improvement tracking
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

        # ---- Compute segment length for logging ----
        nodes_since_last_incumbent = max(
            1, post_nodes - (self._episode_stats.last_incumbent_node or 0)
        )

        if done:
            self._done = True
            self._done_reason = done_reason
            self._result = result_obj
            self._episode_stats.done_reason = done_reason
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
        })

        if not done and msg[0] == "branch":
            self._pending_node = next_node
            self._pending_ctx = next_ctx
            self._pending_incumbent = next_incumbent
            obs = self._observe(next_node, next_incumbent, next_ctx)
        else:
            obs = {}

        return StepOutput(obs, done, info)

    @property
    def episode_stats(self) -> EpisodeStats:
        return self._episode_stats

    @property
    def result(self) -> Optional[SolverResult]:
        """The full SolverResult of the finished episode (None until done)."""
        return self._result

    def search_tree(self) -> Optional[Dict[str, object]]:
        """
        The complete search tree of the finished episode for the subtree-return
        backup. Returns None until episode is done. Dict with keys:
          "nodes": list of {id, parent_id, depth, status, is_incumbent, makespan}
          "edges": list of (parent_id, child_id)
          "root_id": root node id
          "root_lb": root lower bound
        """
        res = self._result
        if res is None:
            return None

        # Mark incumbent-improving solution nodes
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
            "root_lb": float(self._cp_lb),
        }
