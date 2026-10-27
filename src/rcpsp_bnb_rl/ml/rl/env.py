from __future__ import annotations

from dataclasses import dataclass, field, replace
import math
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch

from rcpsp_bnb_rl.bnb.dominance import normalize_dominance_spec
from rcpsp_bnb_rl.bnb.lower_bounds import DEFAULT_LOWER_BOUND_ID, lower_bound
from rcpsp_bnb_rl.bnb.scheduling import build_profile, earliest_feasible_start
from rcpsp_bnb_rl.bnb.solver import BBNode, BnBSolver, ScheduleEntry, SolverResult, StepContext, current_makespan
from rcpsp_bnb_rl.data.parsing import RCPSPInstance, load_instance
from rcpsp_bnb_rl.ml.action_order import resolve_activity_order
from rcpsp_bnb_rl.ml.il.featurize import (
    InstanceStatics,
    NodeContext,
    build_instance_statics,
    candidate_features,
    critic_features,
    global_features,
    instance_static_features,
    candidate_set_features,
    global_set_features,
    resource_set_features,
    candidate_resource_features,
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


@dataclass
class BootstrapFrontier:
    """Boundary states for training targets, without changing solver closure.

    Completion ratios cover every pending node. Observations contain only
    eligible, nonterminal states with at least one feasible action. Under the
    unit expansion reward the remaining collections are exact: zero remaining
    cost for prunable, dead-end and complete-schedule states, and a single
    expansion (-1) for eligible states with no feasible action.
    """

    completion_ratios: Dict[int, float] = field(default_factory=dict)
    observations: Dict[int, Dict[str, torch.Tensor]] = field(default_factory=dict)
    zero_return_ids: List[int] = field(default_factory=list)
    expansion_only_ids: List[int] = field(default_factory=list)


def real_activity_ids(instance: RCPSPInstance) -> set[int]:
    """Exclude identifiable global dummy endpoints, retaining interior tasks.

    Input files have no dummy labels. A candidate must have zero duration and
    resource demand, be the unique graph source/sink, and connect to every
    activity in the corresponding direction. Ambiguous boundaries are retained;
    in particular, ordinary zero-duration activities are not removed by type.
    """
    activities = instance.activities
    all_ids = set(activities)
    successors = {aid: set(a.successors) & all_ids for aid, a in activities.items()}
    predecessors = {aid: set() for aid in all_ids}
    for aid, children in successors.items():
        for child in children:
            predecessors[child].add(aid)

    def reaches_all(start: int, adjacency: Dict[int, set[int]]) -> bool:
        visited: set[int] = set()
        pending = [start]
        while pending:
            aid = pending.pop()
            if aid not in visited:
                visited.add(aid)
                pending.extend(adjacency[aid] - visited)
        return visited == all_ids

    real_ids = set(all_ids)
    for adjacency, boundary in (
        (successors, [aid for aid in all_ids if not predecessors[aid]]),
        (predecessors, [aid for aid in all_ids if not successors[aid]]),
    ):
        if len(boundary) != 1:
            continue
        aid = boundary[0]
        activity = activities[aid]
        if (
            activity.duration == 0
            and not any(activity.resources)
            and reaches_all(aid, adjacency)
        ):
            real_ids.discard(aid)
    return real_ids


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
        resource_encoder: str = "legacy_flat",
        time_limit_s: float = 60.0,
        dominance: object = "set_based",
        lb_spec: object = DEFAULT_LOWER_BOUND_ID,
    ) -> None:
        self.instance_source = instance_source
        self.max_resources = max_resources
        self.resource_encoder = str(resource_encoder).strip().lower()
        if self.resource_encoder not in {"legacy_flat", "set"}:
            raise ValueError("resource_encoder must be 'legacy_flat' or 'set'")
        self.time_limit_s = time_limit_s
        self.dominance_spec = normalize_dominance_spec(dominance)
        self.lb_spec = lb_spec

        # Set at reset()
        self.instance: Optional[RCPSPInstance] = None
        self._episode_stats: EpisodeStats = EpisodeStats()
        self._n_activities: int = 0
        self._real_activity_ids: set[int] = set()
        self._cp_lb: int = 0
        self._static_feats: Optional[List[float]] = None
        self._statics: Optional[InstanceStatics] = None

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
    ) -> Dict[str, torch.Tensor]:
        ready_sorted = sorted(node.ready)
        # Reuse the solver's shared earliest-start map when present; only build
        # a profile + recompute if the node did not carry one.
        if node.est_map is not None:
            earliest_starts: Dict[int, Optional[int]] = {
                rid: node.est_map.get(rid) for rid in ready_sorted
            }
        else:
            horizon = sum(a.duration for a in self.instance.activities.values())
            profile = build_profile(
                self.instance.activities,
                self.instance.resource_caps,
                node.scheduled,
                horizon=horizon,
            )
            predecessors = self._statics.predecessors
            earliest_starts = {
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
            statics=self._statics,
        )
        # Build tensors via numpy: torch.tensor() on nested Python lists walks
        # every element to infer shape/dtype (slow); np.asarray + from_numpy is
        # the fast path to the identical tensor. Values are unchanged.
        # New policies use a variable-length resource set. Keep the legacy
        # fixed-width observation path available for old checkpoints/configs.
        use_resource_set = bool(getattr(self, "resource_encoder", "legacy_flat") == "set")
        if use_resource_set:
            glob_values = global_set_features(ctx, depth=node.depth)
            cand_values = [candidate_set_features(ctx, rid) for rid in ready_sorted]
            resource_values = resource_set_features(ctx)
            interaction_values = [candidate_resource_features(ctx, rid) for rid in ready_sorted]
        else:
            glob_values = global_features(ctx, self.max_resources, depth=node.depth)
            cand_values = [candidate_features(ctx, rid, self.max_resources) for rid in ready_sorted]
            resource_values = None
            interaction_values = None
        glob = torch.from_numpy(np.asarray(glob_values, dtype=np.float32))
        cand = torch.from_numpy(np.asarray(cand_values, dtype=np.float32))
        mask = torch.from_numpy(
            np.fromiter(
                (earliest_starts.get(rid) is not None for rid in ready_sorted),
                dtype=bool,
                count=len(ready_sorted),
            )
        )
        critic = torch.from_numpy(
            np.asarray(self._critic_features(node, incumbent), dtype=np.float32)
        )
        observation = {
            "global_feats": glob,
            "candidate_feats": cand,
            "ready_ids": torch.tensor(ready_sorted, dtype=torch.long),
            "action_mask": mask,
            "critic_feats": critic,
        }
        if resource_values is not None:
            observation["resource_feats"] = torch.from_numpy(
                np.asarray(resource_values, dtype=np.float32)
            )
            observation["candidate_resource_feats"] = torch.from_numpy(
                np.asarray(interaction_values, dtype=np.float32)
            )
            observation["resource_mask"] = torch.ones(len(resource_values), dtype=torch.bool)
        return observation

    def _critic_features(self, node: BBNode, incumbent: Optional[int]) -> List[float]:
        """Critic-only features: a function of the node's own subproblem."""
        return critic_features(
            incumbent=incumbent,
            node_lb=node.lower_bound,
            cp_lb=self._cp_lb,
            depth=node.depth,
            n_unscheduled=len(node.unscheduled),
            n_ready=len(node.ready),
            num_activities=self._n_activities,
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
        self._real_activity_ids = real_activity_ids(self.instance)
        self._cp_lb = int(lower_bound(
            self.instance,
            set(self.instance.activities.keys()),
            {},
            lb_id=self.lb_spec,
        ))
        # Instance-static difficulty features (constant for the whole episode).
        # Computed once here and reused for every node's critic vector.
        self._static_feats = instance_static_features(self.instance)
        # Instance-invariant structures shared by every NodeContext this episode.
        self._statics = build_instance_statics(self.instance)

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

        return self._observe(node, incumbent)

    def step(
        self,
        action_index: int,
        action_order_indices: Optional[Sequence[int]] = None,
    ) -> StepOutput:
        """
        Branch on the activity at position action_index in sorted(node.ready).

        When action_order_indices is supplied, it must be a complete permutation
        of candidate indices with action_index first. Otherwise the remaining
        activities follow ascending activity-ID order. The solver explores the
        resulting order directly (via reverse push onto its DFS/LIFO stack).
        """
        if self._done or self._solver_gen is None:
            raise RuntimeError("Call reset() before step().")

        node = self._pending_node
        ctx = self._pending_ctx
        incumbent = self._pending_incumbent
        ready_sorted = sorted(node.ready)
        real_count = len(self._real_activity_ids)
        info: Dict = {
            "completion_ratio": (
                len(self._real_activity_ids.intersection(node.scheduled)) / real_count
                if real_count else 1.0
            ),
        }

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
        ordering = resolve_activity_order(
            ready_sorted,
            selected_index=action_index,
            action_order_indices=action_order_indices,
        )

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
            obs = self._observe(next_node, next_incumbent)
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

    def bootstrap_frontier(self, min_completion: float) -> BootstrapFrontier:
        """Build final pending boundary observations for selective bootstrapping.

        Eligibility uses scheduled real tasks divided by total real tasks.
        Exact pruning/terminal cases are reported even below the threshold.
        Feasibility and critic features are evaluated against the final
        incumbent. No solver nodes are expanded, sampled, relabelled, or
        otherwise mutated.
        """
        if not math.isfinite(min_completion) or not 0.0 <= min_completion <= 1.0:
            raise ValueError("min_completion must be a finite number in [0, 1]")
        res = self._result
        if res is None or not self._done or self.instance is None:
            raise RuntimeError("bootstrap_frontier() requires a finished episode")

        frontier = BootstrapFrontier()
        real_ids = real_activity_ids(self.instance)
        real_count = len(real_ids)
        pending_nodes = [node for node in res.nodes if node.status == "pending"]
        incumbent = res.best_makespan
        for node in pending_nodes:
            completion = (
                len(real_ids.intersection(node.scheduled)) / real_count
                if real_count else 1.0
            )
            frontier.completion_ratios[node.node_id] = completion
            # Match the solver's terminal order: bound pruning comes first.
            if incumbent is not None and node.lower_bound >= incumbent:
                frontier.zero_return_ids.append(node.node_id)
                continue
            if not node.unscheduled:
                # Complete schedule: popping it costs no further expansion.
                frontier.zero_return_ids.append(node.node_id)
                continue
            if not node.ready:
                frontier.zero_return_ids.append(node.node_id)
                continue
            if completion < min_completion:
                continue

            # Recompute feasibility under the final incumbent on a shallow
            # copy so neither cache nor solver state changes.
            obs = self._observe(replace(node, est_map=None), incumbent)
            if bool(obs["action_mask"].any()):
                frontier.observations[node.node_id] = obs
            else:
                # Conditional on the final incumbent: the solver counts this
                # as an expansion, unlike empty-ready pruning. Future incumbent
                # tightening can instead cause bound pruning before expansion.
                frontier.expansion_only_ids.append(node.node_id)
        return frontier

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
