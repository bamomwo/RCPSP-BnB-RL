from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, List, Optional, Set, Tuple

from rcpsp_bb_rl.bnb.branching import (
    ParallelBranchingScheme,
    ReadyOrderFn,
    SerialBranchingScheme,
)
from rcpsp_bb_rl.bnb.dominance import build_dominance_engine, normalize_dominance_spec
from rcpsp_bb_rl.bnb.lower_bounds import DEFAULT_LOWER_BOUND_ID, lower_bound
from rcpsp_bb_rl.bnb.precedence import build_predecessors, compute_ready_set
from rcpsp_bb_rl.bnb.scheduling import build_profile, earliest_feasible_start, resource_feasible

if TYPE_CHECKING:
    from rcpsp_bb_rl.data.parsing import RCPSPInstance


@dataclass
class ScheduleEntry:
    start: int
    finish: int
    duration: int


@dataclass
class BBNode:
    node_id: int
    scheduled: Dict[int, ScheduleEntry]
    ready: Set[int]
    unscheduled: Set[int]
    lower_bound: int
    parent_id: Optional[int]
    action: Optional[str]
    depth: int
    current_time: int = 0
    status: str = "pending"  # pending | expanded | pruned | solution


@dataclass
class IncumbentEvent:
    """Records a single improvement to the best known makespan."""
    rank: int               # 1-based index of this improvement
    makespan: int
    nodes_expanded: int     # how many nodes had been expanded when this was found
    depth: int              # depth of the solution node in the tree


@dataclass
class DebugInfo:
    incumbent_history: List[IncumbentEvent] = field(default_factory=list)
    all_makespans: List[int] = field(default_factory=list)   # every complete schedule seen
    lb_pruned: int = 0
    dominance_pruned: int = 0


@dataclass
class SolverResult:
    best_makespan: Optional[int]
    best_schedule: Optional[Dict[int, ScheduleEntry]]
    nodes: List[BBNode]
    edges: List[Tuple[int, int]]
    nodes_expanded: int
    nodes_pruned: int
    nodes_expanded_after_incumbent: int
    nodes_pruned_after_incumbent: int
    first_incumbent_expanded: Optional[int]
    dominance_enabled: bool
    dominance_rules: Tuple[str, ...]
    dominance_pruned_children: int
    dominance_pruned_by_rule: Dict[str, int]
    debug_info: Optional[DebugInfo] = None


def current_makespan(scheduled: Dict[int, ScheduleEntry]) -> int:
    return max((entry.finish for entry in scheduled.values()), default=0)


class BnBSolver:
    """
    DFS-based branch-and-bound solver for RCPSP.
    """

    def __init__(
        self,
        instance: RCPSPInstance,
        branching_scheme=None,
    ) -> None:
        self.instance = instance
        self.predecessors = build_predecessors(instance)
        self.branching_scheme = branching_scheme or SerialBranchingScheme()
        self.nodes: List[BBNode] = []
        self.edges: List[Tuple[int, int]] = []
        self._next_node_id = 0

    def _new_node_id(self) -> int:
        nid = self._next_node_id
        self._next_node_id += 1
        return nid

    def solve(
        self,
        max_nodes: Optional[int] = None,
        order_ready_fn: Optional[ReadyOrderFn] = None,
        time_limit_s: Optional[float] = None,
        lb_spec: object = DEFAULT_LOWER_BOUND_ID,
        dominance: object = False,
        target_makespan: Optional[int] = None,
        stop_on_first_solution: bool = False,
        debug: bool = False,
    ) -> SolverResult:
        unscheduled = set(self.instance.activities.keys())
        ready = compute_ready_set(unscheduled, set(), self.predecessors)
        dominance_cfg = normalize_dominance_spec(dominance)
        dominance_engine = build_dominance_engine(
            instance=self.instance,
            predecessors=self.predecessors,
            dominance=dominance_cfg,
        )

        root_id = self._new_node_id()
        root = BBNode(
            node_id=root_id,
            scheduled={},
            ready=ready,
            unscheduled=unscheduled,
            lower_bound=lower_bound(self.instance, unscheduled, {}, lb_id=lb_spec),
            parent_id=None,
            action=None,
            depth=0,
            current_time=0,
        )
        self.nodes.append(root)
        dominance_engine.register_state(
            root.unscheduled,
            root.scheduled,
            root.lower_bound,
            current_time=root.current_time,
        )

        stack: List[int] = [root_id]
        if target_makespan is not None and target_makespan < 0:
            raise ValueError("target_makespan must be >= 0 when provided.")

        incumbent_bound: Optional[int]
        if target_makespan is None:
            incumbent_bound = None
        else:
            incumbent_bound = int(target_makespan) + 1

        best_makespan: Optional[int] = None
        best_schedule: Optional[Dict[int, ScheduleEntry]] = None
        nodes_expanded = 0
        nodes_pruned = 0
        nodes_expanded_after_incumbent = 0
        nodes_pruned_after_incumbent = 0
        first_incumbent_expanded: Optional[int] = None
        seen_incumbent = False
        debug_info: Optional[DebugInfo] = DebugInfo() if debug else None

        start_time_monotonic = time.perf_counter()

        def time_exceeded() -> bool:
            if time_limit_s is None:
                return False
            return (time.perf_counter() - start_time_monotonic) >= time_limit_s

        while stack and ((max_nodes is None) or (nodes_expanded < max_nodes)) and not time_exceeded():
            if stop_on_first_solution and best_makespan is not None:
                break

            node_id = stack.pop()
            node = self.nodes[node_id]

            incumbent = incumbent_bound
            if incumbent is not None and node.lower_bound >= incumbent:
                node.status = "pruned"
                nodes_pruned += 1
                if debug_info is not None:
                    debug_info.lb_pruned += 1
                if seen_incumbent:
                    nodes_pruned_after_incumbent += 1
                continue

            if not node.unscheduled:
                node.status = "solution"
                makespan = current_makespan(node.scheduled)
                if debug_info is not None:
                    debug_info.all_makespans.append(makespan)
                if incumbent_bound is None or makespan < incumbent_bound:
                    incumbent_bound = makespan
                    best_makespan = makespan
                    best_schedule = node.scheduled
                    if debug_info is not None:
                        debug_info.incumbent_history.append(IncumbentEvent(
                            rank=len(debug_info.incumbent_history) + 1,
                            makespan=makespan,
                            nodes_expanded=nodes_expanded,
                            depth=node.depth,
                        ))
                    if not seen_incumbent:
                        first_incumbent_expanded = nodes_expanded
                        seen_incumbent = True
                continue

            is_parallel = isinstance(self.branching_scheme, ParallelBranchingScheme)
            acts: List[int] = []
            if is_parallel:
                while True:
                    decision = self.branching_scheme.decide(
                        node=node,
                        instance=self.instance,
                        predecessors=self.predecessors,
                        incumbent=incumbent_bound,
                        order_ready_fn=order_ready_fn,
                    )
                    if decision.start_activities:
                        completed_now = {
                            act_id
                            for act_id, entry in node.scheduled.items()
                            if entry.finish <= node.current_time
                        }
                        node.ready = compute_ready_set(
                            node.unscheduled,
                            completed_now,
                            self.predecessors,
                        )
                        acts = list(decision.start_activities)
                        break

                    if decision.next_time is None:
                        node.status = "pruned"
                        nodes_pruned += 1
                        if seen_incumbent:
                            nodes_pruned_after_incumbent += 1
                        break

                    node.current_time = int(decision.next_time)
                    completed_now = {
                        act_id
                        for act_id, entry in node.scheduled.items()
                        if entry.finish <= node.current_time
                    }
                    node.ready = compute_ready_set(
                        node.unscheduled,
                        completed_now,
                        self.predecessors,
                    )

                    if incumbent_bound is not None and node.current_time >= incumbent_bound:
                        node.status = "pruned"
                        nodes_pruned += 1
                        if seen_incumbent:
                            nodes_pruned_after_incumbent += 1
                        break

                    if time_exceeded():
                        break

                if node.status == "pruned" or not acts:
                    continue
            else:
                if not node.ready:
                    node.status = "pruned"
                    nodes_pruned += 1
                    if seen_incumbent:
                        nodes_pruned_after_incumbent += 1
                    continue

                acts = self.branching_scheme.choose_activities(
                    node=node,
                    incumbent=incumbent_bound,
                    order_ready_fn=order_ready_fn,
                )

            node.status = "expanded"
            nodes_expanded += 1
            if seen_incumbent:
                nodes_expanded_after_incumbent += 1

            # Build the resource profile once for this node; reused across all children.
            horizon_hint = sum(act.duration for act in self.instance.activities.values())
            node_horizon = incumbent_bound if incumbent_bound is not None else horizon_hint
            node_profile = build_profile(
                self.instance.activities,
                self.instance.resource_caps,
                node.scheduled,
                horizon=node_horizon,
            )

            # Reverse push for DFS/LIFO.
            for act_id in reversed(acts):
                if is_parallel:
                    est_start = int(node.current_time)
                    completed_now = {
                        aid
                        for aid, entry in node.scheduled.items()
                        if entry.finish <= est_start
                    }
                    if not self.predecessors.get(act_id, set()).issubset(completed_now):
                        continue
                    if not resource_feasible(
                        self.instance.activities,
                        self.instance.resource_caps,
                        node.scheduled,
                        act_id,
                        est_start,
                        profile=node_profile,
                    ):
                        continue
                else:
                    est_start = earliest_feasible_start(
                        self.instance,
                        self.predecessors,
                        node.scheduled,
                        act_id,
                        incumbent_bound,
                        profile=node_profile,
                    )
                    if est_start is None:
                        continue

                duration = self.instance.activities[act_id].duration
                finish = est_start + duration

                child_scheduled = dict(node.scheduled)
                child_scheduled[act_id] = ScheduleEntry(
                    start=est_start,
                    finish=finish,
                    duration=duration,
                )

                child_unscheduled = set(node.unscheduled)
                child_unscheduled.discard(act_id)

                child_time = int(node.current_time) if is_parallel else 0
                if is_parallel:
                    completed_child = {
                        aid
                        for aid, entry in child_scheduled.items()
                        if entry.finish <= child_time
                    }
                    child_ready = compute_ready_set(
                        child_unscheduled,
                        completed_child,
                        self.predecessors,
                    )
                else:
                    child_ready = compute_ready_set(
                        child_unscheduled,
                        set(child_scheduled.keys()),
                        self.predecessors,
                    )

                child_lb = lower_bound(
                    self.instance,
                    child_unscheduled,
                    child_scheduled,
                    lb_id=lb_spec,
                )

                pruned_rule = dominance_engine.prune_child(
                    parent_scheduled=node.scheduled,
                    child_scheduled=child_scheduled,
                    child_unscheduled=child_unscheduled,
                    child_lb=child_lb,
                    act_id=act_id,
                    child_start=est_start,
                    parent_time=int(node.current_time),
                    child_time=child_time,
                )
                if pruned_rule is not None:
                    if debug_info is not None:
                        debug_info.dominance_pruned += 1
                    continue

                child_id = self._new_node_id()
                child_node = BBNode(
                    node_id=child_id,
                    scheduled=child_scheduled,
                    ready=child_ready,
                    unscheduled=child_unscheduled,
                    lower_bound=child_lb,
                    parent_id=node_id,
                    action=f"act {act_id}@{est_start}",
                    depth=node.depth + 1,
                    current_time=child_time,
                )

                self.nodes.append(child_node)
                self.edges.append((node_id, child_id))
                stack.append(child_id)

        return SolverResult(
            best_makespan=best_makespan,
            best_schedule=best_schedule,
            nodes=self.nodes,
            edges=self.edges,
            nodes_expanded=nodes_expanded,
            nodes_pruned=nodes_pruned,
            nodes_expanded_after_incumbent=nodes_expanded_after_incumbent,
            nodes_pruned_after_incumbent=nodes_pruned_after_incumbent,
            first_incumbent_expanded=first_incumbent_expanded,
            dominance_enabled=dominance_cfg.enabled,
            dominance_rules=tuple(dominance_cfg.rules),
            dominance_pruned_children=dominance_engine.stats.pruned_children,
            dominance_pruned_by_rule=dict(dominance_engine.stats.pruned_by_rule),
            debug_info=debug_info,
        )


def solve_serial(
    instance: RCPSPInstance,
    max_nodes: Optional[int] = None,
    order_ready_fn: Optional[ReadyOrderFn] = None,
    time_limit_s: Optional[float] = None,
    lb_spec: object = DEFAULT_LOWER_BOUND_ID,
    dominance: object = False,
    target_makespan: Optional[int] = None,
    stop_on_first_solution: bool = False,
    debug: bool = False,
) -> SolverResult:
    solver = BnBSolver(
        instance=instance,
        branching_scheme=SerialBranchingScheme(),
    )
    return solver.solve(
        max_nodes=max_nodes,
        order_ready_fn=order_ready_fn,
        time_limit_s=time_limit_s,
        lb_spec=lb_spec,
        dominance=dominance,
        target_makespan=target_makespan,
        stop_on_first_solution=stop_on_first_solution,
        debug=debug,
    )


def solve_parallel(
    instance: RCPSPInstance,
    max_nodes: Optional[int] = None,
    order_ready_fn: Optional[ReadyOrderFn] = None,
    time_limit_s: Optional[float] = None,
    max_children: Optional[int] = None,
    lb_spec: object = DEFAULT_LOWER_BOUND_ID,
    dominance: object = False,
    target_makespan: Optional[int] = None,
    stop_on_first_solution: bool = False,
    debug: bool = False,
) -> SolverResult:
    solver = BnBSolver(
        instance=instance,
        branching_scheme=ParallelBranchingScheme(max_children=max_children),
    )
    return solver.solve(
        max_nodes=max_nodes,
        order_ready_fn=order_ready_fn,
        time_limit_s=time_limit_s,
        lb_spec=lb_spec,
        dominance=dominance,
        target_makespan=target_makespan,
        stop_on_first_solution=stop_on_first_solution,
        debug=debug,
    )
