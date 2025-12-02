from dataclasses import dataclass
from math import inf
from typing import Dict, Iterable, List, Optional, Set, Tuple

from rcpsp_bb_rl.data.parsing import Activity, RCPSPInstance


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
    status: str = "pending"  # pending | expanded | pruned | solution


@dataclass
class SolverResult:
    best_makespan: Optional[int]
    best_schedule: Optional[Dict[int, ScheduleEntry]]
    nodes: List[BBNode]
    edges: List[Tuple[int, int]]


def build_predecessors(instance: RCPSPInstance) -> Dict[int, Set[int]]:
    preds: Dict[int, Set[int]] = {act_id: set() for act_id in instance.activities}
    for act_id, activity in instance.activities.items():
        for succ in activity.successors:
            preds[succ].add(act_id)
    return preds


def compute_ready_set(
    unscheduled: Set[int], scheduled: Set[int], predecessors: Dict[int, Set[int]]
) -> Set[int]:
    ready: Set[int] = set()
    for act_id in unscheduled:
        if predecessors.get(act_id, set()).issubset(scheduled):
            ready.add(act_id)
    return ready


def current_makespan(scheduled: Dict[int, ScheduleEntry]) -> int:
    return max((entry.finish for entry in scheduled.values()), default=0)


def resource_feasible(
    activities: Dict[int, Activity],
    resource_caps: List[int],
    scheduled: Dict[int, ScheduleEntry],
    act_id: int,
    start: int,
) -> bool:
    """Check if scheduling act_id at start time fits resource capacities."""
    duration = activities[act_id].duration
    finish = start + duration
    reqs = activities[act_id].resources

    for t in range(start, finish):
        for r, cap in enumerate(resource_caps):
            used = 0
            for other_id, entry in scheduled.items():
                if entry.start <= t < entry.finish:
                    used += activities[other_id].resources[r]
            if used + reqs[r] > cap:
                return False
    return True


def earliest_feasible_start(
    instance: RCPSPInstance,
    predecessors: Dict[int, Set[int]],
    scheduled: Dict[int, ScheduleEntry],
    act_id: int,
    incumbent: Optional[int],
) -> Optional[int]:
    """Find the earliest start time that satisfies precedence and resources."""
    if not predecessors.get(act_id):
        earliest = 0
    else:
        earliest = max(scheduled[pred].finish for pred in predecessors[act_id])

    horizon_hint = sum(act.duration for act in instance.activities.values())
    horizon = incumbent if incumbent is not None else horizon_hint
    horizon = max(horizon, earliest + instance.activities[act_id].duration)

    t = earliest
    while t <= horizon:
        if resource_feasible(instance.activities, instance.resource_caps, scheduled, act_id, t):
            return t
        t += 1
    return None


def lower_bound(
    instance: RCPSPInstance,
    unscheduled: Iterable[int],
    scheduled: Dict[int, ScheduleEntry],
) -> int:
    """
    Very rough lower bound: current makespan plus the longest remaining duration.
    """
    cur = current_makespan(scheduled)
    remaining = max((instance.activities[a].duration for a in unscheduled), default=0)
    return cur + remaining


class BnBSolver:
    """Simple DFS-based branch-and-bound for RCPSP."""

    def __init__(self, instance: RCPSPInstance) -> None:
        self.instance = instance
        self.predecessors = build_predecessors(instance)
        self.nodes: List[BBNode] = []
        self.edges: List[Tuple[int, int]] = []
        self._next_node_id = 0

    def _new_node_id(self) -> int:
        nid = self._next_node_id
        self._next_node_id += 1
        return nid

    def solve(self, max_nodes: int = 10000) -> SolverResult:
        unscheduled = set(self.instance.activities.keys())
        ready = compute_ready_set(unscheduled, set(), self.predecessors)
        root_id = self._new_node_id()
        root = BBNode(
            node_id=root_id,
            scheduled={},
            ready=ready,
            unscheduled=unscheduled,
            lower_bound=lower_bound(self.instance, unscheduled, {}),
            parent_id=None,
            action=None,
            depth=0,
        )
        self.nodes.append(root)

        stack: List[int] = [root_id]
        best_makespan: Optional[int] = None
        best_schedule: Optional[Dict[int, ScheduleEntry]] = None
        nodes_expanded = 0

        while stack and nodes_expanded < max_nodes:
            node_id = stack.pop()
            node = self.nodes[node_id]

            if node.lower_bound is None:
                node.status = "pruned"
                continue

            incumbent = best_makespan
            if incumbent is not None and node.lower_bound >= incumbent:
                node.status = "pruned"
                continue

            if not node.unscheduled:
                node.status = "solution"
                makespan = current_makespan(node.scheduled)
                if best_makespan is None or makespan < best_makespan:
                    best_makespan = makespan
                    best_schedule = node.scheduled
                continue

            node.status = "expanded"
            nodes_expanded += 1

            for act_id in sorted(node.ready):
                start_time = earliest_feasible_start(
                    self.instance,
                    self.predecessors,
                    node.scheduled,
                    act_id,
                    best_makespan,
                )
                if start_time is None:
                    continue

                duration = self.instance.activities[act_id].duration
                finish = start_time + duration

                child_scheduled = dict(node.scheduled)
                child_scheduled[act_id] = ScheduleEntry(
                    start=start_time,
                    finish=finish,
                    duration=duration,
                )
                child_unscheduled = set(node.unscheduled)
                child_unscheduled.discard(act_id)

                child_ready = compute_ready_set(
                    child_unscheduled,
                    set(child_scheduled.keys()),
                    self.predecessors,
                )

                child_lb = lower_bound(self.instance, child_unscheduled, child_scheduled)
                child_id = self._new_node_id()
                child_node = BBNode(
                    node_id=child_id,
                    scheduled=child_scheduled,
                    ready=child_ready,
                    unscheduled=child_unscheduled,
                    lower_bound=child_lb,
                    parent_id=node_id,
                    action=f"act {act_id}@{start_time}",
                    depth=node.depth + 1,
                )
                self.nodes.append(child_node)
                self.edges.append((node_id, child_id))
                stack.append(child_id)

        return SolverResult(
            best_makespan=best_makespan,
            best_schedule=best_schedule,
            nodes=self.nodes,
            edges=self.edges,
        )
