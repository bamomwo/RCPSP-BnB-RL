import math
import time
from collections import deque
from dataclasses import dataclass
from math import inf
from typing import Callable, Dict, Iterable, List, Optional, Set, Tuple

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
    Lower bound on makespan: max(precedence-based, resource-based).
    """
    return max(
        _precedence_lower_bound(instance, scheduled),
        _resource_lower_bound(instance, unscheduled, scheduled),
    )


def _precedence_lower_bound(
    instance: RCPSPInstance,
    scheduled: Dict[int, ScheduleEntry],
) -> int:
    preds = build_predecessors(instance)
    succs: Dict[int, Iterable[int]] = {
        act_id: activity.successors for act_id, activity in instance.activities.items()
    }
    in_deg = {act_id: len(preds[act_id]) for act_id in instance.activities}

    queue = deque(act_id for act_id, deg in in_deg.items() if deg == 0)
    earliest_finish: Dict[int, int] = {}

    while queue:
        act_id = queue.popleft()
        pred_finish = max((earliest_finish[p] for p in preds[act_id]), default=0)
        if act_id in scheduled:
            entry = scheduled[act_id]
            est = max(pred_finish, entry.start)
            eft = max(entry.finish, est + entry.duration)
        else:
            est = pred_finish
            eft = est + instance.activities[act_id].duration
        earliest_finish[act_id] = eft

        for succ in succs[act_id]:
            in_deg[succ] -= 1
            if in_deg[succ] == 0:
                queue.append(succ)

    return max(earliest_finish.values(), default=0)


def _resource_lower_bound(
    instance: RCPSPInstance,
    unscheduled: Iterable[int],
    scheduled: Dict[int, ScheduleEntry],
) -> int:
    cur = current_makespan(scheduled)
    max_inc = 0
    for r, cap in enumerate(instance.resource_caps):
        if cap <= 0:
            continue
        energy = sum(
            instance.activities[a].duration * instance.activities[a].resources[r]
            for a in unscheduled
        )
        if energy > 0:
            max_inc = max(max_inc, int(math.ceil(energy / cap)))
    return cur + max_inc


ReadyOrderFn = Callable[[BBNode, Optional[int]], Iterable[int]]


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

    def solve(
        self,
        max_nodes: int = 10000,
        order_ready_fn: Optional[ReadyOrderFn] = None,
        time_limit_s: Optional[float] = None,
    ) -> SolverResult:
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
        order_fn = order_ready_fn or (lambda node, _inc: sorted(node.ready))
        start_time_monotonic = time.perf_counter()

        def time_exceeded() -> bool:
            if time_limit_s is None:
                return False
            return (time.perf_counter() - start_time_monotonic) >= time_limit_s

        while stack and nodes_expanded < max_nodes and not time_exceeded():
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

            ready_order = list(order_fn(node, best_makespan))

            # Push in reverse: DFS pops LIFO, so reversing makes the highest-priority
            # (first in ready_order, e.g., top policy score) expand first.
            for act_id in reversed(ready_order):
                est_start = earliest_feasible_start(
                    self.instance,
                    self.predecessors,
                    node.scheduled,
                    act_id,
                    best_makespan,
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
                    action=f"act {act_id}@{est_start}",
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
