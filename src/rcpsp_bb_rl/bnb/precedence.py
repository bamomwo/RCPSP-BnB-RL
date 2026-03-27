from __future__ import annotations
from collections import deque
from typing import TYPE_CHECKING, Dict, Iterable, List, Mapping, Sequence, Set, Tuple

if TYPE_CHECKING:
    from rcpsp_bb_rl.data.parsing import RCPSPInstance


def build_predecessors(instance: RCPSPInstance) -> Dict[int, Set[int]]:
    """
    Build predecessor sets from successor lists in the instance.
    """
    preds: Dict[int, Set[int]] = {act_id: set() for act_id in instance.activities}
    for act_id, activity in instance.activities.items():
        for succ in activity.successors:
            preds.setdefault(succ, set()).add(act_id)
    return preds


def build_successors(instance: RCPSPInstance) -> Dict[int, List[int]]:
    """
    Materialize successor lists as plain Python lists.
    """
    return {
        act_id: list(activity.successors)
        for act_id, activity in instance.activities.items()
    }


def topological_order(instance: RCPSPInstance) -> List[int]:
    """
    Return a topological ordering of the precedence graph.
    Raise ValueError if the graph is not a DAG.
    """
    preds = build_predecessors(instance)
    succs = build_successors(instance)

    indegree: Dict[int, int] = {
        act_id: len(preds[act_id]) for act_id in instance.activities
    }
    queue = deque([act_id for act_id, deg in indegree.items() if deg == 0])
    order: List[int] = []

    while queue:
        act_id = queue.popleft()
        order.append(act_id)
        for succ in succs[act_id]:
            indegree[succ] -= 1
            if indegree[succ] == 0:
                queue.append(succ)

    if len(order) != len(instance.activities):
        raise ValueError("RCPSP precedence graph is not a DAG.")

    return order


def compute_ready_set(
    unscheduled: Set[int],
    scheduled: Set[int],
    predecessors: Mapping[int, Set[int]],
) -> Set[int]:
    """
    Return currently precedence-feasible unscheduled activities.
    """
    ready: Set[int] = set()
    for act_id in unscheduled:
        if predecessors.get(act_id, set()).issubset(scheduled):
            ready.add(act_id)
    return ready
