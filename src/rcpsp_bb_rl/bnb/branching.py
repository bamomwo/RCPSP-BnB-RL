from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import TYPE_CHECKING, List, Mapping, Optional, Sequence, Set

from rcpsp_bb_rl.bnb.branching_order import NodeLike, ReadyOrderFn, order_by_activity_id
from rcpsp_bb_rl.bnb.scheduling import build_profile, entry_finish, resource_feasible

if TYPE_CHECKING:
    from rcpsp_bb_rl.data.parsing import RCPSPInstance

class SerialBranchingScheme:
    """
    Branch on all ready activities.
    """

    def choose_activities(
        self,
        node: NodeLike,
        incumbent: Optional[int],
        order_ready_fn: Optional[ReadyOrderFn],
    ) -> List[int]:
        if order_ready_fn is None:
            return order_by_activity_id(node, incumbent)
        return list(order_ready_fn(node, incumbent))


@dataclass
class ParallelDecision:
    start_activities: List[int]
    next_time: Optional[int]
    eligible: Set[int]


class ParallelBranchingScheme:
    """
    Time-increment parallel branching scheme.

    At a node time pointer t:
    - identify activities eligible to start at t
    - branch on start decisions at t if any
    - otherwise advance to the next event time
    """

    def __init__(self, max_children: Optional[int] = None) -> None:
        self.max_children = max_children

    def _completed_at_time(self, scheduled: Mapping[int, object], t: int) -> Set[int]:
        return {
            int(act_id)
            for act_id, entry in scheduled.items()
            if entry_finish(entry) <= int(t)
        }

    def _eligible_at_time(
        self,
        *,
        instance: RCPSPInstance,
        predecessors: Mapping[int, Set[int]],
        scheduled: Mapping[int, object],
        unscheduled: Set[int],
        t: int,
        incumbent: Optional[int],
    ) -> Set[int]:
        completed = self._completed_at_time(scheduled, t)

        horizon_hint = sum(act.duration for act in instance.activities.values())
        node_horizon = incumbent if incumbent is not None else horizon_hint
        profile = build_profile(
            instance.activities,
            instance.resource_caps,
            scheduled,
            horizon=node_horizon,
        )

        eligible: Set[int] = set()
        for act_id in unscheduled:
            if not predecessors.get(act_id, set()).issubset(completed):
                continue
            if not resource_feasible(
                instance.activities,
                instance.resource_caps,
                scheduled,
                act_id,
                t,
                profile=profile,
            ):
                continue
            if incumbent is not None:
                finish = int(t) + int(instance.activities[act_id].duration)
                if finish >= int(incumbent):
                    continue
            eligible.add(int(act_id))
        return eligible

    def _next_event_time(self, scheduled: Mapping[int, object], t: int) -> Optional[int]:
        candidates: List[int] = []
        for entry in scheduled.values():
            finish = entry_finish(entry)
            if finish > int(t):
                candidates.append(finish)
        if not candidates:
            return None
        return min(candidates)

    def decide(
        self,
        node: NodeLike,
        *,
        instance: RCPSPInstance,
        predecessors: Mapping[int, Set[int]],
        incumbent: Optional[int],
        order_ready_fn: Optional[ReadyOrderFn],
    ) -> ParallelDecision:
        scheduled = getattr(node, "scheduled", None)
        unscheduled = getattr(node, "unscheduled", None)
        t = int(getattr(node, "current_time", 0))
        if scheduled is None or unscheduled is None:
            raise AttributeError(
                "parallel branching requires node.scheduled and node.unscheduled fields."
            )

        eligible = self._eligible_at_time(
            instance=instance,
            predecessors=predecessors,
            scheduled=scheduled,
            unscheduled=set(unscheduled),
            t=t,
            incumbent=incumbent,
        )

        if not eligible:
            return ParallelDecision(
                start_activities=[],
                next_time=self._next_event_time(scheduled, t),
                eligible=set(),
            )

        proxy = SimpleNamespace(
            ready=set(eligible),
            scheduled=scheduled,
            unscheduled=set(unscheduled),
            current_time=t,
        )
        if order_ready_fn is None:
            ordered = order_by_activity_id(proxy, incumbent)
        else:
            ordered = list(order_ready_fn(proxy, incumbent))
            ordered = [int(a) for a in ordered if int(a) in eligible]

        if self.max_children is not None:
            ordered = ordered[: self.max_children]

        return ParallelDecision(
            start_activities=ordered,
            next_time=None,
            eligible=set(eligible),
        )
