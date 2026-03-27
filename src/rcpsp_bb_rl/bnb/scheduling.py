from __future__ import annotations

from typing import TYPE_CHECKING, Mapping, Optional, Sequence, Set

if TYPE_CHECKING:
    from rcpsp_bb_rl.data.parsing import Activity, RCPSPInstance


def entry_start(entry: object) -> int:
    if hasattr(entry, "start"):
        return int(getattr(entry, "start"))
    if isinstance(entry, Mapping):
        return int(entry.get("start", 0))
    raise TypeError("Scheduled entry must provide a start time.")


def entry_duration(entry: object) -> int:
    if hasattr(entry, "duration"):
        return int(getattr(entry, "duration"))
    if isinstance(entry, Mapping):
        return int(entry.get("duration", 0))
    raise TypeError("Scheduled entry must provide a duration.")


def entry_finish(entry: object) -> int:
    if hasattr(entry, "finish"):
        return int(getattr(entry, "finish"))
    if isinstance(entry, Mapping):
        if "finish" in entry:
            return int(entry["finish"])
        return entry_start(entry) + entry_duration(entry)
    raise TypeError("Scheduled entry must provide a finish time or start+duration.")


def resource_feasible(
    activities: Mapping[int, Activity],
    resource_caps: Sequence[int],
    scheduled: Mapping[int, object],
    act_id: int,
    start: int,
) -> bool:
    """
    Check whether scheduling activity act_id at start violates renewable capacities.
    """
    duration = int(activities[act_id].duration)
    finish = start + duration
    reqs = activities[act_id].resources

    for t in range(start, finish):
        for r, cap in enumerate(resource_caps):
            used = 0
            for other_id, entry in scheduled.items():
                if entry_start(entry) <= t < entry_finish(entry):
                    used += int(activities[other_id].resources[r])
            if used + int(reqs[r]) > int(cap):
                return False
    return True


def earliest_feasible_start(
    instance: RCPSPInstance,
    predecessors: Mapping[int, Set[int]],
    scheduled: Mapping[int, object],
    act_id: int,
    incumbent: Optional[int],
) -> Optional[int]:
    """
    Earliest time satisfying precedence and renewable resource feasibility.
    """
    if not predecessors.get(act_id):
        earliest = 0
    else:
        earliest = max(entry_finish(scheduled[pred]) for pred in predecessors[act_id])

    horizon_hint = sum(act.duration for act in instance.activities.values())
    horizon = incumbent if incumbent is not None else horizon_hint
    horizon = max(horizon, earliest + int(instance.activities[act_id].duration))

    t = earliest
    while t <= horizon:
        if resource_feasible(
            instance.activities,
            instance.resource_caps,
            scheduled,
            act_id,
            t,
        ):
            return t
        t += 1
    return None
