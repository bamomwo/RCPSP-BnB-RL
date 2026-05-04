from __future__ import annotations

from typing import TYPE_CHECKING, List, Mapping, Optional, Sequence, Set

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


class ResourceProfile:
    """
    Precomputed resource-usage array over time.

    usage[r][t] = total units of resource r consumed at time t by all
    activities currently in `scheduled`.  Replaces the O(|scheduled|)
    inner loop in resource_feasible with an O(1) lookup.
    """

    __slots__ = ("usage", "num_resources", "horizon")

    def __init__(self, num_resources: int, horizon: int) -> None:
        self.num_resources = num_resources
        self.horizon = horizon
        self.usage: List[List[int]] = [[0] * (horizon + 1) for _ in range(num_resources)]

    def add_activity(
        self,
        resources: Sequence[int],
        start: int,
        finish: int,
    ) -> None:
        """Increment usage for [start, finish)."""
        for r in range(self.num_resources):
            req = int(resources[r])
            if req == 0:
                continue
            row = self.usage[r]
            for t in range(start, min(finish, self.horizon + 1)):
                row[t] += req

    def remove_activity(
        self,
        resources: Sequence[int],
        start: int,
        finish: int,
    ) -> None:
        """Decrement usage for [start, finish)."""
        for r in range(self.num_resources):
            req = int(resources[r])
            if req == 0:
                continue
            row = self.usage[r]
            for t in range(start, min(finish, self.horizon + 1)):
                row[t] -= req

    def copy(self) -> "ResourceProfile":
        clone = ResourceProfile(self.num_resources, self.horizon)
        clone.usage = [list(row) for row in self.usage]
        return clone

    def feasible_at(
        self,
        resources: Sequence[int],
        caps: Sequence[int],
        start: int,
        finish: int,
    ) -> bool:
        """Return True iff adding activity with given resources at [start,finish) stays within caps."""
        for r in range(self.num_resources):
            req = int(resources[r])
            if req == 0:
                continue
            cap = int(caps[r])
            row = self.usage[r]
            for t in range(start, min(finish, self.horizon + 1)):
                if row[t] + req > cap:
                    return False
        return True


def build_profile(
    activities: Mapping[int, "Activity"],
    resource_caps: Sequence[int],
    scheduled: Mapping[int, object],
    horizon: Optional[int] = None,
) -> ResourceProfile:
    """
    Build a ResourceProfile from a scheduled dict in O(|scheduled| * max_duration * R).
    When horizon is None it is derived from the latest finish time in scheduled.
    """
    num_resources = len(resource_caps)
    if horizon is None:
        horizon = max(
            (entry_finish(e) for e in scheduled.values()),
            default=0,
        )
    profile = ResourceProfile(num_resources, horizon)
    for act_id, entry in scheduled.items():
        profile.add_activity(
            activities[act_id].resources,
            entry_start(entry),
            entry_finish(entry),
        )
    return profile


def resource_feasible(
    activities: Mapping[int, "Activity"],
    resource_caps: Sequence[int],
    scheduled: Mapping[int, object],
    act_id: int,
    start: int,
    profile: Optional[ResourceProfile] = None,
) -> bool:
    """
    Check whether scheduling act_id at start violates renewable capacities.

    If a pre-built ResourceProfile is supplied the check is O(duration * R).
    Without one a profile is built on the fly (still faster than the old
    triple-nested loop for repeated calls on the same scheduled dict).
    """
    duration = int(activities[act_id].duration)
    finish = start + duration
    reqs = activities[act_id].resources

    if profile is not None:
        return profile.feasible_at(reqs, resource_caps, start, finish)

    # Fallback: build a temporary profile for this single check.
    tmp_horizon = max(finish, max((entry_finish(e) for e in scheduled.values()), default=0))
    tmp = build_profile(activities, resource_caps, scheduled, horizon=tmp_horizon)
    return tmp.feasible_at(reqs, resource_caps, start, finish)


def earliest_feasible_start(
    instance: "RCPSPInstance",
    predecessors: Mapping[int, Set[int]],
    scheduled: Mapping[int, object],
    act_id: int,
    incumbent: Optional[int],
    profile: Optional[ResourceProfile] = None,
) -> Optional[int]:
    """
    Earliest time satisfying precedence and renewable resource feasibility.

    Accepts an optional pre-built ResourceProfile.  When supplied, each
    feasibility probe drops from O(|scheduled|) to O(duration * R).
    """
    if not predecessors.get(act_id):
        earliest = 0
    else:
        earliest = max(entry_finish(scheduled[pred]) for pred in predecessors[act_id])

    horizon_hint = sum(act.duration for act in instance.activities.values())
    horizon = incumbent if incumbent is not None else horizon_hint
    horizon = max(horizon, earliest + int(instance.activities[act_id].duration))

    # Build profile once for all probes if not provided.
    if profile is None:
        profile = build_profile(
            instance.activities,
            instance.resource_caps,
            scheduled,
            horizon=horizon,
        )
    elif profile.horizon < horizon:
        # Extend the profile's horizon if needed (rare: only when incumbent is None
        # and the hint exceeds the profile's pre-allocated size).
        extended = build_profile(
            instance.activities,
            instance.resource_caps,
            scheduled,
            horizon=horizon,
        )
        profile = extended

    reqs = instance.activities[act_id].resources
    duration = int(instance.activities[act_id].duration)

    t = earliest
    while t + duration <= horizon + 1:
        if profile.feasible_at(reqs, instance.resource_caps, t, t + duration):
            return t
        t += 1
    return None
