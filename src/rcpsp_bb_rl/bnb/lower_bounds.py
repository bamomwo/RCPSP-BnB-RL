from __future__ import annotations

from typing import Callable, Dict, Iterable, List, Mapping, Tuple

from rcpsp_bb_rl.data.parsing import RCPSPInstance
from rcpsp_bb_rl.bnb.precedence_utils import build_predecessors, topological_order


LowerBoundFn = Callable[[RCPSPInstance, Iterable[int], Mapping[int, object]], int]


# -----------------------------------------------------------------------------
# Helpers for scheduled entries
# -----------------------------------------------------------------------------

def _entry_start(entry: object) -> int:
    if hasattr(entry, "start"):
        return int(getattr(entry, "start"))
    if isinstance(entry, Mapping):
        return int(entry.get("start", 0))
    raise TypeError("Scheduled entry must provide a start time.")


def _entry_duration(entry: object) -> int:
    if hasattr(entry, "duration"):
        return int(getattr(entry, "duration"))
    if isinstance(entry, Mapping):
        return int(entry.get("duration", 0))
    raise TypeError("Scheduled entry must provide a duration.")


def _entry_finish(entry: object) -> int:
    if hasattr(entry, "finish"):
        return int(getattr(entry, "finish"))
    if isinstance(entry, Mapping):
        if "finish" in entry:
            return int(entry["finish"])
        return _entry_start(entry) + _entry_duration(entry)
    raise TypeError("Scheduled entry must provide a finish time or start+duration.")


# -----------------------------------------------------------------------------
# Lower bounds
# -----------------------------------------------------------------------------


def _ceil_div(num: int, den: int) -> int:
    if den <= 0:
        raise ValueError("Denominator must be positive for ceil division.")
    return (num + den - 1) // den


def _aggregate_resource_events(
    instance: RCPSPInstance,
    scheduled: Mapping[int, object],
    resource_idx: int,
) -> List[Tuple[int, int]]:
    """
    Build sorted event list (time, delta_usage) for one resource.
    """
    events: Dict[int, int] = {}
    for act_id, entry in scheduled.items():
        req = int(instance.activities[act_id].resources[resource_idx])
        if req <= 0:
            continue

        start = _entry_start(entry)
        finish = _entry_finish(entry)
        if finish <= start:
            continue

        events[start] = events.get(start, 0) + req
        events[finish] = events.get(finish, 0) - req

    return sorted(events.items())


def lb_cp(
    instance: RCPSPInstance,
    unscheduled: Iterable[int],
    scheduled: Mapping[int, object],
) -> int:
    """
    Critical-path lower bound.

    Ignores resource constraints and computes a precedence-only lower bound.
    Scheduled activities are treated as fixed according to their recorded
    start/finish information.
    """
    _ = unscheduled  # kept for interface consistency across all lower bounds

    predecessors = build_predecessors(instance)
    order = topological_order(instance)

    earliest_finish: Dict[int, int] = {}

    for act_id in order:
        pred_finish = max(
            (earliest_finish[pred] for pred in predecessors.get(act_id, set())),
            default=0,
        )

        if act_id in scheduled:
            entry = scheduled[act_id]
            start = max(pred_finish, _entry_start(entry))
            finish = max(_entry_finish(entry), start + _entry_duration(entry))
        else:
            duration = int(instance.activities[act_id].duration)
            finish = pred_finish + duration

        earliest_finish[act_id] = finish

    return max(earliest_finish.values(), default=0)


def lb_cc(
    instance: RCPSPInstance,
    unscheduled: Iterable[int],
    scheduled: Mapping[int, object],
) -> int:
    """
    Critical-capacity lower bound.

    For each renewable resource r, compute the minimum horizon T_r required to
    process all remaining "energy" from unscheduled activities under the
    residual capacity profile induced by currently fixed scheduled activities.
    Return max_r T_r, also respecting already fixed scheduled finish times.

    This is a valid relaxation (preemptive/fractional packing on each resource),
    hence a lower bound on the final makespan of any completion of this node.
    """
    unscheduled_ids = list(unscheduled)
    fixed_horizon = max((_entry_finish(entry) for entry in scheduled.values()), default=0)
    if not unscheduled_ids:
        return fixed_horizon

    # Resource-energy required by unscheduled activities.
    remaining_energy: List[int] = [0 for _ in range(instance.num_resources)]
    for act_id in unscheduled_ids:
        act = instance.activities[act_id]
        dur = int(act.duration)
        for r in range(instance.num_resources):
            req = int(act.resources[r])
            if req > 0 and dur > 0:
                remaining_energy[r] += dur * req

    # Keep this finite but very large to force pruning once an incumbent exists.
    inf_lb = 10**15
    per_resource_lb: List[int] = []

    for r, cap in enumerate(instance.resource_caps):
        energy = remaining_energy[r]
        if energy <= 0:
            per_resource_lb.append(0)
            continue

        events = _aggregate_resource_events(instance, scheduled, r)
        used = 0
        prev_t = 0
        accumulated = 0
        reached = False

        for t, delta in events:
            if t > prev_t:
                avail = cap - used
                if avail > 0:
                    gain = (t - prev_t) * avail
                    if accumulated + gain >= energy:
                        need = energy - accumulated
                        per_resource_lb.append(prev_t + _ceil_div(need, avail))
                        reached = True
                        break
                    accumulated += gain
                prev_t = t
            used += delta

        if reached:
            continue

        if cap <= 0:
            per_resource_lb.append(inf_lb)
            continue

        rem = energy - accumulated
        per_resource_lb.append(prev_t + _ceil_div(rem, cap))

    return max([fixed_horizon, *per_resource_lb], default=fixed_horizon)


# -----------------------------------------------------------------------------
# Registry
# -----------------------------------------------------------------------------

LOWER_BOUND_FNS: Dict[str, LowerBoundFn] = {
    "lb_cp": lb_cp,
    "lb_cc": lb_cc,
}

DEFAULT_LOWER_BOUND_ID = "lb_cp"


def list_lower_bound_ids() -> List[str]:
    return list(LOWER_BOUND_FNS.keys())


def get_lower_bound_fn(lb_id: str) -> LowerBoundFn:
    try:
        return LOWER_BOUND_FNS[lb_id]
    except KeyError as exc:
        known = ", ".join(sorted(LOWER_BOUND_FNS))
        raise KeyError(f"Unknown lower bound '{lb_id}'. Available: [{known}]") from exc


def lower_bound(
    instance: RCPSPInstance,
    unscheduled: Iterable[int],
    scheduled: Mapping[int, object],
    lb_id: str = DEFAULT_LOWER_BOUND_ID,
) -> int:
    """
    Compute the requested lower bound.
    """
    fn = get_lower_bound_fn(lb_id)
    return fn(instance, unscheduled, scheduled)
