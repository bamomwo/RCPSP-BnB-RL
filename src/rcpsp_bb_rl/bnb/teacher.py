from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from ortools.sat.python import cp_model

from rcpsp_bb_rl.bnb.core import ScheduleEntry, build_predecessors, compute_ready_set, current_makespan, lower_bound
from rcpsp_bb_rl.data.parsing import RCPSPInstance


def _build_cp_model(instance: RCPSPInstance) -> tuple[cp_model.CpModel, Dict[int, cp_model.IntVar]]:
    """Build a simple RCPSP CP-SAT model for single-mode renewable resources."""
    model = cp_model.CpModel()
    horizon = sum(act.duration for act in instance.activities.values())

    start_vars: Dict[int, cp_model.IntVar] = {}
    end_vars: Dict[int, cp_model.IntVar] = {}
    intervals = []

    for act_id, act in instance.activities.items():
        start = model.new_int_var(0, horizon, f"start_{act_id}")
        end = model.new_int_var(0, horizon, f"end_{act_id}")
        interval = model.new_interval_var(start, act.duration, end, f"interval_{act_id}")
        start_vars[act_id] = start
        end_vars[act_id] = end
        intervals.append((interval, act.resources))

    # Precedence constraints.
    for act_id, act in instance.activities.items():
        for succ in act.successors:
            model.add(end_vars[act_id] <= start_vars[succ])

    # Renewable resource cumulatives.
    for r, cap in enumerate(instance.resource_caps):
        model.add_cumulative(
            [iv for iv, _ in intervals],
            [res[r] for _, res in intervals],
            cap,
        )

    makespan = model.new_int_var(0, horizon, "makespan")
    for end in end_vars.values():
        model.add(end <= makespan)
    model.minimize(makespan)

    return model, start_vars


def solve_optimal_schedule(instance: RCPSPInstance, time_limit_s: Optional[float] = None) -> Dict[int, int]:
    """Solve instance with OR-Tools and return start times per activity."""
    model, start_vars = _build_cp_model(instance)
    solver = cp_model.CpSolver()
    if time_limit_s is not None:
        solver.parameters.max_time_in_seconds = time_limit_s

    status = solver.solve(model)
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        raise RuntimeError(f"CP-SAT failed with status {solver.status_name(status)}")

    return {act_id: int(solver.value(var)) for act_id, var in start_vars.items()}


def _optimal_order(starts: Dict[int, int]) -> List[int]:
    """Order tasks by start time then id for a deterministic replay."""
    return [act_id for act_id, _ in sorted(starts.items(), key=lambda kv: (kv[1], kv[0]))]


def generate_trace(
    instance: RCPSPInstance,
    start_times: Dict[int, int],
    instance_name: str | None = None,
) -> List[Dict]:
    """Replay the optimal schedule to produce (state, action) records."""
    predecessors = build_predecessors(instance)

    scheduled: Dict[int, ScheduleEntry] = {}
    unscheduled = set(instance.activities.keys())
    records: List[Dict] = []

    order = _optimal_order(start_times)

    for depth, act_id in enumerate(order):
        ready = compute_ready_set(unscheduled, set(scheduled.keys()), predecessors)
        lb = lower_bound(instance, unscheduled, scheduled)

        # Resource usage at current time (makespan so far).
        ms = current_makespan(scheduled)
        res_used = [0 for _ in instance.resource_caps]
        for other_id, entry in scheduled.items():
            for r, cap in enumerate(instance.resource_caps):
                if entry.start <= ms < entry.finish:
                    res_used[r] += instance.activities[other_id].resources[r]

        static_acts = {
            str(aid): {
                "duration": act.duration,
                "resources": act.resources,
                "successors": act.successors,
                "num_successors": len(act.successors),
                "num_predecessors": len(predecessors.get(aid, ())),
            }
            for aid, act in instance.activities.items()
        }

        # Simple summaries of remaining work.
        rem_durations = [instance.activities[a].duration for a in unscheduled]
        rem_energy = [
            sum(instance.activities[a].duration * instance.activities[a].resources[r] for a in unscheduled)
            for r in range(len(instance.resource_caps))
        ]

        records.append(
            {
                "instance": instance_name,
                "num_activities": instance.num_activities,
                "num_resources": instance.num_resources,
                "resource_caps": instance.resource_caps,
                "depth": depth,
                "ready": sorted(ready),
                "unscheduled": sorted(unscheduled),
                "scheduled": {
                    str(k): {"start": v.start, "finish": v.finish, "duration": v.duration}
                    for k, v in scheduled.items()
                },
                "activities": static_acts,
                "lower_bound": lb,
                "makespan_so_far": ms,
                "resource_used_now": res_used,
                "resource_available_now": [cap - u for cap, u in zip(instance.resource_caps, res_used)],
                "remaining_durations": rem_durations,
                "remaining_energy_per_resource": rem_energy,
                "action": {"task": act_id, "start": start_times[act_id]},
            }
        )

        # Apply action: schedule the task at its optimal start.
        dur = instance.activities[act_id].duration
        st = start_times[act_id]
        scheduled[act_id] = ScheduleEntry(start=st, finish=st + dur, duration=dur)
        unscheduled.discard(act_id)

    return records


def write_trace(records: Iterable[Dict], path: Path | str) -> Path:
    """Write trajectory records to JSONL."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for rec in records:
            f.write(json.dumps(rec))
            f.write("\n")
    return path
