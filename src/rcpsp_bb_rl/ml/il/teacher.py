from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from ortools.sat.python import cp_model

from rcpsp_bb_rl.bnb.lower_bounds import lower_bound
from rcpsp_bb_rl.bnb.precedence import build_predecessors, compute_ready_set
from rcpsp_bb_rl.bnb.scheduling import build_profile, earliest_feasible_start
from rcpsp_bb_rl.bnb.solver import ScheduleEntry, current_makespan
from rcpsp_bb_rl.data.parsing import RCPSPInstance


def _build_cp_model(instance: RCPSPInstance) -> tuple[cp_model.CpModel, Dict[int, cp_model.IntVar]]:
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

    for act_id, act in instance.activities.items():
        for succ in act.successors:
            model.add(end_vars[act_id] <= start_vars[succ])

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


def solve_optimal_schedule(
    instance: RCPSPInstance,
    time_limit_s: Optional[float] = None,
) -> Dict[int, int]:
    """Solve instance with OR-Tools CP-SAT and return optimal start times per activity."""
    model, start_vars = _build_cp_model(instance)
    solver = cp_model.CpSolver()
    if time_limit_s is not None:
        solver.parameters.max_time_in_seconds = time_limit_s

    status = solver.solve(model)
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        raise RuntimeError(f"CP-SAT failed with status {solver.status_name(status)}")

    return {act_id: int(solver.value(var)) for act_id, var in start_vars.items()}


def _optimal_order(starts: Dict[int, int]) -> List[int]:
    """Order activities by start time then id for a deterministic replay."""
    return [act_id for act_id, _ in sorted(starts.items(), key=lambda kv: (kv[1], kv[0]))]


def generate_trace(
    instance: RCPSPInstance,
    start_times: Dict[int, int],
    instance_name: str | None = None,
) -> List[Dict]:
    """
    Replay the optimal schedule to produce (state, action) records.

    Each record contains exactly the fields that NodeContext and TrajectoryRecord
    need — no redundant activity static data (the instance file is the source of
    truth for that).
    """
    predecessors = build_predecessors(instance)
    optimal_makespan = max(
        start_times[a] + instance.activities[a].duration
        for a in start_times
    )

    scheduled: Dict[int, ScheduleEntry] = {}
    unscheduled = set(instance.activities.keys())
    records: List[Dict] = []

    order = _optimal_order(start_times)

    for depth, act_id in enumerate(order):
        ready = compute_ready_set(unscheduled, set(scheduled.keys()), predecessors)
        ready_sorted = sorted(ready)
        unscheduled_sorted = sorted(unscheduled)

        lb = lower_bound(instance, unscheduled, scheduled)
        ms = current_makespan(scheduled)

        # Resource usage at current makespan frontier
        res_used = [0] * instance.num_resources
        for other_id, entry in scheduled.items():
            if entry.start <= ms < entry.finish:
                for r in range(instance.num_resources):
                    res_used[r] += instance.activities[other_id].resources[r]

        # Remaining energy per resource
        rem_energy = [
            sum(
                instance.activities[a].duration * instance.activities[a].resources[r]
                for a in unscheduled
            )
            for r in range(instance.num_resources)
        ]

        # Earliest feasible starts for each ready activity
        node_profile = build_profile(
            instance.activities,
            instance.resource_caps,
            scheduled,
            horizon=sum(act.duration for act in instance.activities.values()),
        )
        earliest_starts = {
            str(rid): earliest_feasible_start(
                instance,
                predecessors,
                scheduled,
                rid,
                incumbent=None,
                profile=node_profile,
            )
            for rid in ready_sorted
        }

        records.append({
            "instance": instance_name,
            "num_activities": instance.num_activities,
            "num_resources": instance.num_resources,
            "resource_caps": instance.resource_caps,
            "optimal_makespan": optimal_makespan,
            "depth": depth,
            "ready": ready_sorted,
            "unscheduled": unscheduled_sorted,
            "scheduled": {
                str(k): {"start": v.start, "finish": v.finish, "duration": v.duration}
                for k, v in scheduled.items()
            },
            "earliest_start": earliest_starts,
            "lower_bound": lb,
            "incumbent": None,   # no incumbent during trace generation (optimal path)
            "makespan_so_far": ms,
            "resource_used_now": res_used,
            "resource_available_now": [cap - u for cap, u in zip(instance.resource_caps, res_used)],
            "remaining_durations": [instance.activities[a].duration for a in unscheduled],
            "remaining_energy_per_resource": rem_energy,
            "action": {"task": act_id, "start": start_times[act_id]},
        })

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
