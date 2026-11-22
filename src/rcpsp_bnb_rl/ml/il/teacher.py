from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from ortools.sat.python import cp_model

from rcpsp_bnb_rl.bnb.precedence import build_predecessors, compute_ready_set
from rcpsp_bnb_rl.bnb.solver import ScheduleEntry
from rcpsp_bnb_rl.data.parsing import RCPSPInstance


def _build_cp_model(
    instance: RCPSPInstance,
    known_makespan: Optional[int] = None,
) -> tuple[cp_model.CpModel, Dict[int, cp_model.IntVar]]:
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
    if known_makespan is not None:
        if known_makespan < 0:
            raise ValueError("known_makespan must be non-negative")
        model.add(makespan == int(known_makespan))
    else:
        model.minimize(makespan)

    return model, start_vars


def solve_optimal_schedule(
    instance: RCPSPInstance,
    time_limit_s: Optional[float] = None,
    known_makespan: Optional[int] = None,
    num_search_workers: Optional[int] = None,
) -> Dict[int, int]:
    """Solve instance and return start times, optionally at a known makespan."""
    model, start_vars = _build_cp_model(instance, known_makespan=known_makespan)
    solver = cp_model.CpSolver()
    if time_limit_s is not None:
        solver.parameters.max_time_in_seconds = time_limit_s
    if num_search_workers is not None:
        if num_search_workers < 1:
            raise ValueError("num_search_workers must be at least 1")
        solver.parameters.num_search_workers = num_search_workers

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
