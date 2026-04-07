from __future__ import annotations

from collections import deque
from collections.abc import Sequence
from itertools import permutations
from typing import TYPE_CHECKING, Callable, Dict, Iterable, List, Mapping, Set, Tuple

from rcpsp_bb_rl.bnb.precedence import build_predecessors, topological_order
from rcpsp_bb_rl.bnb.scheduling import (
    entry_duration as _entry_duration,
    entry_finish as _entry_finish,
    entry_start as _entry_start,
)

if TYPE_CHECKING:
    from rcpsp_bb_rl.data.parsing import RCPSPInstance

LowerBoundFn = Callable[[object, Iterable[int], Mapping[int, object]], int]
_VERY_LARGE_LB = 10**15


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


def _build_successors_from_predecessors(
    predecessors: Mapping[int, Set[int]],
    activities: Iterable[int],
) -> Dict[int, Set[int]]:
    successors: Dict[int, Set[int]] = {int(act_id): set() for act_id in activities}
    for succ, preds in predecessors.items():
        succ_id = int(succ)
        successors.setdefault(succ_id, set())
        for pred in preds:
            successors.setdefault(int(pred), set()).add(succ_id)
    return successors


def _topological_order_from_predecessors(
    predecessors: Mapping[int, Set[int]],
    activities: Iterable[int],
) -> List[int]:
    indegree: Dict[int, int] = {int(act_id): 0 for act_id in activities}
    for act_id, preds in predecessors.items():
        aid = int(act_id)
        indegree.setdefault(aid, 0)
        indegree[aid] = len(preds)
        for pred in preds:
            indegree.setdefault(int(pred), 0)

    successors = _build_successors_from_predecessors(predecessors, indegree.keys())
    queue = deque(sorted(aid for aid, deg in indegree.items() if deg == 0))
    order: List[int] = []

    while queue:
        act_id = queue.popleft()
        order.append(act_id)
        for succ in sorted(successors.get(act_id, set())):
            indegree[succ] -= 1
            if indegree[succ] == 0:
                queue.append(succ)

    if len(order) != len(indegree):
        raise ValueError("Precedence graph is not a DAG after inserting pairwise arc.")

    return order


def _compute_reachability(
    order: Sequence[int],
    successors: Mapping[int, Set[int]],
) -> Dict[int, Set[int]]:
    reachable: Dict[int, Set[int]] = {int(act_id): set() for act_id in order}
    for act_id in reversed(order):
        for succ in successors.get(act_id, set()):
            succ_id = int(succ)
            reachable[act_id].add(succ_id)
            reachable[act_id].update(reachable.get(succ_id, set()))
    return reachable


def _can_add_arc_without_cycle(
    predecessors: Mapping[int, Set[int]],
    reachable_from: Mapping[int, Set[int]],
    src: int,
    dst: int,
) -> bool:
    if src == dst:
        return False
    if src in predecessors.get(dst, set()):
        return True
    return src not in reachable_from.get(dst, set())


def _resource_incompatible(
    instance: RCPSPInstance,
    i: int,
    j: int,
) -> bool:
    dur_i = int(instance.activities[i].duration)
    dur_j = int(instance.activities[j].duration)
    if dur_i <= 0 or dur_j <= 0:
        return False

    req_i = instance.activities[i].resources
    req_j = instance.activities[j].resources
    for r, cap in enumerate(instance.resource_caps):
        if int(req_i[r]) + int(req_j[r]) > int(cap):
            return True
    return False


def _resource_incompatible_triplet(
    instance: RCPSPInstance,
    i: int,
    j: int,
    k: int,
) -> bool:
    dur_i = int(instance.activities[i].duration)
    dur_j = int(instance.activities[j].duration)
    dur_k = int(instance.activities[k].duration)
    if dur_i <= 0 or dur_j <= 0 or dur_k <= 0:
        return False

    req_i = instance.activities[i].resources
    req_j = instance.activities[j].resources
    req_k = instance.activities[k].resources
    for r, cap in enumerate(instance.resource_caps):
        if int(req_i[r]) + int(req_j[r]) + int(req_k[r]) > int(cap):
            return True
    return False


def _precedence_lb(
    instance: RCPSPInstance,
    scheduled: Mapping[int, object],
    predecessors: Mapping[int, Set[int]],
    order: Sequence[int],
) -> int:
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


def _precedence_lb_with_added_arcs(
    instance: RCPSPInstance,
    scheduled: Mapping[int, object],
    predecessors: Mapping[int, Set[int]],
    activities: Sequence[int],
    arcs: Sequence[Tuple[int, int]],
) -> int | None:
    preds_tmp: Dict[int, Set[int]] = {
        int(act_id): set(predecessors.get(int(act_id), set()))
        for act_id in activities
    }
    for src, dst in arcs:
        if src == dst:
            return None
        preds_tmp.setdefault(int(src), set())
        preds_tmp.setdefault(int(dst), set()).add(int(src))

    try:
        order_tmp = _topological_order_from_predecessors(preds_tmp, activities)
    except ValueError:
        return None
    return _precedence_lb(instance, scheduled, preds_tmp, order_tmp)


def _compute_heads(
    instance: RCPSPInstance,
    scheduled: Mapping[int, object],
    predecessors: Mapping[int, Set[int]],
    order: Sequence[int],
) -> Dict[int, int]:
    """
    Earliest start times under precedence with currently fixed scheduled entries.
    """
    heads: Dict[int, int] = {}
    finishes: Dict[int, int] = {}

    for act_id in order:
        pred_finish = max(
            (finishes[pred] for pred in predecessors.get(act_id, set())),
            default=0,
        )
        if act_id in scheduled:
            entry = scheduled[act_id]
            start = max(pred_finish, _entry_start(entry))
            finish = max(_entry_finish(entry), start + _entry_duration(entry))
        else:
            duration = int(instance.activities[act_id].duration)
            start = pred_finish
            finish = start + duration

        heads[act_id] = int(start)
        finishes[act_id] = int(finish)

    return heads


def _compute_tails(
    instance: RCPSPInstance,
    successors: Mapping[int, Set[int]],
    order: Sequence[int],
) -> Dict[int, int]:
    """
    Longest precedence tail after each activity (excluding own duration).
    """
    tails: Dict[int, int] = {int(act_id): 0 for act_id in order}
    for act_id in reversed(order):
        best = 0
        for succ in successors.get(act_id, set()):
            dur_succ = int(instance.activities[succ].duration)
            candidate = dur_succ + tails[succ]
            if candidate > best:
                best = candidate
        tails[act_id] = best
    return tails


def _selected_by_pm_rule(
    instance: RCPSPInstance,
    act_id: int,
    resource_idx: int,
    k: int,
) -> bool:
    cap = int(instance.resource_caps[resource_idx])
    demand = int(instance.activities[act_id].resources[resource_idx])
    return demand > (cap - int(k))


def _build_pm2_k_values(capacity: int) -> List[int]:
    """
    Small selected subset of k-cases for PM2.

    Includes boundary and representative interior values.
    """
    cap = int(capacity)
    if cap <= 0:
        return []

    raw = {
        1,
        2,
        cap // 2,
        cap - 1,
        cap,
    }
    return sorted(k for k in raw if 1 <= k <= cap)


def _parallel_machine_relaxation_bound(
    instance: RCPSPInstance,
    resource_idx: int,
    k: int,
    jobs: Sequence[int],
    heads: Mapping[int, int],
    tails: Mapping[int, int],
) -> int:
    """
    Lower bound for selected jobs on a k-machine relaxation.

    Uses two standard valid terms:
    - job term: max_j (head_j + p_j + tail_j)
    - load term: min_head + ceil(sum_j p_j / k) + min_tail
    """
    _ = resource_idx  # kept for signature symmetry with future PM variants

    if int(k) <= 0 or not jobs:
        return 0

    proc_sum = 0
    job_term = 0
    min_head = None
    min_tail = None
    for act_id in jobs:
        p = int(instance.activities[act_id].duration)
        h = int(heads[act_id])
        q = int(tails[act_id])

        proc_sum += p
        candidate = h + p + q
        if candidate > job_term:
            job_term = candidate
        if min_head is None or h < min_head:
            min_head = h
        if min_tail is None or q < min_tail:
            min_tail = q

    if min_head is None or min_tail is None:
        return job_term

    load_term = int(min_head) + _ceil_div(proc_sum, int(k)) + int(min_tail)
    return max(job_term, load_term)


def _are_activities_compatible(
    instance: RCPSPInstance,
    i: int,
    j: int,
    reachable_from: Mapping[int, Set[int]],
) -> bool:
    """
    Return True if i and j can be processed in parallel in the NP sense.

    Two activities are considered incompatible when one precedes the other
    (transitively) or when their combined demand exceeds any resource capacity.
    """
    if i == j:
        return False

    # Precedence-related activities cannot overlap.
    if j in reachable_from.get(i, set()) or i in reachable_from.get(j, set()):
        return False

    req_i = instance.activities[i].resources
    req_j = instance.activities[j].resources
    for r, cap in enumerate(instance.resource_caps):
        if int(req_i[r]) + int(req_j[r]) > int(cap):
            return False
    return True


def _node_packing_bound_value(
    packed_set: Sequence[int],
    heads: Mapping[int, int],
    tails: Mapping[int, int],
    durations: Mapping[int, int],
) -> int:
    """
    Evaluate NP bound value for a packed set of pairwise incompatible activities.
    """
    if not packed_set:
        return 0
    min_head = min(int(heads[act_id]) for act_id in packed_set)
    min_tail = min(int(tails[act_id]) for act_id in packed_set)
    sum_dur = sum(int(durations[act_id]) for act_id in packed_set)
    return min_head + sum_dur + min_tail


def _resource_projection_bound(
    instance: RCPSPInstance,
    packed_set: Sequence[int],
    heads: Mapping[int, int],
    tails: Mapping[int, int],
    scheduled: Mapping[int, object],
) -> int:
    """
    Resource-projection bound over a packed activity subset.

    For each resource r, project total subset energy onto the window
    [min(head), Cmax - min(tail)] to obtain:
        Cmax >= min_head + ceil(total_energy_r / cap_r) + min_tail.
    """
    _ = scheduled  # reserved for future strengthening using residual profiles

    if not packed_set:
        return 0

    min_head = min(int(heads[act_id]) for act_id in packed_set)
    min_tail = min(int(tails[act_id]) for act_id in packed_set)
    best = 0

    for r, cap in enumerate(instance.resource_caps):
        cap_int = int(cap)
        if cap_int <= 0:
            continue
        energy = 0
        for act_id in packed_set:
            dur = int(instance.activities[act_id].duration)
            req = int(instance.activities[act_id].resources[r])
            if dur <= 0 or req <= 0:
                continue
            energy += dur * req
        if energy <= 0:
            continue
        candidate = min_head + _ceil_div(energy, cap_int) + min_tail
        if candidate > best:
            best = candidate

    return best


def _single_machine_bound(
    packed_set: Sequence[int],
    heads: Mapping[int, int],
    tails: Mapping[int, int],
    durations: Mapping[int, int],
) -> int:
    """
    Single-machine lower bound on a packed subset.

    Every packed activity must be processed on one serial machine in this
    relaxation, yielding:
        Cmax >= min_head + sum_durations + min_tail.
    """
    if not packed_set:
        return 0
    min_head = min(int(heads[act_id]) for act_id in packed_set)
    min_tail = min(int(tails[act_id]) for act_id in packed_set)
    sum_dur = sum(int(durations[act_id]) for act_id in packed_set)
    return min_head + sum_dur + min_tail


def _copy_predecessors(
    predecessors: Mapping[int, Set[int]],
    activities: Iterable[int],
) -> Dict[int, Set[int]]:
    return {int(act_id): set(predecessors.get(int(act_id), set())) for act_id in activities}


def _violates_horizon(
    instance: RCPSPInstance,
    heads: Mapping[int, int],
    tails: Mapping[int, int],
    horizon: int,
) -> bool:
    for act_id, head in heads.items():
        dur = int(instance.activities[act_id].duration)
        tail = int(tails.get(act_id, 0))
        if int(head) + dur + tail > int(horizon):
            return True
    return False


def _test_pair_direction(
    instance: RCPSPInstance,
    scheduled: Mapping[int, object],
    predecessors: Mapping[int, Set[int]],
    activities: Sequence[int],
    horizon: int,
    src: int,
    dst: int,
) -> bool:
    """
    Test whether adding arc src -> dst can still satisfy the given horizon.
    """
    if src == dst:
        return False

    preds_tmp = _copy_predecessors(predecessors, activities)
    preds_tmp.setdefault(int(src), set())
    preds_tmp.setdefault(int(dst), set()).add(int(src))

    try:
        order_tmp = _topological_order_from_predecessors(preds_tmp, activities)
    except ValueError:
        return False

    succ_tmp = _build_successors_from_predecessors(preds_tmp, order_tmp)
    heads_tmp = _compute_heads(instance, scheduled, preds_tmp, order_tmp)
    tails_tmp = _compute_tails(instance, succ_tmp, order_tmp)
    return not _violates_horizon(instance, heads_tmp, tails_tmp, horizon)


def _compute_latest_starts(
    instance: RCPSPInstance,
    scheduled: Mapping[int, object],
    successors: Mapping[int, Set[int]],
    order: Sequence[int],
    horizon: int,
) -> Dict[int, int]:
    """
    Latest-start upper bounds under a fixed horizon and precedence.

    Scheduled activities are treated as fixed starts; therefore their latest
    start is additionally capped by the recorded start time.
    """
    latest: Dict[int, int] = {}
    for act_id in reversed(order):
        dur = int(instance.activities[act_id].duration)
        if act_id in scheduled:
            ub = int(_entry_start(scheduled[act_id]))
        else:
            ub = int(horizon) - dur

        succs = successors.get(act_id, set())
        if succs:
            succ_ub = min(int(latest[succ]) - dur for succ in succs)
            ub = min(ub, succ_ub)

        latest[act_id] = int(ub)

    return latest


def _build_elementary_intervals(
    core_intervals: Sequence[Tuple[int, int, int]],
) -> List[Tuple[int, int]]:
    """
    Build sorted elementary half-open intervals from core interval boundaries.
    """
    points = sorted({int(cstart) for _, cstart, _ in core_intervals} | {int(cend) for _, _, cend in core_intervals})
    intervals: List[Tuple[int, int]] = []
    for idx in range(len(points) - 1):
        intervals.append((points[idx], points[idx + 1]))
    return intervals


def _build_time_periods(
    es: Mapping[int, int],
    ls: Mapping[int, int],
    durations: Mapping[int, int],
    horizon: int,
) -> List[Tuple[int, int]]:
    """
    Build elementary time periods for time-period energy checks.
    """
    points: Set[int] = {0, int(horizon)}
    for act_id, e in es.items():
        d = int(durations.get(act_id, 0))
        l = int(ls.get(act_id, e))
        if d <= 0:
            continue
        candidates = (
            int(e),
            int(l),
            int(e) + d,
            int(l) + d,
        )
        for t in candidates:
            if 0 <= t <= int(horizon):
                points.add(t)

    sorted_points = sorted(points)
    periods: List[Tuple[int, int]] = []
    for idx in range(len(sorted_points) - 1):
        periods.append((sorted_points[idx], sorted_points[idx + 1]))
    return periods


def _minimum_processing_in_period(
    es_i: int,
    ls_i: int,
    dur_i: int,
    a: int,
    b: int,
) -> int:
    """
    Mandatory processing amount of one activity in period [a, b).
    """
    if dur_i <= 0 or b <= a:
        return 0
    compulsory_lo = int(ls_i)
    compulsory_hi = int(es_i) + int(dur_i)
    overlap = min(int(b), compulsory_hi) - max(int(a), compulsory_lo)
    return max(0, overlap)


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
    return _precedence_lb(instance, scheduled, predecessors, order)


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
            per_resource_lb.append(_VERY_LARGE_LB)
            continue

        rem = energy - accumulated
        per_resource_lb.append(prev_t + _ceil_div(rem, cap))

    return max([fixed_horizon, *per_resource_lb], default=fixed_horizon)


def lb_ip0(
    instance: RCPSPInstance,
    unscheduled: Iterable[int],
    scheduled: Mapping[int, object],
) -> int:
    """
    Pairwise-incompatibility lower bound (IP0).

    For each resource-incompatible pair (i, j), enforce at least one ordering
    relation (i -> j) or (j -> i). The pair contributes:
        min( LB with i->j, LB with j->i )
    and the final bound is the max over all incompatible pairs and the base
    precedence bound.
    """
    _ = unscheduled  # kept for interface consistency across all lower bounds

    predecessors = build_predecessors(instance)
    order = topological_order(instance)
    successors = _build_successors_from_predecessors(predecessors, order)
    reachable_from = _compute_reachability(order, successors)

    fixed_horizon = max((_entry_finish(entry) for entry in scheduled.values()), default=0)
    base_lb = _precedence_lb(instance, scheduled, predecessors, order)
    best_lb = max(base_lb, fixed_horizon)

    scheduled_ids = {int(aid) for aid in scheduled.keys()}
    for idx_i, i in enumerate(order):
        for j in order[idx_i + 1 :]:
            if i in scheduled_ids and j in scheduled_ids:
                continue

            if not _resource_incompatible(instance, i, j):
                continue

            candidate_bounds: List[int] = []

            if _can_add_arc_without_cycle(predecessors, reachable_from, i, j):
                preds_ij = {aid: set(preds) for aid, preds in predecessors.items()}
                preds_ij.setdefault(i, set())
                preds_ij.setdefault(j, set()).add(i)
                order_ij = _topological_order_from_predecessors(preds_ij, order)
                candidate_bounds.append(
                    _precedence_lb(instance, scheduled, preds_ij, order_ij)
                )

            if _can_add_arc_without_cycle(predecessors, reachable_from, j, i):
                preds_ji = {aid: set(preds) for aid, preds in predecessors.items()}
                preds_ji.setdefault(j, set())
                preds_ji.setdefault(i, set()).add(j)
                order_ji = _topological_order_from_predecessors(preds_ji, order)
                candidate_bounds.append(
                    _precedence_lb(instance, scheduled, preds_ji, order_ji)
                )

            if not candidate_bounds:
                return _VERY_LARGE_LB

            pair_lb = min(candidate_bounds)
            if pair_lb > best_lb:
                best_lb = pair_lb

    return max(best_lb, fixed_horizon)


def lb_ip1(
    instance: RCPSPInstance,
    unscheduled: Iterable[int],
    scheduled: Mapping[int, object],
) -> int:
    """
    Triplet-incompatibility lower bound (IP1).

    For each resource-incompatible triplet (i, j, k), enforce one serial order
    among its six permutations. For each permutation (a, b, c), add temporary
    arcs (a -> b) and (b -> c), compute the precedence LB if acyclic, and keep
    the minimum over feasible permutations. The final bound is the max over
    all considered triplets and the base precedence bound.
    """
    _ = unscheduled  # kept for interface consistency across all lower bounds

    predecessors = build_predecessors(instance)
    order = topological_order(instance)

    fixed_horizon = max((_entry_finish(entry) for entry in scheduled.values()), default=0)
    base_lb = _precedence_lb(instance, scheduled, predecessors, order)
    best_lb = max(base_lb, fixed_horizon)

    scheduled_ids = {int(aid) for aid in scheduled.keys()}
    for idx_i, i in enumerate(order):
        for idx_j in range(idx_i + 1, len(order)):
            j = order[idx_j]
            for idx_k in range(idx_j + 1, len(order)):
                k = order[idx_k]
                if i in scheduled_ids and j in scheduled_ids and k in scheduled_ids:
                    continue

                if not _resource_incompatible_triplet(instance, i, j, k):
                    continue

                triplet_candidates: List[int] = []
                for a, b, c in permutations((i, j, k)):
                    lb_tmp = _precedence_lb_with_added_arcs(
                        instance=instance,
                        scheduled=scheduled,
                        predecessors=predecessors,
                        activities=order,
                        arcs=[(a, b), (b, c)],
                    )
                    if lb_tmp is not None:
                        triplet_candidates.append(lb_tmp)

                if not triplet_candidates:
                    return _VERY_LARGE_LB

                triplet_lb = min(triplet_candidates)
                if triplet_lb > best_lb:
                    best_lb = triplet_lb

    return max(best_lb, fixed_horizon)


def lb_pm1(
    instance: RCPSPInstance,
    unscheduled: Iterable[int],
    scheduled: Mapping[int, object],
) -> int:
    """
    Parallel-machine lower bound (PM1).

    PM1 uses the full-capacity case k = B_r for each resource r, selecting jobs
    through the PM rule and applying a k-machine relaxation bound.
    """
    _ = unscheduled  # kept for interface consistency across all lower bounds

    fixed_horizon = max((_entry_finish(entry) for entry in scheduled.values()), default=0)
    predecessors = build_predecessors(instance)
    order = topological_order(instance)
    successors = _build_successors_from_predecessors(predecessors, order)

    heads = _compute_heads(instance, scheduled, predecessors, order)
    tails = _compute_tails(instance, successors, order)

    base_lb = 0
    for act_id in order:
        duration = int(instance.activities[act_id].duration)
        candidate = int(heads[act_id]) + duration + int(tails[act_id])
        if candidate > base_lb:
            base_lb = candidate

    best_lb = max(base_lb, fixed_horizon)

    for r in range(instance.num_resources):
        capacity = int(instance.resource_caps[r])
        k = capacity  # PM1 only
        if k <= 0:
            continue

        jobs: List[int] = []
        for act_id in order:
            duration = int(instance.activities[act_id].duration)
            demand = int(instance.activities[act_id].resources[r])
            if duration <= 0 or demand <= 0:
                continue
            if not _selected_by_pm_rule(instance, act_id, r, k):
                continue
            jobs.append(act_id)

        if not jobs:
            continue

        lb_r = _parallel_machine_relaxation_bound(
            instance=instance,
            resource_idx=r,
            k=k,
            jobs=jobs,
            heads=heads,
            tails=tails,
        )
        if lb_r > best_lb:
            best_lb = lb_r

    return max(best_lb, fixed_horizon)


def lb_pm2(
    instance: RCPSPInstance,
    unscheduled: Iterable[int],
    scheduled: Mapping[int, object],
) -> int:
    """
    Parallel-machine lower bound (PM2).

    PM2 evaluates a restricted subset of k-machine cases for each resource
    and keeps the strongest resulting bound.
    """
    _ = unscheduled  # kept for interface consistency across all lower bounds

    fixed_horizon = max((_entry_finish(entry) for entry in scheduled.values()), default=0)
    predecessors = build_predecessors(instance)
    successors = _build_successors_from_predecessors(predecessors, instance.activities.keys())
    order = topological_order(instance)

    heads = _compute_heads(instance, scheduled, predecessors, order)
    tails = _compute_tails(instance, successors, order)

    base_lb = 0
    for act_id in order:
        duration = int(instance.activities[act_id].duration)
        candidate = int(heads[act_id]) + duration + int(tails[act_id])
        if candidate > base_lb:
            base_lb = candidate

    best_lb = max(base_lb, fixed_horizon)

    for r in range(instance.num_resources):
        capacity = int(instance.resource_caps[r])
        if capacity <= 0:
            continue

        for k in _build_pm2_k_values(capacity):
            if k <= 0:
                continue

            jobs: List[int] = []
            for act_id in order:
                duration = int(instance.activities[act_id].duration)
                demand = int(instance.activities[act_id].resources[r])
                if duration <= 0:
                    continue
                if demand <= 0:
                    continue
                if not _selected_by_pm_rule(instance, act_id, r, k):
                    continue
                jobs.append(act_id)

            if not jobs:
                continue

            lb_rk = _parallel_machine_relaxation_bound(
                instance=instance,
                resource_idx=r,
                k=k,
                jobs=jobs,
                heads=heads,
                tails=tails,
            )
            if lb_rk > best_lb:
                best_lb = lb_rk

    return max(best_lb, fixed_horizon)


def lb_np0(
    instance: RCPSPInstance,
    unscheduled: Iterable[int],
    scheduled: Mapping[int, object],
) -> int:
    """
    Node-packing lower bound (NP0).

    Builds a greedy set of pairwise incompatible unscheduled activities and
    evaluates a packing-style bound value on that set.
    """
    unscheduled_set = {int(a) for a in unscheduled}
    fixed_horizon = max((_entry_finish(entry) for entry in scheduled.values()), default=0)

    predecessors = build_predecessors(instance)
    order = topological_order(instance)
    successors = _build_successors_from_predecessors(predecessors, order)

    heads = _compute_heads(instance, scheduled, predecessors, order)
    tails = _compute_tails(instance, successors, order)
    durations: Dict[int, int] = {
        int(act_id): int(instance.activities[act_id].duration) for act_id in order
    }

    base_lb = 0
    for act_id in order:
        candidate = int(heads[act_id]) + durations[act_id] + int(tails[act_id])
        if candidate > base_lb:
            base_lb = candidate

    best_lb = max(base_lb, fixed_horizon)

    critical_set: Set[int] = set()
    for act_id in order:
        if int(heads[act_id]) + durations[act_id] + int(tails[act_id]) == base_lb:
            critical_set.add(act_id)

    act_ids: List[int] = [
        act_id for act_id in order if act_id in unscheduled_set and durations[act_id] > 0
    ]
    if not act_ids:
        return best_lb

    reachable_from = _compute_reachability(order, successors)
    compatible: Dict[int, Set[int]] = {act_id: set() for act_id in act_ids}
    for idx_i, i in enumerate(act_ids):
        for j in act_ids[idx_i + 1 :]:
            if _are_activities_compatible(instance, i, j, reachable_from):
                compatible[i].add(j)
                compatible[j].add(i)

    act_ids.sort(
        key=lambda act_id: (
            -int(act_id in critical_set),
            len(compatible[act_id]),
            -durations[act_id],
            act_id,
        )
    )

    remaining = list(act_ids)
    packed_set: List[int] = []
    while remaining:
        i = remaining.pop(0)
        packed_set.append(i)
        remaining = [j for j in remaining if j not in compatible[i]]

    candidate_lb = _node_packing_bound_value(
        packed_set=packed_set,
        heads=heads,
        tails=tails,
        durations=durations,
    )
    if candidate_lb > best_lb:
        best_lb = candidate_lb

    return max(best_lb, fixed_horizon)


def lb_np1(
    instance: RCPSPInstance,
    unscheduled: Iterable[int],
    scheduled: Mapping[int, object],
) -> int:
    """
    Node-packing lower bound (NP1).

    Extends NP0 by combining the NP packing value with a resource-projection
    term computed on the same packed set.
    """
    unscheduled_set = {int(a) for a in unscheduled}
    fixed_horizon = max((_entry_finish(entry) for entry in scheduled.values()), default=0)

    predecessors = build_predecessors(instance)
    order = topological_order(instance)
    successors = _build_successors_from_predecessors(predecessors, order)

    heads = _compute_heads(instance, scheduled, predecessors, order)
    tails = _compute_tails(instance, successors, order)
    durations: Dict[int, int] = {
        int(act_id): int(instance.activities[act_id].duration) for act_id in order
    }

    base_lb_cp = 0
    for act_id in order:
        candidate = int(heads[act_id]) + durations[act_id] + int(tails[act_id])
        if candidate > base_lb_cp:
            base_lb_cp = candidate

    best_lb = max(base_lb_cp, fixed_horizon)

    critical_set: Set[int] = set()
    for act_id in order:
        if int(heads[act_id]) + durations[act_id] + int(tails[act_id]) == base_lb_cp:
            critical_set.add(act_id)

    act_ids: List[int] = [
        act_id for act_id in order if act_id in unscheduled_set and durations[act_id] > 0
    ]
    if not act_ids:
        return best_lb

    reachable_from = _compute_reachability(order, successors)
    compatible: Dict[int, Set[int]] = {act_id: set() for act_id in act_ids}
    for idx_i, i in enumerate(act_ids):
        for j in act_ids[idx_i + 1 :]:
            if _are_activities_compatible(instance, i, j, reachable_from):
                compatible[i].add(j)
                compatible[j].add(i)

    act_ids.sort(
        key=lambda act_id: (
            -int(act_id in critical_set),
            len(compatible[act_id]),
            -durations[act_id],
            act_id,
        )
    )

    remaining = list(act_ids)
    packed_set: List[int] = []
    while remaining:
        i = remaining.pop(0)
        packed_set.append(i)
        remaining = [j for j in remaining if j not in compatible[i]]

    lb_np0_value = _node_packing_bound_value(
        packed_set=packed_set,
        heads=heads,
        tails=tails,
        durations=durations,
    )
    lb_rp_value = _resource_projection_bound(
        instance=instance,
        packed_set=packed_set,
        heads=heads,
        tails=tails,
        scheduled=scheduled,
    )
    candidate_lb = max(base_lb_cp, lb_np0_value, lb_rp_value)
    if candidate_lb > best_lb:
        best_lb = candidate_lb

    return max(best_lb, fixed_horizon)


def lb_np2(
    instance: RCPSPInstance,
    unscheduled: Iterable[int],
    scheduled: Mapping[int, object],
) -> int:
    """
    Node-packing lower bound (NP2).

    Extends NP0 by combining the NP packing value with a single-machine
    relaxation bound computed on the same packed set.
    """
    unscheduled_set = {int(a) for a in unscheduled}
    fixed_horizon = max((_entry_finish(entry) for entry in scheduled.values()), default=0)

    predecessors = build_predecessors(instance)
    order = topological_order(instance)
    successors = _build_successors_from_predecessors(predecessors, order)

    heads = _compute_heads(instance, scheduled, predecessors, order)
    tails = _compute_tails(instance, successors, order)
    durations: Dict[int, int] = {
        int(act_id): int(instance.activities[act_id].duration) for act_id in order
    }

    base_lb_cp = 0
    for act_id in order:
        candidate = int(heads[act_id]) + durations[act_id] + int(tails[act_id])
        if candidate > base_lb_cp:
            base_lb_cp = candidate

    best_lb = max(base_lb_cp, fixed_horizon)

    critical_set: Set[int] = set()
    for act_id in order:
        if int(heads[act_id]) + durations[act_id] + int(tails[act_id]) == base_lb_cp:
            critical_set.add(act_id)

    act_ids: List[int] = [
        act_id for act_id in order if act_id in unscheduled_set and durations[act_id] > 0
    ]
    if not act_ids:
        return best_lb

    reachable_from = _compute_reachability(order, successors)
    compatible: Dict[int, Set[int]] = {act_id: set() for act_id in act_ids}
    for idx_i, i in enumerate(act_ids):
        for j in act_ids[idx_i + 1 :]:
            if _are_activities_compatible(instance, i, j, reachable_from):
                compatible[i].add(j)
                compatible[j].add(i)

    act_ids.sort(
        key=lambda act_id: (
            -int(act_id in critical_set),
            len(compatible[act_id]),
            -durations[act_id],
            act_id,
        )
    )

    remaining = list(act_ids)
    packed_set: List[int] = []
    while remaining:
        i = remaining.pop(0)
        packed_set.append(i)
        remaining = [j for j in remaining if j not in compatible[i]]

    lb_np0_value = _node_packing_bound_value(
        packed_set=packed_set,
        heads=heads,
        tails=tails,
        durations=durations,
    )
    lb_sm_value = _single_machine_bound(
        packed_set=packed_set,
        heads=heads,
        tails=tails,
        durations=durations,
    )
    candidate_lb = max(base_lb_cp, lb_np0_value, lb_sm_value)
    if candidate_lb > best_lb:
        best_lb = candidate_lb

    return max(best_lb, fixed_horizon)


def lb_pr(
    instance: RCPSPInstance,
    unscheduled: Iterable[int],
    scheduled: Mapping[int, object],
) -> int:
    """
    Destructive precedence-reduction lower bound (PR).

    Iteratively tests incompatible pairs under a target horizon T. If one
    direction of an unordered pair is impossible under T, force the opposite
    precedence arc. If both directions are impossible, increase T and restart.
    """
    unscheduled_set = {int(a) for a in unscheduled}
    fixed_horizon = max((_entry_finish(entry) for entry in scheduled.values()), default=0)

    T = max(
        fixed_horizon,
        lb_cp(instance, unscheduled_set, scheduled),
        lb_cc(instance, unscheduled_set, scheduled),
    )
    activity_ids = list(instance.activities.keys())

    while True:
        preds = build_predecessors(instance)
        preds_cur = _copy_predecessors(preds, activity_ids)

        changed = True
        contradiction = False

        while changed and not contradiction:
            changed = False

            try:
                order_cur = _topological_order_from_predecessors(preds_cur, activity_ids)
            except ValueError:
                contradiction = True
                break

            succ_cur = _build_successors_from_predecessors(preds_cur, order_cur)
            heads = _compute_heads(instance, scheduled, preds_cur, order_cur)
            tails = _compute_tails(instance, succ_cur, order_cur)

            if _violates_horizon(instance, heads, tails, T):
                contradiction = True
                break

            reachable = _compute_reachability(order_cur, succ_cur)
            act_ids = [
                act_id
                for act_id in order_cur
                if act_id in unscheduled_set and int(instance.activities[act_id].duration) > 0
            ]

            restart_scan = False
            for idx_i, i in enumerate(act_ids):
                for j in act_ids[idx_i + 1 :]:
                    if not _resource_incompatible(instance, i, j):
                        continue

                    # Already ordered by current precedence closure.
                    if j in reachable.get(i, set()) or i in reachable.get(j, set()):
                        continue

                    can_ij = _test_pair_direction(
                        instance=instance,
                        scheduled=scheduled,
                        predecessors=preds_cur,
                        activities=order_cur,
                        horizon=T,
                        src=i,
                        dst=j,
                    )
                    can_ji = _test_pair_direction(
                        instance=instance,
                        scheduled=scheduled,
                        predecessors=preds_cur,
                        activities=order_cur,
                        horizon=T,
                        src=j,
                        dst=i,
                    )

                    if not can_ij and not can_ji:
                        contradiction = True
                        restart_scan = True
                        break

                    if can_ij and not can_ji:
                        preds_cur.setdefault(j, set()).add(i)
                        changed = True
                        restart_scan = True
                        break

                    if (not can_ij) and can_ji:
                        preds_cur.setdefault(i, set()).add(j)
                        changed = True
                        restart_scan = True
                        break

                if restart_scan:
                    break

        if contradiction:
            T += 1
            continue

        return max(T, fixed_horizon)


def lb_ct(
    instance: RCPSPInstance,
    unscheduled: Iterable[int],
    scheduled: Mapping[int, object],
) -> int:
    """
    Destructive core-times lower bound (CT).

    Increases horizon T until no contradiction is found from:
    - precedence timing inconsistency (ES > LS)
    - mandatory-part resource overload on elementary core intervals.
    """
    _ = unscheduled  # kept for interface consistency across all lower bounds

    fixed_horizon = max((_entry_finish(entry) for entry in scheduled.values()), default=0)
    T = max(
        fixed_horizon,
        lb_cp(instance, unscheduled, scheduled),
        lb_cc(instance, unscheduled, scheduled),
    )

    while True:
        predecessors = build_predecessors(instance)
        order = topological_order(instance)
        successors = _build_successors_from_predecessors(predecessors, order)

        es = _compute_heads(instance, scheduled, predecessors, order)
        ls = _compute_latest_starts(
            instance=instance,
            scheduled=scheduled,
            successors=successors,
            order=order,
            horizon=T,
        )

        contradiction = False
        for act_id in order:
            if int(es[act_id]) > int(ls[act_id]):
                contradiction = True
                break

        if contradiction:
            T += 1
            continue

        core_intervals: List[Tuple[int, int, int]] = []
        for act_id in order:
            dur = int(instance.activities[act_id].duration)
            if dur <= 0:
                continue

            cstart = int(ls[act_id])
            cend = int(es[act_id]) + dur
            if cstart < cend:
                core_intervals.append((act_id, cstart, cend))

        intervals = _build_elementary_intervals(core_intervals)
        for a, b in intervals:
            if b <= a:
                continue
            for r in range(instance.num_resources):
                usage = 0
                for act_id, cstart, cend in core_intervals:
                    if cstart <= a and b <= cend:
                        usage += int(instance.activities[act_id].resources[r])
                if usage > int(instance.resource_caps[r]):
                    contradiction = True
                    break
            if contradiction:
                break

        if contradiction:
            T += 1
            continue

        return max(T, fixed_horizon)


def lb_tp(
    instance: RCPSPInstance,
    unscheduled: Iterable[int],
    scheduled: Mapping[int, object],
) -> int:
    """
    Destructive time-period lower bound (TP).

    Increases horizon T until no contradiction remains from:
    - precedence timing inconsistency (ES > LS)
    - per-period resource energy overload.
    """
    _ = unscheduled  # kept for interface consistency across all lower bounds

    fixed_horizon = max((_entry_finish(entry) for entry in scheduled.values()), default=0)
    T = max(
        fixed_horizon,
        lb_cp(instance, unscheduled, scheduled),
        lb_cc(instance, unscheduled, scheduled),
    )

    while True:
        predecessors = build_predecessors(instance)
        order = topological_order(instance)
        successors = _build_successors_from_predecessors(predecessors, order)

        es = _compute_heads(instance, scheduled, predecessors, order)
        ls = _compute_latest_starts(
            instance=instance,
            scheduled=scheduled,
            successors=successors,
            order=order,
            horizon=T,
        )

        contradiction = False
        for act_id in order:
            if int(es[act_id]) > int(ls[act_id]):
                contradiction = True
                break

        if contradiction:
            T += 1
            continue

        durations: Dict[int, int] = {
            int(act_id): int(instance.activities[act_id].duration) for act_id in order
        }
        periods = _build_time_periods(
            es=es,
            ls=ls,
            durations=durations,
            horizon=T,
        )

        for a, b in periods:
            if b <= a:
                continue
            width = int(b) - int(a)
            for r in range(instance.num_resources):
                required = 0
                for act_id in order:
                    dur = int(instance.activities[act_id].duration)
                    req = int(instance.activities[act_id].resources[r])
                    if dur <= 0 or req <= 0:
                        continue
                    mandatory = _minimum_processing_in_period(
                        es_i=int(es[act_id]),
                        ls_i=int(ls[act_id]),
                        dur_i=dur,
                        a=int(a),
                        b=int(b),
                    )
                    required += mandatory * req

                available = width * int(instance.resource_caps[r])
                if required > available:
                    contradiction = True
                    break
            if contradiction:
                break

        if contradiction:
            T += 1
            continue

        return max(T, fixed_horizon)


def lb_cs(
    instance: RCPSPInstance,
    unscheduled: Iterable[int],
    scheduled: Mapping[int, object],
) -> int:
    """
    Critical-sequence lower bound.

    This bound extends the precedence critical-path value z by checking, for each
    non-critical unscheduled activity i, how much uninterrupted processing time can
    be embedded inside [ES_i, LF_i] when only the resource profile of one selected
    critical path is reserved. If i cannot fit fully in that interval/profile, then
    the project duration must be extended by at least d_i - e_i.
    """
    unscheduled_set: Set[int] = {int(a) for a in unscheduled}
    fixed_horizon = max((_entry_finish(entry) for entry in scheduled.values()), default=0)
    if not unscheduled_set:
        return fixed_horizon

    predecessors = build_predecessors(instance)
    order = topological_order(instance)

    # Forward pass with scheduled activities treated as fixed (same convention as lb_cp).
    es: Dict[int, int] = {}
    ef: Dict[int, int] = {}
    best_pred: Dict[int, int | None] = {}
    for act_id in order:
        preds = predecessors.get(act_id, set())
        pred_finish = 0
        pred_argmax: int | None = None
        for pred in preds:
            val = ef[pred]
            if pred_argmax is None or val > pred_finish:
                pred_finish = val
                pred_argmax = pred

        if act_id in scheduled:
            entry = scheduled[act_id]
            start = max(pred_finish, _entry_start(entry))
            finish = max(_entry_finish(entry), start + _entry_duration(entry))
        else:
            duration = int(instance.activities[act_id].duration)
            start = pred_finish
            finish = start + duration

        es[act_id] = int(start)
        ef[act_id] = int(finish)
        best_pred[act_id] = pred_argmax

    z = max(ef.values(), default=0)

    # Build successors and backward pass (precedence windows under horizon z).
    successors: Dict[int, List[int]] = {act_id: [] for act_id in order}
    for succ, preds in predecessors.items():
        for pred in preds:
            successors[pred].append(succ)

    lf: Dict[int, int] = {act_id: z for act_id in order}
    for act_id in reversed(order):
        succs = successors.get(act_id, [])
        if succs:
            latest = min(lf[s] - int(instance.activities[s].duration) for s in succs)
        else:
            latest = z
        # Guard against negative slack due to fixed starts in partial schedules.
        lf[act_id] = max(int(latest), int(ef[act_id]))

    # Recover one critical path by predecessor traceback from a max-EF sink.
    sink = max(order, key=lambda aid: (ef[aid], aid)) if order else None
    critical_path: Set[int] = set()
    cur = sink
    while cur is not None and cur not in critical_path:
        critical_path.add(cur)
        cur = best_pred.get(cur)

    # Resource profile induced by this critical path over [0, z).
    profile: List[List[int]] = [[0 for _ in range(instance.num_resources)] for _ in range(max(0, z))]
    for act_id in critical_path:
        start = es[act_id]
        finish = ef[act_id]
        if finish <= start:
            continue
        reqs = instance.activities[act_id].resources
        lo = max(0, int(start))
        hi = min(int(finish), z)
        for t in range(lo, hi):
            row = profile[t]
            for r in range(instance.num_resources):
                row[r] += int(reqs[r])

    current_lb = z
    for act_id in unscheduled_set:
        if act_id in critical_path:
            continue

        dur = int(instance.activities[act_id].duration)
        if dur <= 0:
            continue

        win_start = int(es[act_id])
        win_end = int(lf[act_id])
        if win_end <= win_start:
            e_i = 0
        else:
            reqs = instance.activities[act_id].resources
            best_run = 0
            run = 0
            for t in range(win_start, win_end):
                feasible = True
                for r in range(instance.num_resources):
                    used = profile[t][r] if 0 <= t < z else 0
                    if used + int(reqs[r]) > int(instance.resource_caps[r]):
                        feasible = False
                        break
                if feasible:
                    run += 1
                    if run > best_run:
                        best_run = run
                else:
                    run = 0
            e_i = best_run

        extension = max(0, dur - e_i)
        candidate = z + extension
        if candidate > current_lb:
            current_lb = candidate

    return max(fixed_horizon, current_lb)


# -----------------------------------------------------------------------------
# Registry
# -----------------------------------------------------------------------------

LOWER_BOUND_FNS: Dict[str, LowerBoundFn] = {
    "lb_cp": lb_cp,
    "lb_cc": lb_cc,
    "lb_ct": lb_ct,
    "lb_cs": lb_cs,
    "lb_ip0": lb_ip0,
    "lb_ip1": lb_ip1,
    "lb_np0": lb_np0,
    "lb_np1": lb_np1,
    "lb_np2": lb_np2,
    "lb_pr": lb_pr,
    "lb_pm1": lb_pm1,
    "lb_pm2": lb_pm2,
    "lb_tp": lb_tp,
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


def normalize_lower_bound_spec(lb_spec: object = DEFAULT_LOWER_BOUND_ID) -> List[str]:
    """
    Normalize lower-bound selection into a validated non-empty list of LB ids.

    Accepted forms:
    - "lb_cp"
    - "lb_cp,lb_cc"
    - ["lb_cp", "lb_cc"]
    """
    if lb_spec is None:
        raw_ids = [DEFAULT_LOWER_BOUND_ID]
    elif isinstance(lb_spec, str):
        parts = [part.strip() for part in lb_spec.split(",")]
        raw_ids = [part for part in parts if part]
    elif isinstance(lb_spec, Sequence):
        raw_ids = [str(item).strip() for item in lb_spec if str(item).strip()]
    else:
        raise TypeError("lower bound spec must be a string, list/tuple of strings, or None.")

    if not raw_ids:
        raise ValueError("lower bound spec cannot be empty.")

    known = set(LOWER_BOUND_FNS.keys())
    unknown = [lb_id for lb_id in raw_ids if lb_id not in known]
    if unknown:
        names = ", ".join(sorted(known))
        raise ValueError(f"Unknown lower bound id(s): {', '.join(unknown)}. Available: {names}")

    return raw_ids


def format_lower_bound_spec(lb_spec: object = DEFAULT_LOWER_BOUND_ID) -> str:
    ids = normalize_lower_bound_spec(lb_spec)
    if len(ids) == 1:
        return ids[0]
    return "max(" + ", ".join(ids) + ")"


def lower_bound(
    instance: RCPSPInstance,
    unscheduled: Iterable[int],
    scheduled: Mapping[int, object],
    lb_id: str = DEFAULT_LOWER_BOUND_ID,
) -> int:
    """
    Compute the requested lower bound.

    `lb_id` may be:
    - one lower-bound id (e.g. "lb_cp")
    - a comma-separated string (e.g. "lb_cp,lb_cc")
    - a sequence of ids (e.g. ["lb_cp", "lb_cc"])

    Composite semantics: max over selected bounds.
    """
    lb_ids = normalize_lower_bound_spec(lb_id)
    values = [
        get_lower_bound_fn(bound_id)(instance, unscheduled, scheduled)
        for bound_id in lb_ids
    ]
    return max(values)
