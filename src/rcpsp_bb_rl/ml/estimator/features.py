"""
Instance-level feature extraction for the search-effort estimator.

Maps one RCPSP instance to a fixed-length feature vector x(I) used to predict
log(1 + N_baseline(I)) — the expected node count at the locked episode horizon.

Features are static (no solver run, no live policy): they depend only on the
instance structure, so the same vector can be computed at dataset-generation
time and at inference time without skew.
"""

from __future__ import annotations

from typing import Dict, List, Mapping, Set

import numpy as np

from rcpsp_bb_rl.bnb.lower_bounds import lower_bound
from rcpsp_bb_rl.bnb.precedence import build_predecessors, build_successors, topological_order
from rcpsp_bb_rl.bnb.scheduling import earliest_feasible_start
from rcpsp_bb_rl.data.parsing import RCPSPInstance


# Fixed feature order. The estimator consumes vectors in exactly this order, so
# never reorder — append only.
FEATURE_NAMES: List[str] = [
    "n_activities",          # real activities (dummy source/sink excluded)
    "n_resources",           # renewable resources
    "mean_duration",
    "std_duration",
    "total_duration",
    "n_precedence_edges",    # direct arcs among real activities
    "precedence_density",    # edges / (n*(n-1)/2)
    "order_strength",        # transitive-closure related pairs / (n*(n-1)/2)
    "unordered_pair_ratio",  # 1 - order_strength
    "critical_path_length",  # precedence-only LB (lb_cp)
    "resource_factor",       # mean fraction of resources each activity uses
    "max_resource_pressure", # per-resource energy / (cap * CP), max over r
    "mean_resource_pressure",
    "n_conflict_pairs",      # unordered pairs that cannot co-execute on a resource
    "conflict_density",      # conflict pairs / unordered pairs
    "resource_lower_bound",  # resource-capacity LB (lb_cc)
    "heuristic_ub",          # serial SGS makespan (topological priority)
    "ub_lb_gap",             # heuristic_ub - max(CP, resource LB)
]

NUM_FEATURES = len(FEATURE_NAMES)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _real_activities(instance: RCPSPInstance) -> List[int]:
    """Real activity ids, excluding dummy source/sink (duration 0, no resources)."""
    real: List[int] = []
    for act_id, act in instance.activities.items():
        is_dummy = act.duration == 0 and all(r == 0 for r in act.resources)
        if not is_dummy:
            real.append(act_id)
    return sorted(real)


def _descendants(
    real: Set[int],
    successors: Mapping[int, List[int]],
    order: List[int],
) -> Dict[int, Set[int]]:
    """
    Transitive closure restricted to real activities.

    desc[u] = all real activities reachable from u. Dummies cannot be
    intermediate on a real-to-real path (source has no preds, sink no succs),
    so restricting to real ids is exact.
    """
    desc: Dict[int, Set[int]] = {u: set() for u in order}
    for u in reversed(order):
        acc: Set[int] = set()
        for v in successors.get(u, []):
            if v in real:
                acc.add(v)
            acc |= desc.get(v, set())
        desc[u] = acc
    return {u: (d & real) for u, d in desc.items()}


def _serial_sgs_makespan(instance: RCPSPInstance) -> int:
    """
    Heuristic upper bound via the serial schedule generation scheme.

    Activities are placed one at a time in topological order, each at its
    earliest precedence- and resource-feasible start. Returns the resulting
    makespan (max finish over all activities).
    """
    predecessors = build_predecessors(instance)
    order = topological_order(instance)
    scheduled: Dict[int, Dict[str, int]] = {}

    for act_id in order:
        duration = int(instance.activities[act_id].duration)
        start = earliest_feasible_start(
            instance, predecessors, scheduled, act_id, incumbent=None,
        )
        if start is None:
            # Should not happen for a feasible instance; fall back to precedence start.
            start = max(
                (scheduled[p]["finish"] for p in predecessors.get(act_id, set())),
                default=0,
            )
        scheduled[act_id] = {
            "start": int(start),
            "finish": int(start) + duration,
            "duration": duration,
        }

    return max((e["finish"] for e in scheduled.values()), default=0)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_features(instance: RCPSPInstance) -> np.ndarray:
    """Compute the fixed-order feature vector x(I) for one instance."""
    return np.asarray(
        [extract_feature_dict(instance)[name] for name in FEATURE_NAMES],
        dtype=np.float64,
    )


def extract_feature_dict(instance: RCPSPInstance) -> Dict[str, float]:
    """Compute named features for one instance (same values as extract_features)."""
    real = _real_activities(instance)
    n = len(real)
    m = int(instance.num_resources)
    caps = [int(c) for c in instance.resource_caps]
    real_set = set(real)

    durations = [int(instance.activities[a].duration) for a in real]
    total_duration = float(sum(durations))
    mean_duration = float(np.mean(durations)) if durations else 0.0
    std_duration = float(np.std(durations)) if durations else 0.0

    successors = build_successors(instance)

    # Direct precedence edges among real activities.
    n_edges = 0
    for u in real:
        for v in successors.get(u, []):
            if v in real_set:
                n_edges += 1

    total_pairs = n * (n - 1) / 2.0 if n > 1 else 1.0
    precedence_density = n_edges / total_pairs if n > 1 else 0.0

    # Order strength: fraction of real pairs related in the transitive closure.
    order = topological_order(instance)
    desc = _descendants(real_set, successors, order)
    n_related = sum(len(desc[u]) for u in real)  # each ordered pair counted once
    order_strength = n_related / total_pairs if n > 1 else 0.0
    unordered_pair_ratio = 1.0 - order_strength

    # Lower bounds.
    all_acts = set(instance.activities.keys())
    critical_path_length = float(lower_bound(instance, all_acts, {}, lb_id="lb_cp"))
    resource_lower_bound = float(lower_bound(instance, all_acts, {}, lb_id="lb_cc"))

    # Resource factor: mean fraction of resources each activity demands.
    if n > 0 and m > 0:
        resource_factor = float(
            np.mean([
                sum(1 for r in instance.activities[a].resources if r > 0) / m
                for a in real
            ])
        )
    else:
        resource_factor = 0.0

    # Resource pressure: per-resource energy demand / (capacity * critical path).
    cp_norm = critical_path_length if critical_path_length > 0 else 1.0
    pressures: List[float] = []
    for r in range(m):
        cap = caps[r] if r < len(caps) else 0
        if cap <= 0:
            pressures.append(0.0)
            continue
        energy = sum(
            int(instance.activities[a].resources[r]) * int(instance.activities[a].duration)
            for a in real
        )
        pressures.append(energy / (cap * cp_norm))
    max_resource_pressure = float(max(pressures)) if pressures else 0.0
    mean_resource_pressure = float(np.mean(pressures)) if pressures else 0.0

    # Resource conflicts: unordered pairs that exceed some capacity if run together.
    related = {u: desc[u] for u in real}
    n_conflict_pairs = 0
    for i, a in enumerate(real):
        ra = instance.activities[a].resources
        for b in real[i + 1:]:
            # Skip precedence-ordered pairs (they can never co-execute anyway).
            if b in related[a] or a in related[b]:
                continue
            rb = instance.activities[b].resources
            if any(
                int(ra[r]) + int(rb[r]) > (caps[r] if r < len(caps) else 0)
                for r in range(m)
            ):
                n_conflict_pairs += 1

    n_unordered = total_pairs * unordered_pair_ratio if n > 1 else 0.0
    conflict_density = n_conflict_pairs / n_unordered if n_unordered > 0 else 0.0

    heuristic_ub = float(_serial_sgs_makespan(instance))
    ub_lb_gap = heuristic_ub - max(critical_path_length, resource_lower_bound)

    return {
        "n_activities": float(n),
        "n_resources": float(m),
        "mean_duration": mean_duration,
        "std_duration": std_duration,
        "total_duration": total_duration,
        "n_precedence_edges": float(n_edges),
        "precedence_density": precedence_density,
        "order_strength": order_strength,
        "unordered_pair_ratio": unordered_pair_ratio,
        "critical_path_length": critical_path_length,
        "resource_factor": resource_factor,
        "max_resource_pressure": max_resource_pressure,
        "mean_resource_pressure": mean_resource_pressure,
        "n_conflict_pairs": float(n_conflict_pairs),
        "conflict_density": conflict_density,
        "resource_lower_bound": resource_lower_bound,
        "heuristic_ub": heuristic_ub,
        "ub_lb_gap": ub_lb_gap,
    }
