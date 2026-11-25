from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import torch

from rcpsp_bnb_rl.bnb.lower_bounds import (
    DEFAULT_LOWER_BOUND_ID,
    _build_successors_from_predecessors,
    _compute_heads,
    _compute_latest_starts,
    _compute_tails,
    _topological_order_from_predecessors,
)
from rcpsp_bnb_rl.bnb.precedence import build_predecessors
from rcpsp_bnb_rl.bnb.scheduling import ResourceProfile, build_profile
from rcpsp_bnb_rl.data.parsing import RCPSPInstance
from rcpsp_bnb_rl.ml.il.generate_trajectories import TrajectoryRecord


# ---------------------------------------------------------------------------
# Normalisation helpers
# ---------------------------------------------------------------------------

def _sum_all_durations(instance: RCPSPInstance) -> float:
    return float(sum(act.duration for act in instance.activities.values())) or 1.0



# ---------------------------------------------------------------------------
# Per-instance static bundle
# ---------------------------------------------------------------------------

@dataclass
class InstanceStatics:
    """
    Instance-invariant quantities shared across every node of one episode.

    These depend only on the RCPSP instance structure (precedence graph and
    durations), never on the partial schedule, so they are computed once per
    instance and reused for every NodeContext. Recomputing them per node was a
    major source of per-node overhead in policy-guided search.
    """
    predecessors: Dict[int, set]
    successors: Dict[int, set]
    descendants: Dict[int, set]
    topo_order: List[int]
    tails: Dict[int, int]
    succ_dur_sum: Dict[int, int]
    num_preds: Dict[int, int]
    sum_durations: float
    num_acts: int
    num_res: int
    caps: List[int]
    total_resource_energy: List[float]
    root_cp: int


def build_instance_statics(
    instance: RCPSPInstance,
    predecessors: Optional[Dict[int, set]] = None,
) -> InstanceStatics:
    """
    Compute the instance-invariant structures once. If `predecessors` is already
    available (e.g. from the solver), pass it in to avoid rebuilding it.
    """
    preds = predecessors if predecessors is not None else build_predecessors(instance)
    all_ids = list(instance.activities.keys())
    successors = _build_successors_from_predecessors(preds, all_ids)
    topo_order = _topological_order_from_predecessors(preds, all_ids)
    # Transitive descendants are instance-invariant and are used by the
    # downstream packing features at every rollout node.  Compute them once
    # in reverse topological order instead of traversing the graph per node.
    descendants: Dict[int, set] = {int(a): set() for a in all_ids}
    for act_id in reversed(topo_order):
        for succ in successors.get(act_id, set()):
            descendants[act_id].add(int(succ))
            descendants[act_id].update(descendants.get(int(succ), set()))
    tails = _compute_tails(instance, successors, topo_order)
    succ_dur_sum = {
        a: sum(instance.activities[s].duration for s in instance.activities[a].successors)
        for a in instance.activities
    }
    num_preds = {a: len(preds.get(a, set())) for a in instance.activities}
    root_heads = _compute_heads(instance, {}, preds, topo_order)

    return InstanceStatics(
        predecessors=preds,
        successors=successors,
        descendants=descendants,
        topo_order=topo_order,
        tails=tails,
        succ_dur_sum=succ_dur_sum,
        num_preds=num_preds,
        sum_durations=_sum_all_durations(instance),
        num_acts=instance.num_activities,
        num_res=instance.num_resources,
        caps=instance.resource_caps,
        total_resource_energy=[
            float(sum(act.duration * act.resources[r] for act in instance.activities.values()))
            for r in range(instance.num_resources)
        ],
        root_cp=max((root_heads[a] + instance.activities[a].duration + tails[a]
                     for a in instance.activities), default=0),
    )


# ---------------------------------------------------------------------------
# Precomputed node context
# ---------------------------------------------------------------------------

class NodeContext:
    """
    All scheduling-theoretic quantities needed for featurisation, computed
    once per node and shared across all candidates.

    Parameters
    ----------
    instance      : the RCPSP instance
    scheduled     : dict mapping act_id -> ScheduleEntry (or dict with start/finish/duration)
    unscheduled   : set of unscheduled activity ids
    ready         : set of ready activity ids
    lower_bound   : current node lower bound
    earliest_starts : dict mapping ready act_id -> resource-feasible earliest start (None = infeasible)
    """

    def __init__(
        self,
        instance: RCPSPInstance,
        scheduled: Dict,
        unscheduled: set,
        ready: set,
        lower_bound: int,
        earliest_starts: Dict[int, Optional[int]],
        statics: Optional[InstanceStatics] = None,
        profile: Optional[ResourceProfile] = None,
    ) -> None:
        self.instance = instance
        self.scheduled = scheduled
        self.unscheduled = unscheduled
        self.ready = ready
        self.lower_bound = lower_bound
        self.earliest_starts = earliest_starts

        # Instance-invariant structures: reuse the shared bundle when provided,
        # otherwise build them for a standalone context.
        if statics is None:
            statics = build_instance_statics(instance)

        # Derived instance-level constants
        self.sum_durations: float = statics.sum_durations
        self.num_acts: int = statics.num_acts
        self.num_res: int = statics.num_res
        self.caps: List[int] = statics.caps
        self.total_resource_energy = statics.total_resource_energy
        self.root_cp = statics.root_cp

        # Precedence structures (static)
        self.predecessors = statics.predecessors
        self.successors = statics.successors
        self.descendants = statics.descendants
        self.topo_order = statics.topo_order

        # Heads (earliest starts under precedence) — DYNAMIC (depends on scheduled).
        self.heads: Dict[int, int] = _compute_heads(
            instance, scheduled, self.predecessors, self.topo_order
        )
        # Tails (longest chain after) — STATIC.
        self.tails: Dict[int, int] = statics.tails

        # The target horizon is a feature reference, never a pruning deadline.
        self.cp_value: int = max(
            (self.heads[a] + instance.activities[a].duration + self.tails[a]
             for a in instance.activities), default=0,
        )
        self.target_horizon = max(lower_bound, self.cp_value)
        self.latest_starts: Dict[int, int] = _compute_latest_starts(
            instance=instance,
            scheduled=scheduled,
            successors=self.successors,
            order=self.topo_order,
            horizon=self.target_horizon,
        )

        # Current makespan of partial schedule
        self.makespan_so_far: int = (
            max((self._entry_finish(e) for e in scheduled.values()), default=0)
        )

        self.min_est_ready = min(
            (earliest_starts[a] for a in ready if earliest_starts.get(a) is not None),
            default=self.makespan_so_far,
        )
        # Feature-only horizon remaining after the earliest ready insertion;
        # this is deliberately separate from all solver pruning deadlines.
        self.rem_horizon = max(1.0, float(self.target_horizon - self.min_est_ready))
        self.profile = profile if profile is not None else build_profile(
            instance.activities, instance.resource_caps, scheduled,
            horizon=max(int(self.sum_durations), self.makespan_so_far),
        )
        # Occupancy at the earliest feasible candidate insertion time.
        self.res_used: List[int] = self._compute_res_used()
        self._resource_features: Optional[List[List[float]]] = None
        self._interaction_features: Dict[int, List[List[float]]] = {}

        # Remaining energy per resource
        self.rem_energy: List[float] = [
            float(sum(
                instance.activities[a].duration * instance.activities[a].resources[r]
                for a in unscheduled
            ))
            for r in range(self.num_res)
        ]

        # Successor duration sums (direct successors only) — STATIC.
        self.succ_dur_sum: Dict[int, int] = statics.succ_dur_sum

        # Predecessor counts — STATIC.
        self.num_preds: Dict[int, int] = statics.num_preds

        # Resource conflict map among ready activities
        self.conflict_counts, self.conflict_load = self._compute_ready_conflicts()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _entry_finish(entry) -> int:
        if isinstance(entry, dict):
            return int(entry["finish"])
        return int(entry.finish)

    def _compute_res_used(self) -> List[int]:
        return [self.occupancy_at(self.min_est_ready, r) for r in range(self.num_res)]

    def occupancy_at(self, time: int, resource: int) -> int:
        """Read the pre-action profile; outside its horizon occupancy is zero."""
        row = self.profile.usage[resource]
        return row[time] if 0 <= time < len(row) else 0

    def _compute_ready_conflicts(self) -> Tuple[Dict[int, int], Dict[int, List[float]]]:
        """
        For each ready activity, count how many other ready activities share a
        resource bottleneck with it, and compute the aggregate conflicting demand
        per resource (normalised by cap).
        """
        ready_list = list(self.ready)
        conflict_counts: Dict[int, int] = {a: 0 for a in ready_list}
        conflict_load: Dict[int, List[float]] = {a: [0.0] * self.num_res for a in ready_list}

        for i, a in enumerate(ready_list):
            for b in ready_list[i + 1:]:
                conflicts = False
                for r in range(self.num_res):
                    cap = self.caps[r]
                    if cap > 0:
                        da = self.instance.activities[a].resources[r]
                        db = self.instance.activities[b].resources[r]
                        if da + db > cap:
                            conflicts = True
                            conflict_load[a][r] += db / cap
                            conflict_load[b][r] += da / cap
                if conflicts:
                    conflict_counts[a] += 1
                    conflict_counts[b] += 1

        return conflict_counts, conflict_load


# ---------------------------------------------------------------------------
# Variable-resource (set encoder) features
# ---------------------------------------------------------------------------
#
# resource_set: time quantities use S=sum(durations), counts use N=activities,
# demand and occupancy use resource capacity, and energy pressure uses root CP.
# Dynamic and exclusive pressure use H=max(1, target_horizon-min_est_ready).
# Downstream energy uses cap*root_cp; direct-successor demand uses cap.
# Pressures and downstream loads are unclamped; only occupancy is clamped.

FEATURE_SCHEMA = "resource_set"
SET_GLOBAL_FEATURE_NAMES = [
    "frac_scheduled", "makespan_so_far_norm", "lb_norm", "lb_gap_norm",
    "rem_dur_sum_norm", "rem_dur_mean_norm", "rem_dur_max_norm", "num_ready_norm",
    "n_res_norm", "max_energy_pressure", "mean_occ_at_min_est", "max_occ_at_min_est",
    "min_est_ready_norm", "max_dyn_rem_pressure",
]
SET_CANDIDATE_FEATURE_NAMES = [
    "duration_norm", "res_intensity", "total_resource_load",
    "head_norm", "tail_norm", "head_dur_tail_norm", "is_critical",
    "num_succ_norm", "num_pred_norm", "succ_dur_sum_norm",
    "est_norm", "est_minus_head_norm", "finish_norm", "conflict_count_norm",
    "slack_to_lb_signed", "ls_norm", "time_window_signed", "is_urgent",
    "makespan_ext_norm", "rel_est_norm", "bottleneck_weighted_desc_load",
]
SET_RESOURCE_FEATURE_NAMES = [
    "is_real_resource", "occ_frac_at_min_est", "avail_frac_at_min_est",
    "rem_energy_norm", "energy_pressure", "dyn_rem_energy_pressure",
    "exclusive_load_pressure",
]
SET_INTERACTION_FEATURE_NAMES = [
    "demand_frac", "conflict_load_norm", "has_demand", "rem_after_demand_at_est_frac",
    "desc_demand_energy_frac", "direct_succ_demand_frac", "is_heavy_demand",
    "mean_rem_after_demand_interval_frac",
]


def set_global_feature_dim() -> int:
    return len(SET_GLOBAL_FEATURE_NAMES)


def set_candidate_feature_dim() -> int:
    return len(SET_CANDIDATE_FEATURE_NAMES)


def set_resource_feature_dim() -> int:
    return len(SET_RESOURCE_FEATURE_NAMES)


def set_interaction_feature_dim() -> int:
    return len(SET_INTERACTION_FEATURE_NAMES)


def resource_set_features(ctx: NodeContext) -> List[List[float]]:
    """Resource occupancy, static/dynamic energy and exclusive-load pressure."""
    if ctx._resource_features is not None:
        return ctx._resource_features
    if ctx.num_res == 0:
        ctx._resource_features = [[0.0] * set_resource_feature_dim()]
        return ctx._resource_features
    out: List[List[float]] = []
    for r in range(ctx.num_res):
        cap = float(ctx.caps[r]) if ctx.caps[r] > 0 else 1.0
        occupancy = min(1.0, max(0.0, float(ctx.res_used[r]) / cap))
        dynamic = float(ctx.rem_energy[r]) / (cap * ctx.rem_horizon) if ctx.caps[r] > 0 else 0.0
        exclusive = sum(
            float(ctx.instance.activities[a].duration)
            for a in ctx.unscheduled
            if ctx.instance.activities[a].resources[r] > ctx.caps[r] / 2
        ) / ctx.rem_horizon if ctx.caps[r] > 0 else 0.0
        out.append([
            1.0, occupancy, 1.0 - occupancy,
            float(ctx.rem_energy[r]) / (cap * ctx.sum_durations),
            ctx.total_resource_energy[r] / (cap * max(1, ctx.root_cp)),
            dynamic, exclusive,
        ])
    ctx._resource_features = out
    return out


def candidate_resource_features(ctx: NodeContext, act_id: int) -> List[List[float]]:
    """Packing and unscheduled downstream load, in SET_INTERACTION_FEATURE_NAMES order.

    Direct successors may still have other unsatisfied predecessors. Interval
    residuals average the pre-action profile over [EST, EST + duration).
    """
    cached = ctx._interaction_features.get(act_id)
    if cached is not None:
        return cached
    act = ctx.instance.activities[act_id]
    if ctx.num_res == 0:
        out = [[0.0] * set_interaction_feature_dim()]
        ctx._interaction_features[act_id] = out
        return out
    raw_conflict = ctx.conflict_load.get(act_id, [0.0] * ctx.num_res)
    est = ctx.earliest_starts.get(act_id)
    time = ctx.min_est_ready if est is None else est
    descendants = ctx.descendants.get(act_id, set()).intersection(ctx.unscheduled)
    direct_successors = ctx.successors.get(act_id, set()).intersection(ctx.unscheduled)
    duration = max(0, int(act.duration))
    out: List[List[float]] = []
    for r in range(ctx.num_res):
        cap = float(ctx.caps[r]) if ctx.caps[r] > 0 else 1.0
        demand = float(act.resources[r])
        residual = (cap - ctx.occupancy_at(time, r) - demand) / cap
        if duration > 0 and ctx.caps[r] > 0:
            interval = sum(
                (cap - ctx.occupancy_at(time + t, r) - demand) / cap
                for t in range(duration)
            ) / duration
        else:
            interval = residual
        desc_energy = sum(
            float(ctx.instance.activities[d].duration * ctx.instance.activities[d].resources[r])
            for d in descendants
        ) / (cap * ctx.root_cp) if ctx.caps[r] > 0 and ctx.root_cp > 0 else 0.0
        succ_demand = sum(
            float(ctx.instance.activities[s].resources[r]) for s in direct_successors
        ) / cap if ctx.caps[r] > 0 else 0.0
        out.append([
            demand / cap,
            float(raw_conflict[r]) if r < len(raw_conflict) else 0.0,
            1.0 if demand > 0 else 0.0,
            residual, desc_energy, succ_demand,
            1.0 if ctx.caps[r] > 0 and demand > ctx.caps[r] / 2 else 0.0,
            interval,
        ])
    ctx._interaction_features[act_id] = out
    return out


def global_set_features(ctx: NodeContext) -> List[float]:
    """Emit SET_GLOBAL_FEATURE_NAMES in their declared order."""
    S = ctx.sum_durations
    N = float(ctx.num_acts) or 1.0
    resources = resource_set_features(ctx)
    physical = resources if ctx.num_res else []
    occupancies = [row[1] for row in physical]
    pressure = [row[4] for row in physical]
    dynamic_pressure = [row[5] for row in physical]
    rem_durs = [ctx.instance.activities[a].duration for a in ctx.unscheduled]
    return [
        len(ctx.scheduled) / N,
        ctx.makespan_so_far / S,
        ctx.lower_bound / S,
        max(0.0, ctx.lower_bound - ctx.makespan_so_far) / S,
        sum(rem_durs) / S if rem_durs else 0.0,
        (sum(rem_durs) / len(rem_durs)) / S if rem_durs else 0.0,
        max(rem_durs) / S if rem_durs else 0.0,
        len(ctx.ready) / N,
        min(1.0, ctx.num_res / N),
        max(pressure, default=0.0),
        sum(occupancies) / len(occupancies) if occupancies else 0.0,
        max(occupancies, default=0.0),
        ctx.min_est_ready / S,
        max(dynamic_pressure, default=0.0),
    ]


def candidate_set_features(ctx: NodeContext, act_id: int) -> List[float]:
    """Emit SET_CANDIDATE_FEATURE_NAMES, retaining signed target violations."""
    S = ctx.sum_durations
    N = float(ctx.num_acts) or 1.0
    act = ctx.instance.activities[act_id]
    demands = [
        float(act.resources[r]) / (float(ctx.caps[r]) if ctx.caps[r] > 0 else 1.0)
        for r in range(ctx.num_res)
    ]
    head = float(ctx.heads.get(act_id, 0))
    tail = float(ctx.tails.get(act_id, 0))
    dur = float(act.duration)
    hdt = head + dur + tail
    est = ctx.earliest_starts.get(act_id)
    if est is not None:
        est_f = float(est)
        est_norm = est_f / S
        est_minus_head = max(0.0, est_f - head) / S
        finish_norm = (est_f + dur) / S
        slack_to_lb = (ctx.lower_bound - (est_f + dur)) / S
        ls = float(ctx.latest_starts.get(act_id, est_f))
        ls_norm = ls / S
        tw = (ls - est_f) / S
        is_urgent = 1.0 if tw < (1.0 / N) else 0.0
        extension = max(0.0, est_f + dur - ctx.makespan_so_far) / S
        relative_est = (est_f - ctx.min_est_ready) / S
    else:
        est_norm, est_minus_head, finish_norm = -1.0, 0.0, -1.0
        slack_to_lb, ls_norm, tw, is_urgent = 0.0, -1.0, 0.0, 1.0
        extension, relative_est = 0.0, 0.0
    # Derived candidate summary of downstream energy weighted by live pressure.
    interactions = candidate_resource_features(ctx, act_id)
    resource_features = resource_set_features(ctx)
    weighted_downstream = sum(
        row[4] * resource_features[r][5]
        for r, row in enumerate(interactions)
        if r < len(resource_features) and r < len(ctx.caps) and ctx.caps[r] > 0
    )
    return [
        dur / S, max(demands, default=0.0), sum(demands),
        head / S, tail / S, hdt / S,
        1.0 if int(hdt) >= ctx.cp_value else 0.0,
        len(act.successors) / N, ctx.num_preds.get(act_id, 0) / N,
        float(ctx.succ_dur_sum.get(act_id, 0)) / S,
        est_norm, est_minus_head, finish_norm,
        ctx.conflict_counts.get(act_id, 0) / (float(len(ctx.ready)) or 1.0),
        slack_to_lb, ls_norm, tw, is_urgent, extension, relative_est,
        weighted_downstream,
    ]


def featurize_states_resource_set(
    records: Sequence[TrajectoryRecord],
    instance: RCPSPInstance,
    *,
    lb_spec: object = DEFAULT_LOWER_BOUND_ID,
) -> Dict[str, object]:
    """Rebuild teacher inputs through the same incumbent-free Actor path."""
    from rcpsp_bnb_rl.ml.observations import build_actor_observation

    statics = build_instance_statics(instance)
    cand_feats: List[List[float]] = []
    glob_feats: List[List[float]] = []
    lengths: List[int] = []
    targets: List[int] = []
    depths: List[int] = []
    resource_batches: List[torch.Tensor] = []
    interaction_batches: List[torch.Tensor] = []

    for rec in records:
        ready_sorted = sorted(rec.ready)
        if not ready_sorted:
            continue
        try:
            target_idx = ready_sorted.index(rec.action_task)
        except ValueError:
            continue
        depth = int(rec.raw.get("depth", 0))
        observation = build_actor_observation(
            instance,
            {int(k): value for k, value in rec.raw.get("scheduled", {}).items()},
            {int(a) for a in rec.raw.get("unscheduled", [])},
            set(ready_sorted), depth, statics=statics, lb_spec=lb_spec,
        )
        lengths.append(len(ready_sorted))
        targets.append(target_idx)
        depths.append(depth)
        glob_feats.append(observation["global_feats"].tolist())
        resource_batches.append(observation["resource_feats"])
        interaction_batches.append(observation["candidate_resource_feats"])
        cand_feats.extend(observation["candidate_feats"].tolist())

    return {
        "candidate_feats": torch.tensor(cand_feats, dtype=torch.float32).reshape(-1, set_candidate_feature_dim()),
        "global_feats": torch.tensor(glob_feats, dtype=torch.float32).reshape(-1, set_global_feature_dim()),
        "lengths": torch.tensor(lengths, dtype=torch.long),
        "targets": torch.tensor(targets, dtype=torch.long),
        "depths": torch.tensor(depths, dtype=torch.long),
        "resource_feats": resource_batches,
        "candidate_resource_feats": interaction_batches,
    }
