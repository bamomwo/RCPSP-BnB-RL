from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import torch

from rcpsp_bb_rl.bnb.lower_bounds import (
    _build_successors_from_predecessors,
    _compute_heads,
    _compute_latest_starts,
    _compute_tails,
    _topological_order_from_predecessors,
)
from rcpsp_bb_rl.bnb.precedence import build_predecessors
from rcpsp_bb_rl.data.parsing import RCPSPInstance
from rcpsp_bb_rl.ml.il.generate_trajectories import TrajectoryRecord


# ---------------------------------------------------------------------------
# Normalisation helpers
# ---------------------------------------------------------------------------

def _sum_all_durations(instance: RCPSPInstance) -> float:
    return float(sum(act.duration for act in instance.activities.values())) or 1.0


def _pad(values: Sequence[float], target_len: int) -> List[float]:
    arr = list(values[:target_len])
    arr.extend([0.0] * (target_len - len(arr)))
    return arr


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
    incumbent     : best known makespan so far (None if not yet found)
    earliest_starts : dict mapping ready act_id -> resource-feasible earliest start (None = infeasible)
    """

    def __init__(
        self,
        instance: RCPSPInstance,
        scheduled: Dict,
        unscheduled: set,
        ready: set,
        lower_bound: int,
        incumbent: Optional[int],
        earliest_starts: Dict[int, Optional[int]],
    ) -> None:
        self.instance = instance
        self.scheduled = scheduled
        self.unscheduled = unscheduled
        self.ready = ready
        self.lower_bound = lower_bound
        self.incumbent = incumbent
        self.earliest_starts = earliest_starts

        # Derived instance-level constants
        self.sum_durations: float = _sum_all_durations(instance)
        self.num_acts: int = instance.num_activities
        self.num_res: int = instance.num_resources
        self.caps: List[int] = instance.resource_caps

        # Precedence structures
        self.predecessors = build_predecessors(instance)
        all_ids = list(instance.activities.keys())
        self.successors = _build_successors_from_predecessors(self.predecessors, all_ids)
        self.topo_order = _topological_order_from_predecessors(self.predecessors, all_ids)

        # Heads (earliest starts under precedence) and tails (longest chain after)
        self.heads: Dict[int, int] = _compute_heads(
            instance, scheduled, self.predecessors, self.topo_order
        )
        self.tails: Dict[int, int] = _compute_tails(
            instance, self.successors, self.topo_order
        )

        # Latest starts under current lower bound as horizon
        horizon = incumbent if incumbent is not None else int(self.sum_durations)
        self.latest_starts: Dict[int, int] = _compute_latest_starts(
            instance=instance,
            scheduled=scheduled,
            successors=self.successors,
            order=self.topo_order,
            horizon=horizon,
        )

        # Critical path value (max head + dur + tail over all activities)
        self.cp_value: int = max(
            self.heads[a] + instance.activities[a].duration + self.tails[a]
            for a in instance.activities
        )

        # Current makespan of partial schedule
        self.makespan_so_far: int = (
            max((self._entry_finish(e) for e in scheduled.values()), default=0)
        )

        # Resource usage at makespan_so_far
        self.res_used: List[int] = self._compute_res_used()

        # Remaining energy per resource
        self.rem_energy: List[float] = [
            float(sum(
                instance.activities[a].duration * instance.activities[a].resources[r]
                for a in unscheduled
            ))
            for r in range(self.num_res)
        ]

        # Successor duration sums (direct successors only)
        self.succ_dur_sum: Dict[int, int] = {
            a: sum(instance.activities[s].duration for s in instance.activities[a].successors)
            for a in instance.activities
        }

        # Predecessor counts
        self.num_preds: Dict[int, int] = {
            a: len(self.predecessors.get(a, set()))
            for a in instance.activities
        }

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

    @staticmethod
    def _entry_start(entry) -> int:
        if isinstance(entry, dict):
            return int(entry["start"])
        return int(entry.start)

    def _compute_res_used(self) -> List[int]:
        t = self.makespan_so_far
        usage = [0] * self.num_res
        for act_id, entry in self.scheduled.items():
            s = self._entry_start(entry)
            f = self._entry_finish(entry)
            if s <= t < f:
                for r in range(self.num_res):
                    usage[r] += self.instance.activities[act_id].resources[r]
        return usage

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
# Global features  (one vector per node, fed into the CLS token)
# ---------------------------------------------------------------------------

# Number of global features = 6 + 4*max_resources + 4
GLOBAL_FEATURE_NAMES = [
    # Search progress
    "depth_norm",
    "frac_scheduled",
    "frac_unscheduled",
    "makespan_so_far_norm",
    "lb_norm",
    "lb_gap_norm",
    "incumbent_gap_norm",
    # Resource state  (repeated max_resources times each)
    # "cap_r", "res_util_r", "res_avail_r", "rem_energy_norm_r"
    # Remaining work
    "rem_dur_sum_norm",
    "rem_dur_mean_norm",
    "rem_dur_max_norm",
    "num_ready_norm",
]


def global_features(ctx: NodeContext, max_resources: int, depth: int = 0) -> List[float]:
    S = ctx.sum_durations
    N = float(ctx.num_acts) or 1.0
    n_sched = float(len(ctx.scheduled))
    n_unsched = float(len(ctx.unscheduled))

    lb_gap = max(0.0, ctx.lower_bound - ctx.makespan_so_far) / S
    if ctx.incumbent is not None and ctx.lower_bound > 0:
        inc_gap = max(0.0, ctx.incumbent - ctx.lower_bound) / ctx.lower_bound
    else:
        inc_gap = 0.0

    feats: List[float] = [
        float(depth) / N,
        n_sched / N,
        n_unsched / N,
        ctx.makespan_so_far / S,
        ctx.lower_bound / S,
        lb_gap,
        inc_gap,
    ]

    # Per-resource state (padded / truncated to max_resources)
    for r in range(max_resources):
        if r < ctx.num_res:
            cap = float(ctx.caps[r]) or 1.0
            used = float(ctx.res_used[r])
            rem_e = ctx.rem_energy[r]
            feats.append(cap)
            feats.append(used / cap)
            feats.append((cap - used) / cap)
            feats.append(rem_e / (cap * S))
        else:
            feats.extend([0.0, 0.0, 0.0, 0.0])

    # Remaining work summary
    rem_durs = [ctx.instance.activities[a].duration for a in ctx.unscheduled]
    if rem_durs:
        feats.append(sum(rem_durs) / S)
        feats.append((sum(rem_durs) / len(rem_durs)) / S)
        feats.append(max(rem_durs) / S)
    else:
        feats.extend([0.0, 0.0, 0.0])
    feats.append(float(len(ctx.ready)) / N)

    return feats


# ---------------------------------------------------------------------------
# Candidate features  (one vector per ready activity)
# ---------------------------------------------------------------------------

# Number of candidate features = 8 + 4*max_resources + 7 + 7 + (1 + max_resources)
CANDIDATE_FEATURE_NAMES = [
    # Intrinsic
    "duration_norm",
    "res_intensity",
    "total_resource_load",
    # per resource: "res_demand_r"
    # Precedence structure
    "head_norm",
    "tail_norm",
    "head_dur_tail_norm",
    "is_critical",
    "num_successors_norm",
    "num_predecessors_norm",
    "succ_dur_sum_norm",
    # Scheduling feasibility
    "earliest_start_norm",
    "est_minus_head_norm",
    "finish_norm",
    "slack_to_lb_norm",
    "latest_start_norm",
    "time_window_norm",
    "is_urgent",
    # Contention
    "num_conflicting_ready_norm",
    # per resource: "max_conflict_load_r"
]


def candidate_features(
    ctx: NodeContext,
    act_id: int,
    max_resources: int,
) -> List[float]:
    S = ctx.sum_durations
    N = float(ctx.num_acts) or 1.0
    instance = ctx.instance
    act = instance.activities[act_id]

    dur = float(act.duration)
    dur_norm = dur / S

    # Per-resource demand (normalised by cap)
    res_demand: List[float] = []
    for r in range(max_resources):
        if r < ctx.num_res:
            cap = float(ctx.caps[r]) or 1.0
            res_demand.append(act.resources[r] / cap)
        else:
            res_demand.append(0.0)

    res_intensity = max(res_demand)
    total_load = sum(res_demand)

    # Precedence
    head = float(ctx.heads.get(act_id, 0))
    tail = float(ctx.tails.get(act_id, 0))
    hdt = head + dur + tail
    is_critical = 1.0 if int(hdt) >= ctx.cp_value else 0.0
    num_succ = float(len(act.successors)) / N
    num_pred = float(ctx.num_preds.get(act_id, 0)) / N
    succ_dur = float(ctx.succ_dur_sum.get(act_id, 0)) / S

    # Scheduling feasibility
    est = ctx.earliest_starts.get(act_id)
    if est is not None:
        est_f = float(est)
        est_norm = est_f / S
        est_minus_head = max(0.0, est_f - head) / S
        finish_norm = (est_f + dur) / S
        slack_to_lb = max(0.0, ctx.lower_bound - (est_f + dur)) / S
        ls = float(ctx.latest_starts.get(act_id, est_f))
        ls_norm = ls / S
        tw = max(0.0, ls - est_f) / S
        is_urgent = 1.0 if tw < (1.0 / N) else 0.0
    else:
        # Infeasible candidate — sentinel values
        est_norm = -1.0
        est_minus_head = 0.0
        finish_norm = -1.0
        slack_to_lb = 0.0
        ls_norm = -1.0
        tw = 0.0
        is_urgent = 1.0

    # Contention
    n_ready = float(len(ctx.ready)) or 1.0
    conflict_count_norm = ctx.conflict_counts.get(act_id, 0) / n_ready
    conflict_load_per_res: List[float] = []
    raw_load = ctx.conflict_load.get(act_id, [0.0] * ctx.num_res)
    for r in range(max_resources):
        conflict_load_per_res.append(raw_load[r] if r < ctx.num_res else 0.0)

    feats: List[float] = [
        dur_norm,
        res_intensity,
        total_load,
        *res_demand,
        head / S,
        tail / S,
        hdt / S,
        is_critical,
        num_succ,
        num_pred,
        succ_dur,
        est_norm,
        est_minus_head,
        finish_norm,
        slack_to_lb,
        ls_norm,
        tw,
        is_urgent,
        conflict_count_norm,
        *conflict_load_per_res,
    ]
    return feats


# ---------------------------------------------------------------------------
# Feature dimension helpers
# ---------------------------------------------------------------------------

def global_feature_dim(max_resources: int) -> int:
    return 7 + 4 * max_resources + 4


def candidate_feature_dim(max_resources: int) -> int:
    # 3 intrinsic scalars + max_resources demand + 7 precedence + 7 feasibility + 1 + max_resources contention
    return 3 + max_resources + 7 + 7 + 1 + max_resources


# ---------------------------------------------------------------------------
# Batch featurisation from TrajectoryRecord (IL training path)
# ---------------------------------------------------------------------------

def _ctx_from_record(record: TrajectoryRecord, instance: RCPSPInstance) -> NodeContext:
    """Reconstruct a NodeContext from a logged TrajectoryRecord."""
    raw = record.raw

    # Rebuild scheduled as plain dicts (already that format in JSONL)
    scheduled = {int(k): v for k, v in raw.get("scheduled", {}).items()}
    unscheduled = {int(a) for a in raw.get("unscheduled", [])}
    ready = {int(a) for a in raw.get("ready", [])}
    lb = int(raw.get("lower_bound", 0))
    incumbent = raw.get("incumbent", None)
    if incumbent is not None:
        incumbent = int(incumbent)

    earliest_starts: Dict[int, Optional[int]] = {
        int(k): (None if v is None else int(v))
        for k, v in raw.get("earliest_start", {}).items()
    }

    return NodeContext(
        instance=instance,
        scheduled=scheduled,
        unscheduled=unscheduled,
        ready=ready,
        lower_bound=lb,
        incumbent=incumbent,
        earliest_starts=earliest_starts,
    )


def featurize_states(
    records: Sequence[TrajectoryRecord],
    instance: RCPSPInstance,
    max_resources: int,
) -> Dict[str, torch.Tensor]:
    """
    Convert a batch of TrajectoryRecord instances into tensors for listwise training.

    Returns
    -------
    {
        "candidate_feats": [Nc, Fc],
        "global_feats":    [Nc, Fg],   # same global repeated per candidate
        "lengths":         [B],
        "targets":         [B],
        "depths":          [B],
    }
    """
    cand_feats: List[List[float]] = []
    glob_feats: List[List[float]] = []
    lengths: List[int] = []
    targets: List[int] = []
    depths: List[int] = []

    for rec in records:
        ready_sorted = rec.ready
        if not ready_sorted:
            continue
        try:
            target_idx = ready_sorted.index(rec.action_task)
        except ValueError:
            continue

        ctx = _ctx_from_record(rec, instance)
        depth = int(rec.raw.get("depth", 0))
        glob = global_features(ctx, max_resources, depth=depth)

        lengths.append(len(ready_sorted))
        targets.append(target_idx)
        depths.append(depth)

        for act_id in ready_sorted:
            cand_feats.append(candidate_features(ctx, act_id, max_resources))
            glob_feats.append(glob)

    return {
        "candidate_feats": torch.tensor(cand_feats, dtype=torch.float32),
        "global_feats": torch.tensor(glob_feats, dtype=torch.float32),
        "lengths": torch.tensor(lengths, dtype=torch.long),
        "targets": torch.tensor(targets, dtype=torch.long),
        "depths": torch.tensor(depths, dtype=torch.long),
    }


def collate_state_batch(
    batch: Sequence[Tuple[TrajectoryRecord, RCPSPInstance]],
    max_resources: int = 4,
) -> Dict[str, torch.Tensor]:
    """
    DataLoader collate_fn.  Each item in batch is (TrajectoryRecord, RCPSPInstance).
    """
    if not batch:
        return {
            "candidate_feats": torch.zeros((0, candidate_feature_dim(max_resources))),
            "global_feats": torch.zeros((0, global_feature_dim(max_resources))),
            "lengths": torch.zeros(0, dtype=torch.long),
            "targets": torch.zeros(0, dtype=torch.long),
            "depths": torch.zeros(0, dtype=torch.long),
        }
    records, instances = zip(*batch)
    # Group by instance to share NodeContext construction
    from collections import defaultdict
    groups: Dict[int, List] = defaultdict(list)
    for rec, inst in zip(records, instances):
        groups[id(inst)].append((rec, inst))

    all_cand: List[List[float]] = []
    all_glob: List[List[float]] = []
    all_lengths: List[int] = []
    all_targets: List[int] = []
    all_depths: List[int] = []

    for group in groups.values():
        recs, insts = zip(*group)
        inst = insts[0]
        result = featurize_states(list(recs), inst, max_resources)
        if result["lengths"].numel() == 0:
            continue
        all_cand.extend(result["candidate_feats"].tolist())
        all_glob.extend(result["global_feats"].tolist())
        all_lengths.extend(result["lengths"].tolist())
        all_targets.extend(result["targets"].tolist())
        all_depths.extend(result["depths"].tolist())

    return {
        "candidate_feats": torch.tensor(all_cand, dtype=torch.float32),
        "global_feats": torch.tensor(all_glob, dtype=torch.float32),
        "lengths": torch.tensor(all_lengths, dtype=torch.long),
        "targets": torch.tensor(all_targets, dtype=torch.long),
        "depths": torch.tensor(all_depths, dtype=torch.long),
    }
