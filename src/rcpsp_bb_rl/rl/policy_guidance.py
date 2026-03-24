from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import torch

from rcpsp_bb_rl.bnb.core import BBNode, ScheduleEntry, build_predecessors, current_makespan, earliest_feasible_start
from rcpsp_bb_rl.data.featurize import candidate_features, global_features
from rcpsp_bb_rl.data.parsing import RCPSPInstance
from rcpsp_bb_rl.data.trajectory_dataset import TrajectoryRecord
from rcpsp_bb_rl.models import PolicyMLP


def _resource_usage_now(instance: RCPSPInstance, scheduled: Dict[int, ScheduleEntry], now: int) -> List[int]:
    """Compute resource usage at a specific time point."""
    usage = [0 for _ in instance.resource_caps]
    for act_id, entry in scheduled.items():
        for r, _cap in enumerate(instance.resource_caps):
            if entry.start <= now < entry.finish:
                usage[r] += instance.activities[act_id].resources[r]
    return usage


def _remaining_energy(instance: RCPSPInstance, unscheduled: Iterable[int]) -> List[float]:
    """Sum duration * resource across remaining activities per resource."""
    energy: List[float] = []
    for r in range(len(instance.resource_caps)):
        total = 0.0
        for aid in unscheduled:
            act = instance.activities[aid]
            total += float(act.duration * act.resources[r])
        energy.append(total)
    return energy


def _static_activity_info(
    instance: RCPSPInstance, predecessors: Dict[int, Sequence[int]]
) -> Dict[str, Dict[str, object]]:
    """Mirror the 'activities' structure used when logging teacher trajectories."""
    info: Dict[str, Dict[str, object]] = {}
    for aid, act in instance.activities.items():
        info[str(aid)] = {
            "duration": act.duration,
            "resources": act.resources,
            "successors": act.successors,
            "num_successors": len(act.successors),
            "num_predecessors": len(predecessors.get(aid, ())),
        }
    return info


def _build_record_for_node(
    instance: RCPSPInstance,
    node: BBNode,
    predecessors: Dict[int, Sequence[int]],
    incumbent: Optional[int],
) -> TrajectoryRecord:
    """Create a TrajectoryRecord-like object for the current search node."""
    ready_sorted = sorted(node.ready)
    unscheduled_sorted = sorted(node.unscheduled)
    ms = current_makespan(node.scheduled)
    res_used = _resource_usage_now(instance, node.scheduled, ms)
    rem_durs = [instance.activities[a].duration for a in node.unscheduled]
    rem_energy = _remaining_energy(instance, node.unscheduled)

    earliest_map: Dict[int, Optional[int]] = {}
    for rid in ready_sorted:
        earliest_map[rid] = earliest_feasible_start(
            instance,
            predecessors,
            node.scheduled,
            rid,
            incumbent=incumbent,
        )

    raw = {
        "instance": None,
        "num_activities": instance.num_activities,
        "num_resources": instance.num_resources,
        "resource_caps": instance.resource_caps,
        "depth": node.depth,
        "ready": ready_sorted,
        "unscheduled": unscheduled_sorted,
        "scheduled": {
            str(k): {"start": v.start, "finish": v.finish, "duration": v.duration}
            for k, v in node.scheduled.items()
        },
        "activities": _static_activity_info(instance, predecessors),
        "earliest_start": {str(k): v for k, v in earliest_map.items()},
        "lower_bound": node.lower_bound,
        "makespan_so_far": ms,
        "resource_used_now": res_used,
        "resource_available_now": [cap - use for cap, use in zip(instance.resource_caps, res_used)],
        "remaining_durations": rem_durs,
        "remaining_energy_per_resource": rem_energy,
        "action": {"task": -1, "start": -1},
    }

    return TrajectoryRecord(raw=raw, source=Path("bnb_policy"))


def make_policy_order_fn(
    instance: RCPSPInstance,
    model: PolicyMLP,
    max_resources: int = 4,
    device: torch.device | str = "cpu",
    predecessors: Optional[Dict[int, Sequence[int]]] = None,
) -> callable:
    """
    Build a function that orders a node's ready set by policy scores.

    Returns a callable suitable for BnBSolver.solve(order_ready_fn=...).
    """
    device = torch.device(device)
    model = model.to(device)
    model.eval()

    preds = predecessors if predecessors is not None else build_predecessors(instance)

    def order_ready(node: BBNode, incumbent: Optional[int]) -> List[int]:
        if not node.ready:
            return []

        record = _build_record_for_node(instance, node, preds, incumbent)
        glob_feats = global_features(record, max_resources=max_resources)
        glob_tensor = torch.tensor(glob_feats, dtype=torch.float32, device=device).unsqueeze(0)

        ready_sorted = record.ready
        cand_feats = torch.tensor(
            [candidate_features(record, rid, max_resources=max_resources) for rid in ready_sorted],
            dtype=torch.float32,
            device=device,
        )
        glob_batch = glob_tensor.repeat(len(ready_sorted), 1)

        with torch.no_grad():
            logits = model(cand_feats, glob_batch).cpu().tolist()

        scored = list(zip(ready_sorted, logits))
        scored.sort(key=lambda x: x[1], reverse=True)
        return [aid for aid, _ in scored]

    return order_ready
