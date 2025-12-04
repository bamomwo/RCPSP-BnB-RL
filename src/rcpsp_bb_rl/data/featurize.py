from __future__ import annotations

from statistics import mean
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch

from rcpsp_bb_rl.data.trajectory_dataset import CandidateExample, TrajectoryRecord


def _pad(values: Sequence[float], target_len: int) -> List[float]:
    """Pad or truncate a 1D sequence to target_len."""
    arr = list(values[:target_len])
    if len(arr) < target_len:
        arr.extend([0.0] * (target_len - len(arr)))
    return arr


def _summary_stats(vals: Iterable[int | float]) -> Tuple[float, float, float, float]:
    """
    Return (sum, mean, min, max) for an iterable. If empty, return zeros.
    """
    vals = list(vals)
    if not vals:
        return 0.0, 0.0, 0.0, 0.0
    return (
        float(sum(vals)),
        float(mean(vals)),
        float(min(vals)),
        float(max(vals)),
    )


def global_features(record: TrajectoryRecord, max_resources: int) -> List[float]:
    """Build global features shared across all candidates in a state."""
    caps = _pad(record.raw["resource_caps"], max_resources)

    rem_durs = record.raw.get("remaining_durations", [])
    rem_energy = record.raw.get("remaining_energy_per_resource", [])
    rem_sum, rem_mean, rem_min, rem_max = _summary_stats(rem_durs)
    energy_padded = _pad(rem_energy, max_resources)

    feats: List[float] = []
    feats.append(float(record.raw["num_activities"]))
    feats.append(float(record.raw["num_resources"]))
    feats.extend(caps)
    feats.append(float(record.raw.get("makespan_so_far", 0)))
    feats.append(float(record.raw.get("lower_bound", 0)))
    feats.append(float(len(record.ready)))
    feats.append(float(len(record.raw.get("unscheduled", []))))
    feats.extend([rem_sum, rem_mean, rem_min, rem_max])
    feats.extend(energy_padded)
    return feats


def candidate_features(
    record: TrajectoryRecord,
    candidate: int,
    max_resources: int,
) -> List[float]:
    """Per-candidate features for ranking tasks in the ready set."""
    act = record.raw["activities"][str(candidate)]
    duration = float(act["duration"])
    resources = _pad(act["resources"], max_resources)
    num_succ = float(act.get("num_successors", len(act.get("successors", []))))
    num_pred = float(act.get("num_predecessors", 0))

    earliest_start = record.earliest_start.get(candidate)
    # None -> -1 sentinel; helps the model learn infeasible/uncomputed cases.
    est = -1.0 if earliest_start is None else float(earliest_start)

    lb = float(record.raw.get("lower_bound", 0))
    slack_to_lb = lb - (est + duration) if earliest_start is not None else 0.0

    feats: List[float] = []
    feats.append(duration)
    feats.extend(resources)
    feats.append(num_succ)
    feats.append(num_pred)
    feats.append(est)
    feats.append(slack_to_lb)
    return feats


def featurize_candidates(
    examples: Sequence[CandidateExample],
    max_resources: int,
) -> Dict[str, torch.Tensor]:
    """
    Convert CandidateExample list into tensors suitable for an MLP.

    Returns:
        {
            "candidate_feats": [N, Fc],
            "global_feats": [N, Fg],
            "labels": [N],
            "candidate_ids": [N],
            "depths": [N],
        }
    """
    cand_feats: List[List[float]] = []
    glob_feats: List[List[float]] = []
    labels: List[int] = []
    cand_ids: List[int] = []
    depths: List[int] = []

    for ex in examples:
        cand_feats.append(candidate_features(ex.record, ex.candidate, max_resources))
        glob_feats.append(global_features(ex.record, max_resources))
        labels.append(int(ex.label))
        cand_ids.append(int(ex.candidate))
        depths.append(int(ex.record.raw.get("depth", 0)))

    return {
        "candidate_feats": torch.tensor(cand_feats, dtype=torch.float32),
        "global_feats": torch.tensor(glob_feats, dtype=torch.float32),
        "labels": torch.tensor(labels, dtype=torch.long),
        "candidate_ids": torch.tensor(cand_ids, dtype=torch.long),
        "depths": torch.tensor(depths, dtype=torch.long),
    }


def collate_candidate_batch(
    batch: Sequence[CandidateExample],
    max_resources: int = 4,
) -> Dict[str, torch.Tensor]:
    """
    DataLoader collate_fn for flattened CandidateExample batches.

    Example:
        loader = DataLoader(dataset, batch_size=512, shuffle=True,
                            collate_fn=lambda b: collate_candidate_batch(b, max_resources=4))
    """
    return featurize_candidates(batch, max_resources=max_resources)
