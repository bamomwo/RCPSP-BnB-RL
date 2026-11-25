"""Shared actor inputs independent of search incumbents and pruning history."""
from __future__ import annotations

from typing import Mapping, Optional, Sequence

import numpy as np
import torch

from rcpsp_bnb_rl.bnb.lower_bounds import DEFAULT_LOWER_BOUND_ID, lower_bound, normalize_lower_bound_spec
from rcpsp_bnb_rl.bnb.scheduling import ResourceProfile, build_profile, earliest_feasible_start, entry_finish
from rcpsp_bnb_rl.data.parsing import RCPSPInstance
from rcpsp_bnb_rl.ml.il.featurize import (
    FEATURE_SCHEMA, InstanceStatics, NodeContext, build_instance_statics,
    candidate_set_features, candidate_resource_features,
    global_set_features, resource_set_features,
    set_candidate_feature_dim, set_interaction_feature_dim,
)


OBSERVATION_SCHEMA = FEATURE_SCHEMA


def build_actor_observation(
    instance: RCPSPInstance,
    scheduled: Mapping[int, object],
    unscheduled: set[int],
    ready: set[int],
    depth: int,
    *,
    statics: Optional[InstanceStatics] = None,
    profile: Optional[ResourceProfile] = None,
    lb_spec: object = DEFAULT_LOWER_BOUND_ID,
) -> dict[str, torch.Tensor]:
    """Build ``resource_set`` inputs without search incumbents or cached masks."""
    statics = statics if statics is not None else build_instance_statics(instance)
    ready_ids = sorted(ready)
    max_finish = max((entry_finish(entry) for entry in scheduled.values()), default=0)
    horizon = max(int(statics.sum_durations), max_finish + max(
        (activity.duration for activity in instance.activities.values()), default=0,
    ))
    if profile is None or profile.horizon < max_finish:
        profile = build_profile(
            instance.activities, instance.resource_caps, scheduled,
            horizon=horizon,
        )
    elif profile.horizon < horizon:
        # DFS profiles may stop at the incumbent. Zero extension preserves their
        # occupancy while avoiding one profile rebuild per unrestricted EST.
        for row in profile.usage:
            row.extend([0] * (horizon - profile.horizon))
        profile.horizon = horizon
    starts = {
        aid: earliest_feasible_start(
            instance, statics.predecessors, scheduled, aid,
            incumbent=None, profile=profile,
        )
        for aid in ready_ids
    }
    # Keep observations cheap and incumbent-free: ``lb_cc`` is a solver-only
    # pruning bound and must never be evaluated while constructing Actor input.
    requested_bounds = [name for name in normalize_lower_bound_spec(lb_spec) if name != "lb_cc"]
    # Preserve the caller's composite semantics for inexpensive bounds while
    # excluding the solver-only critical-chain/resource bound.
    observation_lb = requested_bounds if len(requested_bounds) > 1 else (
        requested_bounds[0] if requested_bounds else "lb_cp"
    )
    ctx = NodeContext(
        instance, scheduled, unscheduled, ready,
        lower_bound(instance, unscheduled, scheduled, lb_id=observation_lb, incumbent=None),
        starts, statics, profile=profile,
    )
    glob = global_set_features(ctx)
    candidates = [candidate_set_features(ctx, aid) for aid in ready_ids]
    width = set_candidate_feature_dim()
    resources = resource_set_features(ctx)
    obs = {
        "global_feats": torch.from_numpy(np.asarray(glob, dtype=np.float32)),
        "candidate_feats": torch.from_numpy(np.asarray(candidates, dtype=np.float32).reshape(-1, width)),
        "action_mask": torch.tensor([starts[aid] is not None for aid in ready_ids], dtype=torch.bool),
        "ready_ids": torch.tensor(ready_ids, dtype=torch.long),
        "resource_feats": torch.from_numpy(np.asarray(resources, dtype=np.float32)),
        "candidate_resource_feats": torch.from_numpy(np.asarray(
            [candidate_resource_features(ctx, aid) for aid in ready_ids], dtype=np.float32,
        ).reshape(len(ready_ids), len(resources), set_interaction_feature_dim())),
        "resource_mask": torch.ones(len(resources), dtype=torch.bool),
    }
    return obs


def batch_observations(
    observations: Sequence[dict[str, torch.Tensor]],
    device: torch.device | str,
) -> dict[str, torch.Tensor]:
    """Pad independent observations for one actor ``forward_batch`` call."""
    if not observations:
        raise ValueError("Cannot batch an empty observation list")
    batch_size = len(observations)
    max_ready = max(len(obs["action_mask"]) for obs in observations)
    first = observations[0]
    cand_dim = first["candidate_feats"].shape[-1]
    result = {
        "candidate_feats": torch.zeros(batch_size, max_ready, cand_dim),
        "global_feats": torch.stack([obs["global_feats"] for obs in observations]),
        "action_mask": torch.zeros(batch_size, max_ready, dtype=torch.bool),
        "pad_mask": torch.zeros(batch_size, max_ready, dtype=torch.bool),
    }
    max_res = max(len(obs["resource_feats"]) for obs in observations)
    result.update(
        resource_feats=torch.zeros(batch_size, max_res, first["resource_feats"].shape[-1]),
        candidate_resource_feats=torch.zeros(
            batch_size, max_ready, max_res, first["candidate_resource_feats"].shape[-1],
        ),
        resource_mask=torch.zeros(batch_size, max_res, dtype=torch.bool),
    )
    for row, obs in enumerate(observations):
        count = len(obs["action_mask"])
        result["candidate_feats"][row, :count] = obs["candidate_feats"]
        result["action_mask"][row, :count] = obs["action_mask"]
        result["pad_mask"][row, :count] = True
        n_res = len(obs["resource_feats"])
        result["resource_feats"][row, :n_res] = obs["resource_feats"]
        result["candidate_resource_feats"][row, :count, :n_res] = obs["candidate_resource_feats"]
        result["resource_mask"][row, :n_res] = obs["resource_mask"]
    return {name: tensor.to(device) for name, tensor in result.items()}
