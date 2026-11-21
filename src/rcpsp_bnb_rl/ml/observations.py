"""Shared actor inputs independent of search incumbents and pruning history."""
from __future__ import annotations

from typing import Mapping, Optional, Sequence

import numpy as np
import torch

from rcpsp_bnb_rl.bnb.lower_bounds import DEFAULT_LOWER_BOUND_ID, lower_bound
from rcpsp_bnb_rl.bnb.scheduling import ResourceProfile, build_profile, earliest_feasible_start
from rcpsp_bnb_rl.data.parsing import RCPSPInstance
from rcpsp_bnb_rl.ml.il.featurize import (
    InstanceStatics, NodeContext, build_instance_statics,
    candidate_features, candidate_set_features, candidate_resource_features,
    global_features, global_set_features, resource_set_features,
    candidate_feature_dim, set_candidate_feature_dim, set_interaction_feature_dim,
)


OBSERVATION_SCHEMA = "incumbent_free_v1"
LEGACY_OBSERVATION_SCHEMA = "legacy"


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
    resource_encoder: str = "set",
    max_resources: int = 4,
) -> dict[str, torch.Tensor]:
    """Build intrinsic scheduling features, retaining the BC feature widths.

    Latest-start windows use the sum-of-durations horizon, matching BC's
    pre-incumbent inputs. The reserved incumbent-gap input is always zero.
    Solver-provided bounds, earliest starts, and feasibility masks are not used.
    """
    if resource_encoder not in {"set", "legacy_flat"}:
        raise ValueError("resource_encoder must be 'set' or 'legacy_flat'")
    statics = statics if statics is not None else build_instance_statics(instance)
    ready_ids = sorted(ready)
    if profile is None:
        profile = build_profile(
            instance.activities, instance.resource_caps, scheduled,
            horizon=int(statics.sum_durations),
        )
    starts = {
        aid: earliest_feasible_start(
            instance, statics.predecessors, scheduled, aid,
            incumbent=None, profile=profile,
        )
        for aid in ready_ids
    }
    ctx = NodeContext(
        instance, scheduled, unscheduled, ready,
        lower_bound(instance, unscheduled, scheduled, lb_id=lb_spec, incumbent=None),
        None, starts, statics,
    )
    use_set = resource_encoder == "set"
    glob = global_set_features(ctx, depth) if use_set else global_features(ctx, max_resources, depth)
    candidates = [
        candidate_set_features(ctx, aid) if use_set else candidate_features(ctx, aid, max_resources)
        for aid in ready_ids
    ]
    width = set_candidate_feature_dim() if use_set else candidate_feature_dim(max_resources)
    obs = {
        "global_feats": torch.from_numpy(np.asarray(glob, dtype=np.float32)),
        "candidate_feats": torch.from_numpy(np.asarray(candidates, dtype=np.float32).reshape(-1, width)),
        "action_mask": torch.tensor([starts[aid] is not None for aid in ready_ids], dtype=torch.bool),
        "ready_ids": torch.tensor(ready_ids, dtype=torch.long),
    }
    if use_set:
        resources = resource_set_features(ctx)
        obs.update(
            resource_feats=torch.from_numpy(np.asarray(resources, dtype=np.float32)),
            candidate_resource_feats=torch.from_numpy(np.asarray(
                [candidate_resource_features(ctx, aid) for aid in ready_ids], dtype=np.float32,
            ).reshape(len(ready_ids), len(resources), set_interaction_feature_dim())),
            resource_mask=torch.ones(len(resources), dtype=torch.bool),
        )
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
    use_set = "resource_feats" in first
    result = {
        "candidate_feats": torch.zeros(batch_size, max_ready, cand_dim),
        "global_feats": torch.stack([obs["global_feats"] for obs in observations]),
        "action_mask": torch.zeros(batch_size, max_ready, dtype=torch.bool),
        "pad_mask": torch.zeros(batch_size, max_ready, dtype=torch.bool),
    }
    if use_set:
        max_res = max(len(obs["resource_feats"]) for obs in observations)
        result.update(
            resource_feats=torch.zeros(batch_size, max_res, first["resource_feats"].shape[-1]),
            candidate_resource_feats=torch.zeros(
                batch_size, max_ready, max_res, first["candidate_resource_feats"].shape[-1],
            ),
            resource_mask=torch.zeros(batch_size, max_res, dtype=torch.bool),
        )
    for row, obs in enumerate(observations):
        if ("resource_feats" in obs) != use_set:
            raise ValueError("Cannot mix resource-set and legacy observations")
        count = len(obs["action_mask"])
        result["candidate_feats"][row, :count] = obs["candidate_feats"]
        result["action_mask"][row, :count] = obs["action_mask"]
        result["pad_mask"][row, :count] = True
        if use_set:
            n_res = len(obs["resource_feats"])
            result["resource_feats"][row, :n_res] = obs["resource_feats"]
            result["candidate_resource_feats"][row, :count, :n_res] = obs["candidate_resource_feats"]
            result["resource_mask"][row, :n_res] = obs["resource_mask"]
    return {name: tensor.to(device) for name, tensor in result.items()}
