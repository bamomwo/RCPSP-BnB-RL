from __future__ import annotations

from typing import Dict, List, Optional

import torch

from rcpsp_bnb_rl.bnb.precedence import build_predecessors
from rcpsp_bnb_rl.bnb.lower_bounds import DEFAULT_LOWER_BOUND_ID
from rcpsp_bnb_rl.bnb.solver import BBNode
from rcpsp_bnb_rl.data.parsing import RCPSPInstance
from rcpsp_bnb_rl.ml.il.featurize import build_instance_statics
from rcpsp_bnb_rl.ml.action_order import policy_ranked_indices
from rcpsp_bnb_rl.ml.models.policy import BranchingTransformer
from rcpsp_bnb_rl.ml.observations import OBSERVATION_SCHEMA, build_actor_observation


def _maybe_script_model(
    model: BranchingTransformer,
    device: torch.device,
) -> torch.nn.Module:
    """
    Compile the policy with TorchScript so the per-node forward avoids Python
    module-dispatch overhead (the profile showed ~2.7s of ~4.1s forward time
    spent in `_call_impl`/`__getattr__`, not tensor math).

    We `script` rather than `trace`: scripting keeps the candidate count R
    dynamic, which trace would freeze to the warmup shape. `freeze` is attempted
    as a bonus (inlines eval-mode params as constants); if it fails the scripted
    module is still returned. Any failure at all falls back to the eager model —
    behaviour is identical either way, this only changes execution speed.
    """
    try:
        scripted = torch.jit.script(model)
    except Exception as exc:  # noqa: BLE001 - scripting is best-effort
        print(f"[policy] TorchScript scripting failed ({exc}); using eager model.")
        return model

    compiled: torch.nn.Module = scripted
    try:
        compiled = torch.jit.freeze(scripted)
    except Exception as exc:  # noqa: BLE001 - freeze is a bonus optimization
        print(f"[policy] TorchScript freeze failed ({exc}); using scripted (unfrozen) model.")
        compiled = scripted

    # Warmup: run one forward to trigger JIT compilation and validate the
    # compiled graph produces output before the solver depends on it. Uses the
    # model's own input dims so it matches the real call shape (R kept dynamic).
    try:
        cand_dim = model.cand_proj.in_features
        glob_dim = model.cls_proj.in_features
        dummy_cand = torch.zeros(2, cand_dim, dtype=torch.float32, device=device)
        dummy_glob = torch.zeros(glob_dim, dtype=torch.float32, device=device)
        dummy_mask = torch.ones(2, dtype=torch.bool, device=device)
        with torch.inference_mode():
            compiled(
                dummy_cand, dummy_glob, dummy_mask,
                resource_feats=torch.zeros(3, model.resource_feature_dim, device=device),
                candidate_resource_feats=torch.zeros(2, 3, model.interaction_feature_dim, device=device),
                resource_mask=torch.ones(3, dtype=torch.bool, device=device),
            )
    except Exception as exc:  # noqa: BLE001 - if warmup fails, don't trust the graph
        print(f"[policy] TorchScript warmup failed ({exc}); using eager model.")
        return model

    return compiled


def make_policy_order_fn(
    instance: RCPSPInstance,
    model: BranchingTransformer,
    device: torch.device | str = "cpu",
    predecessors: Optional[Dict] = None,
    lb_spec: object = DEFAULT_LOWER_BOUND_ID,
) -> callable:
    """
    Build a ready-order function driven by the branching policy.

    Returns a callable compatible with BnBSolver.solve(order_ready_fn=...).
    At each node it builds a NodeContext, featurises all candidates, runs
    the transformer, and returns the ready set sorted by descending logit.
    """
    device = torch.device(device)
    if model.observation_schema != OBSERVATION_SCHEMA:
        raise ValueError("Policy ordering requires a resource_set checkpoint")
    model = model.to(device)
    model.eval()
    model = _maybe_script_model(model, device)

    preds = predecessors if predecessors is not None else build_predecessors(instance)
    # Instance-invariant structures: built once, reused for every node. Sharing
    # `preds` with the statics avoids building predecessors twice.
    statics = build_instance_statics(instance, predecessors=preds)

    def order_ready(node: BBNode, incumbent: Optional[int]) -> List[int]:
        if not node.ready:
            return []
        # Singleton shortcut: the ordering of one candidate is itself — skip
        # context construction and the model forward entirely. Exact: the full
        # path would featurise the single candidate and return it unchanged.
        if len(node.ready) == 1:
            return list(node.ready)

        obs = build_actor_observation(
            instance, node.scheduled, node.unscheduled, node.ready, node.depth,
            statics=statics, profile=node.resource_profile, lb_spec=lb_spec,
        )
        ready_ids = obs.pop("ready_ids").tolist()
        with torch.inference_mode():
            logits = model(**{key: value.to(device) for key, value in obs.items()})
        return [ready_ids[index] for index in policy_ranked_indices(logits.cpu().tolist())]

    return order_ready
