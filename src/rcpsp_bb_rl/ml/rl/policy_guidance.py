from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import torch

from rcpsp_bb_rl.bnb.precedence import build_predecessors
from rcpsp_bb_rl.bnb.scheduling import build_profile, earliest_feasible_start
from rcpsp_bb_rl.bnb.solver import BBNode
from rcpsp_bb_rl.data.parsing import RCPSPInstance
from rcpsp_bb_rl.ml.il.featurize import (
    InstanceStatics,
    NodeContext,
    candidate_features,
    global_features,
)
from rcpsp_bb_rl.ml.models.policy import BranchingTransformer


def _build_context_for_node(
    instance: RCPSPInstance,
    node: BBNode,
    predecessors: Dict,
    incumbent: Optional[int],
    statics: Optional[InstanceStatics] = None,
) -> NodeContext:
    """Build a NodeContext from a live B&B node."""
    ready_sorted = sorted(node.ready)
    # Reuse the earliest-start map the solver computed once for this node
    # (shared feasibility). Fall back to computing it here only if it wasn't
    # supplied (e.g. a caller that doesn't populate node.est_map).
    if node.est_map is not None:
        earliest_starts: Dict[int, Optional[int]] = {
            rid: node.est_map.get(rid) for rid in ready_sorted
        }
    else:
        horizon = sum(act.duration for act in instance.activities.values())
        profile = build_profile(
            instance.activities,
            instance.resource_caps,
            node.scheduled,
            horizon=horizon,
        )
        earliest_starts = {
            rid: earliest_feasible_start(
                instance,
                predecessors,
                node.scheduled,
                rid,
                incumbent=incumbent,
                profile=profile,
            )
            for rid in ready_sorted
        }
    return NodeContext(
        instance=instance,
        scheduled=node.scheduled,
        unscheduled=node.unscheduled,
        ready=node.ready,
        lower_bound=node.lower_bound,
        incumbent=incumbent,
        earliest_starts=earliest_starts,
        statics=statics,
    )


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
            compiled(dummy_cand, dummy_glob, dummy_mask)
    except Exception as exc:  # noqa: BLE001 - if warmup fails, don't trust the graph
        print(f"[policy] TorchScript warmup failed ({exc}); using eager model.")
        return model

    return compiled


def make_policy_order_fn(
    instance: RCPSPInstance,
    model: BranchingTransformer,
    max_resources: int = 4,
    device: torch.device | str = "cpu",
    predecessors: Optional[Dict] = None,
) -> callable:
    """
    Build a ready-order function driven by the branching policy.

    Returns a callable compatible with BnBSolver.solve(order_ready_fn=...).
    At each node it builds a NodeContext, featurises all candidates, runs
    the transformer, and returns the ready set sorted by descending logit.
    """
    device = torch.device(device)
    model = model.to(device)
    model.eval()
    model = _maybe_script_model(model, device)

    preds = predecessors if predecessors is not None else build_predecessors(instance)
    # Instance-invariant structures: built once, reused for every node. Sharing
    # `preds` with the statics avoids building predecessors twice.
    from rcpsp_bb_rl.ml.il.featurize import build_instance_statics
    statics = build_instance_statics(instance, predecessors=preds)

    def order_ready(node: BBNode, incumbent: Optional[int]) -> List[int]:
        if not node.ready:
            return []
        # Singleton shortcut: the ordering of one candidate is itself — skip
        # context construction and the model forward entirely. Exact: the full
        # path would featurise the single candidate and return it unchanged.
        if len(node.ready) == 1:
            return list(node.ready)

        ctx = _build_context_for_node(instance, node, preds, incumbent, statics=statics)
        ready_sorted = sorted(node.ready)

        # Build tensors via numpy: torch.tensor() on nested Python lists walks
        # every element to infer shape/dtype (slow); np.asarray + from_numpy is
        # the fast path to the identical tensor. Values are unchanged.
        glob_np = np.asarray(
            global_features(ctx, max_resources, depth=node.depth), dtype=np.float32
        )
        cand_np = np.asarray(
            [candidate_features(ctx, rid, max_resources) for rid in ready_sorted],
            dtype=np.float32,
        )
        mask_np = np.fromiter(
            (ctx.earliest_starts.get(rid) is not None for rid in ready_sorted),
            dtype=bool,
            count=len(ready_sorted),
        )

        glob = torch.from_numpy(glob_np).to(device)
        cand = torch.from_numpy(cand_np).to(device)
        mask = torch.from_numpy(mask_np).to(device)

        with torch.inference_mode():
            logits, _ = model(cand, glob, action_mask=mask)

        scored = sorted(
            zip(ready_sorted, logits.cpu().tolist()),
            key=lambda x: x[1],
            reverse=True,
        )
        return [aid for aid, _ in scored]

    return order_ready
