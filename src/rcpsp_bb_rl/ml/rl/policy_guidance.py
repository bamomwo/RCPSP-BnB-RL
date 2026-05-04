from __future__ import annotations

from typing import Dict, List, Optional

import torch

from rcpsp_bb_rl.bnb.precedence import build_predecessors
from rcpsp_bb_rl.bnb.scheduling import build_profile, earliest_feasible_start
from rcpsp_bb_rl.bnb.solver import BBNode
from rcpsp_bb_rl.data.parsing import RCPSPInstance
from rcpsp_bb_rl.ml.il.featurize import (
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
) -> NodeContext:
    """Build a NodeContext from a live B&B node."""
    ready_sorted = sorted(node.ready)
    horizon = sum(act.duration for act in instance.activities.values())
    profile = build_profile(
        instance.activities,
        instance.resource_caps,
        node.scheduled,
        horizon=horizon,
    )
    earliest_starts: Dict[int, Optional[int]] = {
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
    )


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

    preds = predecessors if predecessors is not None else build_predecessors(instance)

    def order_ready(node: BBNode, incumbent: Optional[int]) -> List[int]:
        if not node.ready:
            return []

        ctx = _build_context_for_node(instance, node, preds, incumbent)
        ready_sorted = sorted(node.ready)

        glob = torch.tensor(
            global_features(ctx, max_resources, depth=node.depth),
            dtype=torch.float32,
            device=device,
        )
        cand = torch.tensor(
            [candidate_features(ctx, rid, max_resources) for rid in ready_sorted],
            dtype=torch.float32,
            device=device,
        )
        mask = torch.tensor(
            [ctx.earliest_starts.get(rid) is not None for rid in ready_sorted],
            dtype=torch.bool,
            device=device,
        )

        with torch.no_grad():
            logits, _ = model(cand, glob, action_mask=mask)

        scored = sorted(
            zip(ready_sorted, logits.cpu().tolist()),
            key=lambda x: x[1],
            reverse=True,
        )
        return [aid for aid, _ in scored]

    return order_ready
