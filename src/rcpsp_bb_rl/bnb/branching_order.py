from __future__ import annotations

from typing import TYPE_CHECKING, Callable, List, Mapping, Optional, Protocol, Set

from rcpsp_bb_rl.bnb.lower_bounds import DEFAULT_LOWER_BOUND_ID, lower_bound
from rcpsp_bb_rl.bnb.precedence import build_predecessors
from rcpsp_bb_rl.bnb.scheduling import build_profile, earliest_feasible_start

if TYPE_CHECKING:
    from rcpsp_bb_rl.data.parsing import RCPSPInstance
    from rcpsp_bb_rl.ml.models import BranchingTransformer


class NodeLike(Protocol):
    ready: Set[int]


ReadyOrderFn = Callable[[NodeLike, Optional[int]], List[int]]


def order_by_activity_id(node: NodeLike, incumbent: Optional[int]) -> List[int]:
    """
    Deterministic default: order the ready set by ascending activity ID.
    """
    _ = incumbent
    return sorted(node.ready)


def make_lower_bound_order_fn(
    *,
    instance: RCPSPInstance,
    predecessors: Optional[Mapping[int, Set[int]]] = None,
    lb_id: object = DEFAULT_LOWER_BOUND_ID,
) -> ReadyOrderFn:
    """
    Order ready activities by child-node lower bound (ascending, then activity ID).
    """
    preds = dict(predecessors) if predecessors is not None else build_predecessors(instance)

    def order_ready(node: NodeLike, incumbent: Optional[int]) -> List[int]:
        ready_sorted = sorted(node.ready)
        if not ready_sorted:
            return []

        scheduled = getattr(node, "scheduled", None)
        unscheduled = getattr(node, "unscheduled", None)
        if scheduled is None or unscheduled is None:
            raise AttributeError(
                "lower_bound ordering requires node.scheduled and node.unscheduled fields."
            )

        horizon_hint = sum(act.duration for act in instance.activities.values())
        node_horizon = incumbent if incumbent is not None else horizon_hint
        node_profile = build_profile(
            instance.activities,
            instance.resource_caps,
            scheduled,
            horizon=node_horizon,
        )

        scored: List[tuple[int, int]] = []
        infeasible: List[int] = []
        for act_id in ready_sorted:
            est_start = earliest_feasible_start(
                instance=instance,
                predecessors=preds,
                scheduled=scheduled,
                act_id=act_id,
                incumbent=incumbent,
                profile=node_profile,
            )
            if est_start is None:
                infeasible.append(act_id)
                continue

            duration = int(instance.activities[act_id].duration)
            finish = est_start + duration

            child_scheduled = dict(scheduled)
            child_scheduled[act_id] = {
                "start": est_start,
                "finish": finish,
                "duration": duration,
            }
            child_unscheduled = set(unscheduled)
            child_unscheduled.discard(act_id)
            child_lb = lower_bound(
                instance=instance,
                unscheduled=child_unscheduled,
                scheduled=child_scheduled,
                lb_id=lb_id,
            )
            scored.append((int(child_lb), act_id))

        scored.sort(key=lambda item: (item[0], item[1]))
        return [act_id for _, act_id in scored] + infeasible

    return order_ready


def make_policy_order_fn(
    *,
    instance: RCPSPInstance,
    model: BranchingTransformer,
    max_resources: int = 4,
    device: object = "cpu",
    predecessors=None,
) -> ReadyOrderFn:
    """
    Thin wrapper around rl.policy_guidance.make_policy_order_fn.
    Keeps policy logic in the RL module while exposing a unified B&B API.
    """
    from rcpsp_bb_rl.ml.rl.policy_guidance import make_policy_order_fn as _make_policy_order_fn

    return _make_policy_order_fn(
        instance=instance,
        model=model,
        max_resources=max_resources,
        device=device,
        predecessors=predecessors,
    )


def make_order_fn(kind: str, **kwargs) -> ReadyOrderFn:
    """
    Factory for ready-order functions.

    Supported:
    - activity_id
    - lower_bound
    - policy
    """
    normalized = str(kind).strip().lower()

    if normalized == "activity_id":
        return order_by_activity_id

    if normalized == "lower_bound":
        if kwargs.get("instance") is None:
            raise ValueError("Missing required argument for lower_bound order: instance")
        return make_lower_bound_order_fn(
            instance=kwargs["instance"],
            predecessors=kwargs.get("predecessors"),
            lb_id=kwargs.get("lb_id", DEFAULT_LOWER_BOUND_ID),
        )

    if normalized == "policy":
        required = ("instance", "model")
        missing = [name for name in required if kwargs.get(name) is None]
        if missing:
            raise ValueError(
                f"Missing required arguments for policy order: {', '.join(missing)}"
            )
        return make_policy_order_fn(
            instance=kwargs["instance"],
            model=kwargs["model"],
            max_resources=int(kwargs.get("max_resources", 4)),
            device=kwargs.get("device", "cpu"),
            predecessors=kwargs.get("predecessors"),
        )

    raise ValueError(
        f"Unknown branching order kind: {kind}. Supported kinds: activity_id, lower_bound, policy"
    )
