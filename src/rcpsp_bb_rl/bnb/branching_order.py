from __future__ import annotations

import random
from typing import TYPE_CHECKING, Callable, Dict, List, Mapping, Optional, Protocol, Set

from rcpsp_bb_rl.bnb.lower_bounds import DEFAULT_LOWER_BOUND_ID, lower_bound
from rcpsp_bb_rl.bnb.precedence import build_predecessors, build_successors, topological_order
from rcpsp_bb_rl.bnb.scheduling import build_profile, earliest_feasible_start
from rcpsp_bb_rl.bnb.scheduling import entry_start

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


def make_random_order_fn(*, seed: int = 0) -> ReadyOrderFn:
    """Randomly permute every ready activity at each branching node.

    All ready activities remain branches in the serial branching scheme. The
    permutation only decides their depth-first exploration order; it does not
    choose one activity and discard the others. A private seeded generator
    makes one run reproducible without modifying global random state.
    """
    rng = random.Random(seed)

    def order_ready(node: NodeLike, incumbent: Optional[int]) -> List[int]:
        _ = incumbent
        ready_sorted = sorted(node.ready)
        # The solver constructs this map before it calls the ordering
        # callback. Keep candidates with an available start ahead of those
        # that cannot be scheduled below the current incumbent, as every
        # other ordering callback does.
        est_map = getattr(node, "est_map", None)
        if est_map is None:
            rng.shuffle(ready_sorted)
            return ready_sorted

        feasible = [act_id for act_id in ready_sorted if est_map.get(act_id) is not None]
        infeasible = [act_id for act_id in ready_sorted if est_map.get(act_id) is None]
        rng.shuffle(feasible)
        rng.shuffle(infeasible)
        return feasible + infeasible

    return order_ready


def _latest_starts_for_horizon(
    *,
    instance: RCPSPInstance,
    scheduled: Mapping[int, object],
    successors: Mapping[int, List[int]],
    topo_order: List[int],
    horizon: int,
) -> Dict[int, int]:
    """Compute precedence-only latest-start limits for a fixed horizon."""
    latest: Dict[int, int] = {}
    for act_id in reversed(topo_order):
        duration = int(instance.activities[act_id].duration)
        if act_id in scheduled:
            latest_start = entry_start(scheduled[act_id])
        else:
            latest_start = int(horizon) - duration
        for successor in successors.get(act_id, []):
            latest_start = min(latest_start, latest[successor] - duration)
        latest[act_id] = int(latest_start)
    return latest


def make_mtw_order_fn(
    *,
    instance: RCPSPInstance,
) -> ReadyOrderFn:
    """Order ready activities by minimal precedence time window.

    The current serial branching scheme fixes each selected activity at its
    earliest resource-feasible start. MTW therefore ranks activities by
    ``LS - ES``: ``ES`` comes from the node's shared feasibility map and ``LS``
    is the latest precedence-feasible start under the incumbent makespan. Before
    the first incumbent, the sum of all activity durations is used as a safe,
    deliberately loose horizon. Infeasible ready activities remain at the end.
    """
    successors = build_successors(instance)
    predecessors = build_predecessors(instance)
    topo_order = topological_order(instance)
    fallback_horizon = sum(int(act.duration) for act in instance.activities.values())

    def order_ready(node: NodeLike, incumbent: Optional[int]) -> List[int]:
        ready_sorted = sorted(node.ready)
        if not ready_sorted:
            return []

        scheduled = getattr(node, "scheduled", None)
        if scheduled is None:
            raise AttributeError("mtw ordering requires node.scheduled fields.")

        horizon = int(incumbent) if incumbent is not None else fallback_horizon
        latest_starts = _latest_starts_for_horizon(
            instance=instance,
            scheduled=scheduled,
            successors=successors,
            topo_order=topo_order,
            horizon=horizon,
        )
        est_map = getattr(node, "est_map", None)
        if est_map is None:
            profile = build_profile(
                instance.activities,
                instance.resource_caps,
                scheduled,
                horizon=horizon,
            )
            est_map = {
                act_id: earliest_feasible_start(
                    instance, predecessors, scheduled, act_id, incumbent, profile=profile
                )
                for act_id in ready_sorted
            }

        scored: List[tuple[int, int]] = []
        infeasible: List[int] = []
        for act_id in ready_sorted:
            earliest = est_map.get(act_id)
            if earliest is None:
                infeasible.append(act_id)
                continue
            window = int(latest_starts[act_id]) - int(earliest)
            scored.append((window, act_id))

        scored.sort(key=lambda item: (item[0], item[1]))
        return [act_id for _, act_id in scored] + infeasible

    return order_ready


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
    resource_encoder: str = "legacy_flat",
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
        resource_encoder=resource_encoder,
        device=device,
        predecessors=predecessors,
    )


def make_order_fn(kind: str, **kwargs) -> ReadyOrderFn:
    """
    Factory for ready-order functions.

    Supported:
    - activity_id
    - lower_bound
    - mtw
    - random
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

    if normalized == "mtw":
        if kwargs.get("instance") is None:
            raise ValueError("Missing required argument for mtw order: instance")
        return make_mtw_order_fn(instance=kwargs["instance"])

    if normalized == "random":
        return make_random_order_fn(seed=int(kwargs.get("seed", 0)))

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
            resource_encoder=str(kwargs.get("resource_encoder", "legacy_flat")),
            device=kwargs.get("device", "cpu"),
            predecessors=kwargs.get("predecessors"),
        )

    raise ValueError(
        f"Unknown branching order kind: {kind}. Supported kinds: activity_id, lower_bound, mtw, random, policy"
    )
