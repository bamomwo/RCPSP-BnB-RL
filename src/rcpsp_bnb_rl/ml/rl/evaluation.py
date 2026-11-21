"""Controlled policy comparisons inside the unchanged exact DFS solver."""
from __future__ import annotations

from pathlib import Path
import time
from typing import Iterable, Mapping, Optional

import torch

from rcpsp_bnb_rl.bnb.lower_bounds import DEFAULT_LOWER_BOUND_ID, lower_bound
from rcpsp_bnb_rl.bnb.solver import BnBSolver
from rcpsp_bnb_rl.data.parsing import load_instance
from rcpsp_bnb_rl.ml.models import BranchingTransformer
from rcpsp_bnb_rl.ml.rl.policy_guidance import make_policy_order_fn


def _mean(values: Iterable[Optional[float]]) -> Optional[float]:
    available = [float(value) for value in values if value is not None]
    return sum(available) / len(available) if available else None


def _gap(makespan: Optional[int], reference: Optional[int]) -> Optional[float]:
    """Relative gaps are fractions, not percentages; zero references are undefined."""
    if makespan is None or reference is None or reference <= 0:
        return None
    return (makespan - reference) / reference


def _references(values: Optional[Mapping[str, int]]) -> dict[str, int]:
    references = {}
    for name, value in (values or {}).items():
        if isinstance(value, bool) or int(value) != value or value < 0:
            raise ValueError("Makespan references must be non-negative integers")
        references[Path(str(name)).stem.lower()] = int(value)
    return references


def evaluate_actor(
    actor: BranchingTransformer,
    paths: Iterable[Path | str],
    *,
    device: torch.device | str = "cpu",
    time_limit_s: Optional[float] = 60.0,
    max_nodes: Optional[int] = None,
    lb_spec: object = DEFAULT_LOWER_BOUND_ID,
    observation_schema: Optional[str] = None,
    optimal_makespans: Optional[Mapping[str, int]] = None,
    target_makespans: Optional[Mapping[str, int]] = None,
) -> dict:
    """Evaluate all instances, separating primal target matches from proof.

    A missing explicit target falls back to the known optimum only when no
    target mapping was supplied. Gap means use available schedules and report
    their denominator; attainment fractions include failures to find a schedule.
    Solver debug mode supplies incumbent events without changing search order.
    """
    if max_nodes is not None and (isinstance(max_nodes, bool) or max_nodes < 0):
        raise ValueError("max_nodes must be non-negative or None")
    if time_limit_s is not None and time_limit_s < 0:
        raise ValueError("time_limit_s must be non-negative or None")
    schema = observation_schema or getattr(actor, "observation_schema", "legacy")
    optima = _references(optimal_makespans)
    targets = _references(target_makespans) if target_makespans is not None else optima
    rows = []
    actor.eval()

    for path_value in paths:
        path = Path(path_value)
        key = path.stem.lower()
        instance = load_instance(path)
        root_lb = lower_bound(instance, set(instance.activities), {}, lb_id=lb_spec)
        optimum = optima.get(key)
        target = targets.get(key)
        solver = BnBSolver(instance)
        order_fn = make_policy_order_fn(
            instance=instance, model=actor, device=device,
            resource_encoder="set" if actor.resource_set_enabled else "legacy_flat",
            predecessors=solver.predecessors, observation_schema=schema,
            lb_spec=lb_spec,
        )
        started = time.perf_counter()
        result = solver.solve(
            max_nodes=max_nodes, time_limit_s=time_limit_s, lb_spec=lb_spec,
            order_ready_fn=order_fn, dominance="set_based", debug=True,
        )
        wall_s = time.perf_counter() - started
        events = result.debug_info.incumbent_history if result.debug_info is not None else []
        history = [
            {
                "makespan": event.makespan,
                "nodes_expanded": event.nodes_expanded,
                "elapsed_s": event.elapsed_s,
                "primal_gap": _gap(event.makespan, optimum),
                "gap_to_root_bound": _gap(event.makespan, root_lb),
            }
            for event in events
        ]
        target_event = next(
            (event for event in events if target is not None and event.makespan <= target),
            None,
        )
        makespan = result.best_makespan
        proven = makespan is not None and result.done_reason in {
            "search_exhausted", "external_bound_matched",
        }
        dive1 = events[0].makespan if events else None
        rows.append({
            "instance": path.stem,
            "makespan": makespan,
            "root_lower_bound": root_lb,
            "optimal_makespan": optimum,
            "target_makespan": target,
            "dive1_makespan": dive1,
            "dive1_primal_gap": _gap(dive1, optimum),
            "dive1_gap_to_root_bound": _gap(dive1, root_lb),
            "primal_gap": _gap(makespan, optimum),
            "gap_to_root_bound": _gap(makespan, root_lb),
            "optimal_primal": None if optimum is None else makespan == optimum,
            "target_reached": None if target is None else makespan is not None and makespan <= target,
            "nodes_to_target": target_event.nodes_expanded if target_event is not None else None,
            "time_to_target_s": target_event.elapsed_s if target_event is not None else None,
            "proven_optimal": proven,
            "nodes_to_proof": result.nodes_expanded if proven else None,
            "nodes_expanded": result.nodes_expanded,
            "nodes_pruned": result.nodes_pruned,
            "done_reason": result.done_reason,
            "wall_s": wall_s,
            "incumbent_history": history,
        })

    known = [row for row in rows if row["optimal_makespan"] is not None]
    targeted = [row for row in rows if row["target_makespan"] is not None]
    summary = {
        "n_instances": len(rows),
        "n_schedules": sum(row["makespan"] is not None for row in rows),
        "n_known_optima": len(known),
        "n_target_instances": len(targeted),
        "n_primal_gaps": sum(row["primal_gap"] is not None for row in rows),
        "n_proven_optimal": sum(row["proven_optimal"] for row in rows),
        "proof_fraction": _mean(row["proven_optimal"] for row in rows),
        "optimal_primal_fraction": _mean(row["optimal_primal"] for row in known),
        "target_fraction": _mean(row["target_reached"] for row in targeted),
        "mean_makespan": _mean(row["makespan"] for row in rows),
        "mean_dive1_makespan": _mean(row["dive1_makespan"] for row in rows),
        "mean_dive1_primal_gap": _mean(row["dive1_primal_gap"] for row in rows),
        "mean_dive1_gap_to_root_bound": _mean(row["dive1_gap_to_root_bound"] for row in rows),
        "mean_primal_gap": _mean(row["primal_gap"] for row in rows),
        "mean_gap_to_root_bound": _mean(row["gap_to_root_bound"] for row in rows),
        "mean_nodes_expanded": _mean(row["nodes_expanded"] for row in rows),
        "mean_nodes_to_target": _mean(row["nodes_to_target"] for row in rows),
        "mean_nodes_to_proof": _mean(row["nodes_to_proof"] for row in rows),
        "mean_wall_s": _mean(row["wall_s"] for row in rows),
    }
    return {
        "summary": summary, "instances": rows, "dominance": "set_based",
        "observation_schema": schema,
        "budgets": {"max_nodes": max_nodes, "time_limit_s": time_limit_s},
        "lower_bound": lb_spec,
    }


def compare_policies(
    bc_actor: BranchingTransformer,
    trained_actor: BranchingTransformer,
    paths: Iterable[Path | str],
    **kwargs,
) -> dict:
    """Run the original BC, intrinsic-observation BC, and trained actor equally."""
    paths = list(paths)
    if "observation_schema" in kwargs:
        raise ValueError("compare_policies fixes the observation schema for each comparison")
    return {
        "original_bc": evaluate_actor(bc_actor, paths, observation_schema="legacy", **kwargs),
        "intrinsic_bc": evaluate_actor(bc_actor, paths, observation_schema="incumbent_free_v1", **kwargs),
        "trained_actor": evaluate_actor(trained_actor, paths, observation_schema="incumbent_free_v1", **kwargs),
    }
