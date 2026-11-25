"""The five B&B outcome metrics used for policy evaluation."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Mapping, Optional

import torch

from rcpsp_bnb_rl.bnb.lower_bounds import DEFAULT_LOWER_BOUND_ID
from rcpsp_bnb_rl.bnb.solver import BnBSolver
from rcpsp_bnb_rl.data.parsing import load_instance
from rcpsp_bnb_rl.ml.models import BranchingTransformer
from rcpsp_bnb_rl.ml.observations import OBSERVATION_SCHEMA
from rcpsp_bnb_rl.ml.rl.policy_guidance import make_policy_order_fn


def _mean(values: Iterable[Optional[float]]) -> Optional[float]:
    available = [float(value) for value in values if value is not None]
    return sum(available) / len(available) if available else None


def _gap(makespan: Optional[int], reference: Optional[int]) -> Optional[float]:
    """Percentage gap to a known optimum; zero references are undefined."""
    if makespan is None or reference is None or reference <= 0:
        return None
    return 100.0 * (makespan - reference) / reference


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
    optimal_makespans: Optional[Mapping[str, int]] = None,
) -> dict:
    """Report solved (proven), primal_opt, primal_gap, dive1_gap, and mean nodes.

    ``solved`` requires a completed optimality proof; matching a known optimum
    on a truncated search only sets ``primal_opt``. Proof rate uses all instances.
    Primal attainment uses known optima, including
    schedule failures; gap means use available schedules with positive optima.
    Gaps are percentages in both instance rows and the summary.
    """
    if max_nodes is not None and (isinstance(max_nodes, bool) or max_nodes < 0):
        raise ValueError("max_nodes must be non-negative or None")
    if time_limit_s is not None and time_limit_s < 0:
        raise ValueError("time_limit_s must be non-negative or None")
    schema = actor.observation_schema
    if schema != OBSERVATION_SCHEMA:
        raise ValueError("Evaluation requires resource_set observations")
    optima = _references(optimal_makespans)
    rows = []
    actor.eval()

    for path_value in paths:
        path = Path(path_value)
        key = path.stem.lower()
        instance = load_instance(path)
        optimum = optima.get(key)
        solver = BnBSolver(instance)
        order_fn = make_policy_order_fn(
            instance=instance, model=actor, device=device,
            predecessors=solver.predecessors,
            lb_spec=lb_spec,
        )
        result = solver.solve(
            max_nodes=max_nodes, time_limit_s=time_limit_s, lb_spec=lb_spec,
            order_ready_fn=order_fn, dominance="set_based", debug=True,
        )
        events = result.debug_info.incumbent_history if result.debug_info is not None else []
        makespan = result.best_makespan
        proven = makespan is not None and result.done_reason in {
            "search_exhausted", "external_bound_matched",
        }
        dive1 = events[0].makespan if events else None
        rows.append({
            "instance": path.stem,
            "solved": proven,
            "primal_opt": None if optimum is None else makespan == optimum,
            "primal_gap": _gap(makespan, optimum),
            "dive1_gap": _gap(dive1, optimum),
            "nodes": result.nodes_expanded,
        })

    return {
        "summary": {
            "solved": _mean(row["solved"] for row in rows),
            "primal_opt": _mean(row["primal_opt"] for row in rows),
            "primal_gap": _mean(row["primal_gap"] for row in rows),
            "dive1_gap": _mean(row["dive1_gap"] for row in rows),
            "nodes": _mean(row["nodes"] for row in rows),
        },
        "instances": rows,
    }


def compare_policies(
    bc_actor: BranchingTransformer,
    trained_actor: BranchingTransformer,
    paths: Iterable[Path | str],
    **kwargs,
) -> dict:
    """Compare BC and trained actors under identical DFS budgets."""
    paths = list(paths)
    return {
        "bc": evaluate_actor(bc_actor, paths, **kwargs),
        "trained_actor": evaluate_actor(trained_actor, paths, **kwargs),
    }
