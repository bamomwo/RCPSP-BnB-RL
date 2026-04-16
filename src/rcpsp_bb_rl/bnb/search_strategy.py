from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, List, Optional, Protocol

from rcpsp_bb_rl.bnb.lower_bounds import DEFAULT_LOWER_BOUND_ID, lower_bound

if TYPE_CHECKING:
    from rcpsp_bb_rl.bnb.branching import ReadyOrderFn
    from rcpsp_bb_rl.bnb.solver import SolverResult
    from rcpsp_bb_rl.data.parsing import RCPSPInstance


class SolverFn(Protocol):
    def __call__(
        self,
        *,
        instance: "RCPSPInstance",
        max_nodes: Optional[int] = None,
        order_ready_fn: Optional["ReadyOrderFn"] = None,
        time_limit_s: Optional[float] = None,
        lb_spec: object = DEFAULT_LOWER_BOUND_ID,
        dominance: object = False,
        target_makespan: Optional[int] = None,
        stop_on_first_solution: bool = False,
    ) -> "SolverResult":
        ...


@dataclass
class SearchPassResult:
    pass_index: int
    target_makespan: Optional[int]
    artificial_upper_bound: Optional[int]
    feasible_found: bool
    best_makespan: Optional[int]
    nodes_expanded: int
    nodes_pruned: int
    search_exhausted: bool


@dataclass
class SearchStrategyResult:
    strategy_name: str
    best_makespan: Optional[int]
    best_schedule: Optional[dict]
    known_lower_bound: Optional[int]
    known_upper_bound: Optional[int]
    optimal: bool
    truncated: bool
    passes: List[SearchPassResult] = field(default_factory=list)
    total_nodes_expanded: int = 0
    total_nodes_pruned: int = 0
    last_solver_result: Optional["SolverResult"] = None


def _compute_root_lower_bound(
    instance: "RCPSPInstance",
    lb_spec: object,
) -> int:
    unscheduled = set(instance.activities.keys())
    return lower_bound(instance, unscheduled, {}, lb_id=lb_spec)


def _search_exhausted(result: "SolverResult") -> bool:
    return not any(node.status == "pending" for node in result.nodes)


class UpperBoundSearchStrategy:
    name = "ubs"

    def run(
        self,
        *,
        solver_fn: SolverFn,
        instance: "RCPSPInstance",
        max_nodes: Optional[int] = None,
        order_ready_fn: Optional["ReadyOrderFn"] = None,
        time_limit_s: Optional[float] = None,
        lb_spec: object = DEFAULT_LOWER_BOUND_ID,
        dominance: object = False,
    ) -> SearchStrategyResult:
        initial_lb = _compute_root_lower_bound(instance, lb_spec)
        result = solver_fn(
            instance=instance,
            max_nodes=max_nodes,
            order_ready_fn=order_ready_fn,
            time_limit_s=time_limit_s,
            lb_spec=lb_spec,
            dominance=dominance,
            target_makespan=None,
            stop_on_first_solution=False,
        )

        exhausted = _search_exhausted(result)
        best_makespan = result.best_makespan
        best_schedule = result.best_schedule

        return SearchStrategyResult(
            strategy_name=self.name,
            best_makespan=best_makespan,
            best_schedule=best_schedule,
            known_lower_bound=initial_lb,
            known_upper_bound=best_makespan,
            optimal=(best_makespan is not None and exhausted),
            truncated=not exhausted,
            passes=[
                SearchPassResult(
                    pass_index=1,
                    target_makespan=None,
                    artificial_upper_bound=None,
                    feasible_found=(best_makespan is not None),
                    best_makespan=best_makespan,
                    nodes_expanded=result.nodes_expanded,
                    nodes_pruned=result.nodes_pruned,
                    search_exhausted=exhausted,
                )
            ],
            total_nodes_expanded=result.nodes_expanded,
            total_nodes_pruned=result.nodes_pruned,
            last_solver_result=result,
        )


class LowerBoundSearchStrategy:
    name = "lbs"

    def __init__(self, initial_lower_bound: Optional[int] = None) -> None:
        self.initial_lower_bound = initial_lower_bound

    def run(
        self,
        *,
        solver_fn: SolverFn,
        instance: "RCPSPInstance",
        max_nodes: Optional[int] = None,
        order_ready_fn: Optional["ReadyOrderFn"] = None,
        time_limit_s: Optional[float] = None,
        lb_spec: object = DEFAULT_LOWER_BOUND_ID,
        dominance: object = False,
    ) -> SearchStrategyResult:
        initial_lb = (
            int(self.initial_lower_bound)
            if self.initial_lower_bound is not None
            else _compute_root_lower_bound(instance, lb_spec)
        )
        if initial_lb < 0:
            raise ValueError("initial_lower_bound must be >= 0.")

        proven_lb = initial_lb
        current_target = initial_lb
        total_nodes_expanded = 0
        total_nodes_pruned = 0
        remaining_nodes = max_nodes
        passes: List[SearchPassResult] = []
        last_result: Optional["SolverResult"] = None

        deadline = None if time_limit_s is None else time.perf_counter() + float(time_limit_s)
        pass_index = 0

        while True:
            pass_index += 1

            if deadline is not None:
                remaining_time = deadline - time.perf_counter()
                if remaining_time <= 0:
                    return SearchStrategyResult(
                        strategy_name=self.name,
                        best_makespan=None,
                        best_schedule=None,
                        known_lower_bound=proven_lb,
                        known_upper_bound=None,
                        optimal=False,
                        truncated=True,
                        passes=passes,
                        total_nodes_expanded=total_nodes_expanded,
                        total_nodes_pruned=total_nodes_pruned,
                        last_solver_result=last_result,
                    )
            else:
                remaining_time = None

            if remaining_nodes is not None and remaining_nodes <= 0:
                return SearchStrategyResult(
                    strategy_name=self.name,
                    best_makespan=None,
                    best_schedule=None,
                    known_lower_bound=proven_lb,
                    known_upper_bound=None,
                    optimal=False,
                    truncated=True,
                    passes=passes,
                    total_nodes_expanded=total_nodes_expanded,
                    total_nodes_pruned=total_nodes_pruned,
                    last_solver_result=last_result,
                )

            artificial_upper_bound = current_target + 1
            result = solver_fn(
                instance=instance,
                max_nodes=remaining_nodes,
                order_ready_fn=order_ready_fn,
                time_limit_s=remaining_time,
                lb_spec=lb_spec,
                dominance=dominance,
                target_makespan=current_target,
                stop_on_first_solution=True,
            )
            last_result = result

            exhausted = _search_exhausted(result)
            feasible_found = (
                result.best_makespan is not None and int(result.best_makespan) <= int(current_target)
            )

            passes.append(
                SearchPassResult(
                    pass_index=pass_index,
                    target_makespan=current_target,
                    artificial_upper_bound=artificial_upper_bound,
                    feasible_found=feasible_found,
                    best_makespan=result.best_makespan,
                    nodes_expanded=result.nodes_expanded,
                    nodes_pruned=result.nodes_pruned,
                    search_exhausted=exhausted,
                )
            )

            total_nodes_expanded += result.nodes_expanded
            total_nodes_pruned += result.nodes_pruned
            if remaining_nodes is not None:
                remaining_nodes -= result.nodes_expanded

            if feasible_found:
                return SearchStrategyResult(
                    strategy_name=self.name,
                    best_makespan=result.best_makespan,
                    best_schedule=result.best_schedule,
                    known_lower_bound=current_target,
                    known_upper_bound=result.best_makespan,
                    optimal=True,
                    truncated=False,
                    passes=passes,
                    total_nodes_expanded=total_nodes_expanded,
                    total_nodes_pruned=total_nodes_pruned,
                    last_solver_result=result,
                )

            if not exhausted:
                return SearchStrategyResult(
                    strategy_name=self.name,
                    best_makespan=None,
                    best_schedule=None,
                    known_lower_bound=proven_lb,
                    known_upper_bound=None,
                    optimal=False,
                    truncated=True,
                    passes=passes,
                    total_nodes_expanded=total_nodes_expanded,
                    total_nodes_pruned=total_nodes_pruned,
                    last_solver_result=result,
                )

            proven_lb = current_target + 1
            current_target += 1


def build_search_strategy(
    name: str,
    **kwargs: Any,
) -> UpperBoundSearchStrategy | LowerBoundSearchStrategy:
    normalized = str(name).strip().lower()

    if normalized == "ubs":
        return UpperBoundSearchStrategy()
    if normalized == "lbs":
        return LowerBoundSearchStrategy(**kwargs)

    raise ValueError(f"Unknown search strategy: {name}. Supported: ubs, lbs")
