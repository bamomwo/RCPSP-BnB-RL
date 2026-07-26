from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, List, Optional, Set, Tuple

from rcpsp_bb_rl.bnb.branching import (
    ReadyOrderFn,
    SerialBranchingScheme,
)
from rcpsp_bb_rl.bnb.dominance import build_dominance_engine, normalize_dominance_spec
from rcpsp_bb_rl.bnb.lower_bounds import DEFAULT_LOWER_BOUND_ID, lower_bound
from rcpsp_bb_rl.bnb.precedence import build_predecessors, compute_ready_set
from rcpsp_bb_rl.bnb.scheduling import build_profile, earliest_feasible_start

if TYPE_CHECKING:
    from rcpsp_bb_rl.data.parsing import RCPSPInstance

# Tolerance for treating a lower-bound increase as a real improvement when
# updating path-local stagnation. LBs are integer-valued here, so this just
# means "strictly greater"; the constant keeps the intent explicit.
STAGNATION_EPSILON = 1e-6


@dataclass
class StepContext:
    """
    Accumulated statistics between two consecutive order_ready_fn calls.

    The solver populates this before each call to order_ready_fn so the RL
    environment can compute per-step rewards without reimplementing B&B logic.

    Fields
    ------
    incumbent_before  : best makespan at the previous branching decision
    incumbent_after   : best makespan just before this branching decision
                        (may have improved due to solutions found while advancing)
    lb_pruned         : LB-pruned nodes since the last branching decision
    dom_pruned        : dominance-pruned children since the last branching decision
    nodes_expanded    : total nodes expanded so far (including this one)
    proof_burden      : sum over open stack nodes of max(0, incumbent - node.lb)
                        — total dangerous remaining search space at this moment.
                        Zero when no incumbent exists, or when the stack contains
                        no node with lb < incumbent. (Legacy diagnostic; the
                        optimality-gap reward uses frontier_min_lb instead.)
    frontier_min_lb   : minimum lower bound over the open frontier — the stack
                        nodes plus the node currently being expanded. This is the
                        dual bound: no open branch can yield a makespan below it.
                        The relative optimality gap is
                        (incumbent - frontier_min_lb) / incumbent. Including the
                        node currently being expanded keeps the bound well-defined
                        when the stack is momentarily empty (e.g. at the root).
                        None when the frontier is empty.
    stack_size        : number of open nodes on the stack at this decision (the
                        live frontier size). A proxy for remaining search work;
                        used as a critic-only runtime feature.
    elapsed_s         : wall-clock seconds since the solve started, at this
                        decision. Critic-only; with time_limit_s it gives the
                        time_fraction that predicts time-limit truncation.
    time_limit_s      : the solve time budget (None if unbounded). Critic-only.
    stagnation_depth  : path-local stagnation of the node about to be branched
                        on — consecutive DFS depth steps since the lower bound
                        last improved on this branch. Drives the stagnation
                        multiplier in the reward and is a critic-only feature.
    node_path_best_lb : the best (max) lower bound seen on the active path to the
                        node about to be branched on. Diagnostic / critic-only.
    """
    incumbent_before: Optional[int]
    incumbent_after: Optional[int]
    lb_pruned: int
    dom_pruned: int
    nodes_expanded: int
    proof_burden: int
    frontier_min_lb: Optional[int] = None
    stack_size: int = 0
    elapsed_s: float = 0.0
    time_limit_s: Optional[float] = None
    stagnation_depth: int = 0
    node_path_best_lb: Optional[int] = None


@dataclass
class ScheduleEntry:
    start: int
    finish: int
    duration: int


@dataclass
class BBNode:
    node_id: int
    scheduled: Dict[int, ScheduleEntry]
    ready: Set[int]
    unscheduled: Set[int]
    lower_bound: int
    parent_id: Optional[int]
    action: Optional[str]
    depth: int
    status: str = "pending"  # pending | expanded | pruned | solution
    # Path-local stagnation tracking (for the stagnation-aware reward).
    # path_best_lb     : the maximum lower bound seen on the root->this-node path.
    # stagnation_depth : consecutive depth steps on this path since path_best_lb
    #                    last improved by more than EPSILON. Inherited from the
    #                    parent at child creation, so it is correct under DFS
    #                    backtracking (each node carries its own branch's count).
    path_best_lb: int = 0
    stagnation_depth: int = 0
    # Feasibility cache computed once when this node is expanded: maps each
    # ready act_id -> its earliest feasible start (None if infeasible). Populated
    # by the solver just before branching so the ordering callback (policy) and
    # the solver's own child loop share one computation instead of duplicating it.
    est_map: Optional[Dict[int, Optional[int]]] = None


@dataclass
class IncumbentEvent:
    """Records a single improvement to the best known makespan."""
    rank: int               # 1-based index of this improvement
    makespan: int
    nodes_expanded: int     # how many nodes had been expanded when this was found
    depth: int              # depth of the solution node in the tree


@dataclass
class DebugInfo:
    incumbent_history: List[IncumbentEvent] = field(default_factory=list)
    all_makespans: List[int] = field(default_factory=list)   # every complete schedule seen
    lb_pruned: int = 0
    dominance_pruned: int = 0


@dataclass
class SolverResult:
    best_makespan: Optional[int]
    best_schedule: Optional[Dict[int, ScheduleEntry]]
    nodes: List[BBNode]
    edges: List[Tuple[int, int]]
    nodes_expanded: int
    nodes_pruned: int
    nodes_expanded_after_incumbent: int
    nodes_pruned_after_incumbent: int
    first_incumbent_expanded: Optional[int]
    dominance_enabled: bool
    dominance_rules: Tuple[str, ...]
    dominance_pruned_children: int
    dominance_pruned_by_rule: Dict[str, int]
    done_reason: str = "search_exhausted"  # "search_exhausted" | "time_limit"
    final_proof_burden: int = 0  # sum(incumbent - lb) over open stack nodes at termination
    final_frontier_min_lb: Optional[int] = None  # min lb over open frontier at termination
    debug_info: Optional[DebugInfo] = None


def current_makespan(scheduled: Dict[int, ScheduleEntry]) -> int:
    return max((entry.finish for entry in scheduled.values()), default=0)


class BnBSolver:
    """
    DFS-based branch-and-bound solver for RCPSP.
    """

    def __init__(
        self,
        instance: RCPSPInstance,
        branching_scheme=None,
    ) -> None:
        self.instance = instance
        self.predecessors = build_predecessors(instance)
        self.branching_scheme = branching_scheme or SerialBranchingScheme()
        self.nodes: List[BBNode] = []
        self.edges: List[Tuple[int, int]] = []
        self._next_node_id = 0

    def _new_node_id(self) -> int:
        nid = self._next_node_id
        self._next_node_id += 1
        return nid

    def solve(
        self,
        max_nodes: Optional[int] = None,
        order_ready_fn: Optional[ReadyOrderFn] = None,
        time_limit_s: Optional[float] = None,
        lb_spec: object = DEFAULT_LOWER_BOUND_ID,
        dominance: object = False,
        target_makespan: Optional[int] = None,
        stop_on_first_solution: bool = False,
        debug: bool = False,
    ) -> SolverResult:
        unscheduled = set(self.instance.activities.keys())
        ready = compute_ready_set(unscheduled, set(), self.predecessors)
        dominance_cfg = normalize_dominance_spec(dominance)
        dominance_engine = build_dominance_engine(
            instance=self.instance,
            predecessors=self.predecessors,
            dominance=dominance_cfg,
        )

        root_id = self._new_node_id()
        root_lb = lower_bound(self.instance, unscheduled, {}, lb_id=lb_spec)
        root = BBNode(
            node_id=root_id,
            scheduled={},
            ready=ready,
            unscheduled=unscheduled,
            lower_bound=root_lb,
            parent_id=None,
            action=None,
            depth=0,
            path_best_lb=root_lb,
            stagnation_depth=0,
        )
        self.nodes.append(root)
        dominance_engine.register_state(
            root.unscheduled,
            root.scheduled,
            root.lower_bound,
        )

        stack: List[int] = [root_id]
        if target_makespan is not None and target_makespan < 0:
            raise ValueError("target_makespan must be >= 0 when provided.")

        incumbent_bound: Optional[int]
        if target_makespan is None:
            incumbent_bound = None
        else:
            incumbent_bound = int(target_makespan) + 1

        best_makespan: Optional[int] = None
        best_schedule: Optional[Dict[int, ScheduleEntry]] = None
        nodes_expanded = 0
        nodes_pruned = 0
        nodes_expanded_after_incumbent = 0
        nodes_pruned_after_incumbent = 0
        first_incumbent_expanded: Optional[int] = None
        seen_incumbent = False
        debug_info: Optional[DebugInfo] = DebugInfo() if debug else None

        # Counters that accumulate between consecutive order_ready_fn calls.
        # Reset each time we are about to call order_ready_fn.
        _step_lb_pruned: int = 0
        _step_dom_pruned: int = 0
        _step_incumbent_before: Optional[int] = None

        start_time_monotonic = time.perf_counter()

        def time_exceeded() -> bool:
            if time_limit_s is None:
                return False
            return (time.perf_counter() - start_time_monotonic) >= time_limit_s

        def _compute_proof_burden() -> int:
            """Sum of (incumbent - node.lb) over open stack nodes with lb < incumbent."""
            if best_makespan is None:
                return 0
            total = 0
            for nid in stack:
                lb = self.nodes[nid].lower_bound
                if lb < best_makespan:
                    total += best_makespan - lb
            return total

        def _compute_frontier_min_lb(current_lb: Optional[int]) -> Optional[int]:
            """
            Minimum lower bound over the open frontier — the nodes still on the
            stack plus the node currently being expanded (current_lb). This is
            the dual bound: no open branch can produce a makespan below it.

            Including current_lb keeps the bound well-defined when the stack is
            momentarily empty (e.g. while expanding the root). Returns None only
            when there is no open node at all (current_lb is None and the stack
            is empty), which at termination means the search is exhausted.
            """
            best = current_lb
            for nid in stack:
                lb = self.nodes[nid].lower_bound
                if best is None or lb < best:
                    best = lb
            return best

        def _make_step_context(current_node: Optional[BBNode] = None) -> StepContext:
            nonlocal _step_incumbent_before, _step_lb_pruned, _step_dom_pruned
            current_lb = current_node.lower_bound if current_node is not None else None
            ctx = StepContext(
                incumbent_before=_step_incumbent_before,
                incumbent_after=best_makespan,
                lb_pruned=_step_lb_pruned,
                dom_pruned=_step_dom_pruned,
                nodes_expanded=nodes_expanded,
                proof_burden=_compute_proof_burden(),
                frontier_min_lb=_compute_frontier_min_lb(current_lb),
                stack_size=len(stack),
                elapsed_s=time.perf_counter() - start_time_monotonic,
                time_limit_s=time_limit_s,
                stagnation_depth=(
                    current_node.stagnation_depth if current_node is not None else 0
                ),
                node_path_best_lb=(
                    current_node.path_best_lb if current_node is not None else None
                ),
            )
            # Snapshot state for the *next* branching call. Any incumbent
            # improvement or pruning that happens between now and the next
            # call will then show up as a delta relative to these values.
            _step_incumbent_before = best_makespan
            _step_lb_pruned = 0
            _step_dom_pruned = 0
            return ctx

        def _wrapped_order_fn(node: BBNode, incumbent: Optional[int]) -> List[int]:
            """Inject StepContext into order_ready_fn when it accepts it."""
            if order_ready_fn is None:
                from rcpsp_bb_rl.bnb.branching_order import order_by_activity_id
                return order_by_activity_id(node, incumbent)
            import inspect
            sig = inspect.signature(order_ready_fn)
            if len(sig.parameters) >= 3:
                return list(order_ready_fn(node, incumbent, _make_step_context(node)))
            return list(order_ready_fn(node, incumbent))

        while stack and ((max_nodes is None) or (nodes_expanded < max_nodes)) and not time_exceeded():
            if stop_on_first_solution and best_makespan is not None:
                break

            node_id = stack.pop()
            node = self.nodes[node_id]

            incumbent = incumbent_bound
            if incumbent is not None and node.lower_bound >= incumbent:
                node.status = "pruned"
                nodes_pruned += 1
                _step_lb_pruned += 1
                if debug_info is not None:
                    debug_info.lb_pruned += 1
                if seen_incumbent:
                    nodes_pruned_after_incumbent += 1
                continue

            if not node.unscheduled:
                node.status = "solution"
                makespan = current_makespan(node.scheduled)
                if debug_info is not None:
                    debug_info.all_makespans.append(makespan)
                if incumbent_bound is None or makespan < incumbent_bound:
                    incumbent_bound = makespan
                    best_makespan = makespan
                    best_schedule = node.scheduled
                    if debug_info is not None:
                        debug_info.incumbent_history.append(IncumbentEvent(
                            rank=len(debug_info.incumbent_history) + 1,
                            makespan=makespan,
                            nodes_expanded=nodes_expanded,
                            depth=node.depth,
                        ))
                    if not seen_incumbent:
                        first_incumbent_expanded = nodes_expanded
                        seen_incumbent = True
                continue

            if not node.ready:
                node.status = "pruned"
                nodes_pruned += 1
                _step_lb_pruned += 1
                if seen_incumbent:
                    nodes_pruned_after_incumbent += 1
                continue

            # Build the resource profile and earliest-feasible-start map ONCE,
            # before ordering. The ordering callback (e.g. the branching
            # policy) needs earliest-starts to featurise candidates, and the
            # child loop below needs them to place activities. Computing them
            # here and attaching to the node lets both share one computation
            # instead of each recomputing profile + earliest_feasible_start.
            horizon_hint = sum(act.duration for act in self.instance.activities.values())
            node_horizon = incumbent_bound if incumbent_bound is not None else horizon_hint
            node_profile = build_profile(
                self.instance.activities,
                self.instance.resource_caps,
                node.scheduled,
                horizon=node_horizon,
            )
            node.est_map = {
                rid: earliest_feasible_start(
                    self.instance,
                    self.predecessors,
                    node.scheduled,
                    rid,
                    incumbent_bound,
                    profile=node_profile,
                )
                for rid in node.ready
            }

            acts = self.branching_scheme.choose_activities(
                node=node,
                incumbent=incumbent_bound,
                order_ready_fn=_wrapped_order_fn,
            )

            node.status = "expanded"
            nodes_expanded += 1
            if seen_incumbent:
                nodes_expanded_after_incumbent += 1

            # Reverse push for DFS/LIFO.
            for act_id in reversed(acts):
                # Reuse the earliest-start computed once above (shared with
                # the ordering callback). Fall back to a direct computation
                # only if this act was not in node.ready (defensive).
                if node.est_map is not None and act_id in node.est_map:
                    est_start = node.est_map[act_id]
                else:
                    est_start = earliest_feasible_start(
                        self.instance,
                        self.predecessors,
                        node.scheduled,
                        act_id,
                        incumbent_bound,
                        profile=node_profile,
                    )
                if est_start is None:
                    continue

                duration = self.instance.activities[act_id].duration
                finish = est_start + duration

                child_scheduled = dict(node.scheduled)
                child_scheduled[act_id] = ScheduleEntry(
                    start=est_start,
                    finish=finish,
                    duration=duration,
                )

                child_unscheduled = set(node.unscheduled)
                child_unscheduled.discard(act_id)

                child_ready = compute_ready_set(
                    child_unscheduled,
                    set(child_scheduled.keys()),
                    self.predecessors,
                )

                child_lb = lower_bound(
                    self.instance,
                    child_unscheduled,
                    child_scheduled,
                    lb_id=lb_spec,
                )

                pruned_rule = dominance_engine.prune_child(
                    parent_scheduled=node.scheduled,
                    child_scheduled=child_scheduled,
                    child_unscheduled=child_unscheduled,
                    child_lb=child_lb,
                    act_id=act_id,
                    child_start=est_start,
                )
                if pruned_rule is not None:
                    if debug_info is not None:
                        debug_info.dominance_pruned += 1
                    _step_dom_pruned += 1
                    continue

                child_id = self._new_node_id()
                # Path-local stagnation: inherit from the parent. If this child's
                # LB strictly improves on the best LB seen along the parent's path,
                # the path advanced -> reset stagnation; otherwise the branch went
                # one step deeper without tightening the bound -> increment. Using
                # the path MAX (not the parent LB) makes this robust to a child LB
                # that dips below its parent, and inheritance makes it correct
                # under DFS backtracking (siblings carry their own branch counts).
                improved = child_lb > node.path_best_lb + STAGNATION_EPSILON
                child_path_best_lb = max(node.path_best_lb, child_lb)
                child_stagnation_depth = 0 if improved else node.stagnation_depth + 1
                child_node = BBNode(
                    node_id=child_id,
                    scheduled=child_scheduled,
                    ready=child_ready,
                    unscheduled=child_unscheduled,
                    lower_bound=child_lb,
                    parent_id=node_id,
                    action=f"act {act_id}@{est_start}",
                    depth=node.depth + 1,
                    path_best_lb=child_path_best_lb,
                    stagnation_depth=child_stagnation_depth,
                )

                self.nodes.append(child_node)
                self.edges.append((node_id, child_id))
                stack.append(child_id)

            # The feasibility cache has served both consumers (ordering callback
            # and the child loop above); free it so long runs don't retain one
            # dict per expanded node in self.nodes.
            node.est_map = None

        solver_done_reason = "time_limit" if (stack and time_exceeded()) else "search_exhausted"
        final_proof_burden = _compute_proof_burden()
        # At termination the frontier is exactly the remaining stack (no node is
        # being expanded). On an exhausted search the stack is empty -> None,
        # which the env treats as a fully closed gap (proof complete).
        final_frontier_min_lb = _compute_frontier_min_lb(None)

        return SolverResult(
            best_makespan=best_makespan,
            best_schedule=best_schedule,
            nodes=self.nodes,
            edges=self.edges,
            nodes_expanded=nodes_expanded,
            nodes_pruned=nodes_pruned,
            nodes_expanded_after_incumbent=nodes_expanded_after_incumbent,
            nodes_pruned_after_incumbent=nodes_pruned_after_incumbent,
            first_incumbent_expanded=first_incumbent_expanded,
            dominance_enabled=dominance_cfg.enabled,
            dominance_rules=tuple(dominance_cfg.rules),
            dominance_pruned_children=dominance_engine.stats.pruned_children,
            dominance_pruned_by_rule=dict(dominance_engine.stats.pruned_by_rule),
            done_reason=solver_done_reason,
            final_proof_burden=final_proof_burden,
            final_frontier_min_lb=final_frontier_min_lb,
            debug_info=debug_info,
        )


def solve_serial(
    instance: RCPSPInstance,
    max_nodes: Optional[int] = None,
    order_ready_fn: Optional[ReadyOrderFn] = None,
    time_limit_s: Optional[float] = None,
    lb_spec: object = DEFAULT_LOWER_BOUND_ID,
    dominance: object = False,
    target_makespan: Optional[int] = None,
    stop_on_first_solution: bool = False,
    debug: bool = False,
) -> SolverResult:
    solver = BnBSolver(
        instance=instance,
        branching_scheme=SerialBranchingScheme(),
    )
    return solver.solve(
        max_nodes=max_nodes,
        order_ready_fn=order_ready_fn,
        time_limit_s=time_limit_s,
        lb_spec=lb_spec,
        dominance=dominance,
        target_makespan=target_makespan,
        stop_on_first_solution=stop_on_first_solution,
        debug=debug,
    )
