from __future__ import annotations

import time
from collections import Counter
from collections.abc import Mapping, Set as AbstractSet
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable, Dict, Iterator, List, Optional, Set, Tuple

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


@dataclass(frozen=True)
class DebugBranchChoice:
    """One early choice; node id distinguishes identical activities on different paths.

    Rank/count refer to ordered candidates BEFORE feasibility/dominance filtering.
    Forced decisions (one candidate) are excluded from prefixes.
    """

    decision_node_id: int
    activity: int
    start: int
    rank: int
    candidate_count: int


DebugPrefix = Tuple[DebugBranchChoice, ...]


@dataclass
class DebugPrefixCount:
    prefix: DebugPrefix
    count: int


def _debug_bound_distribution(values: List[int]) -> Dict[str, object]:
    """Exact histogram and linearly interpolated quartiles of queued node bounds."""
    ordered = sorted(values)

    def quantile(fraction: float) -> Optional[float]:
        if not ordered:
            return None
        index = (len(ordered) - 1) * fraction
        lo = int(index)
        hi = min(lo + 1, len(ordered) - 1)
        return ordered[lo] + (ordered[hi] - ordered[lo]) * (index - lo)

    return {
        "count": len(ordered),
        "min": ordered[0] if ordered else None,
        "p25": quantile(0.25),
        "median": quantile(0.5),
        "p75": quantile(0.75),
        "max": ordered[-1] if ordered else None,
        "histogram": dict(sorted(Counter(values).items())),
    }


@dataclass
class DebugCheckpoint:
    reason: str  # periodic | incumbent | termination
    termination: Optional[str]
    nodes_expanded: int
    elapsed_s: float
    incumbent: Optional[int]
    pruning_cutoff: Optional[int]
    external_lower_bound: Optional[int]
    nodes_since_improvement: Optional[int]
    seconds_since_improvement: Optional[float]
    incumbent_prefix: Optional[DebugPrefix]
    incumbent_depth: Optional[int]
    frontier_size: int
    internal_lb: Dict[str, object]
    effective_lb: Dict[str, object]
    below_incumbent: Optional[int]
    below_incumbent_pct: Optional[float]
    at_or_above_incumbent: Optional[int]
    at_or_above_incumbent_pct: Optional[float]
    internal_at_or_below_external: Optional[int]
    internal_at_or_below_external_pct: Optional[float]
    proof_burden: Optional[int]
    last_expanded_node_id: Optional[int]
    last_expanded_depth: Optional[int]
    last_expanded_prefix: DebugPrefix
    next_pending_prefix: Optional[DebugPrefix]
    recent_expansions_by_prefix: List[DebugPrefixCount]
    frontier_by_first_branch: List[DebugPrefixCount]
    interval: Dict[str, int]
    totals: Dict[str, int]
    interval_elapsed_s: float
    nodes_per_second: Optional[float]
    lb_pruned_pct_of_visited: Optional[float]
    dominance_pruned_pct_of_bounded_children: Optional[float]


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
    # Populated only for opt-in checkpoint diagnostics in full-tree debug mode.
    internal_lower_bound: Optional[int] = None
    debug_prefix: DebugPrefix = ()


class ActivityMaskSet(AbstractSet[int]):
    """Read-only set view backed by an activity bitmask."""

    __slots__ = ("mask", "activity_to_bit", "bit_to_activity")

    def __init__(
        self,
        mask: int,
        activity_to_bit: Dict[int, int],
        bit_to_activity: Tuple[int, ...],
    ) -> None:
        self.mask = int(mask)
        self.activity_to_bit = activity_to_bit
        self.bit_to_activity = bit_to_activity

    def __contains__(self, activity: object) -> bool:
        if not isinstance(activity, int):
            return False
        bit = self.activity_to_bit.get(activity)
        return bit is not None and bool(self.mask & (1 << bit))

    def __iter__(self) -> Iterator[int]:
        remaining = self.mask
        while remaining:
            least_bit = remaining & -remaining
            bit_index = least_bit.bit_length() - 1
            yield self.bit_to_activity[bit_index]
            remaining ^= least_bit

    def __len__(self) -> int:
        return self.mask.bit_count()


class ScheduleOverlay(Mapping[int, ScheduleEntry]):
    """Read-only one-decision overlay on a reconstructed parent schedule."""

    __slots__ = ("parent", "activity", "entry")

    def __init__(
        self,
        parent: Dict[int, ScheduleEntry],
        activity: int,
        entry: ScheduleEntry,
    ) -> None:
        self.parent = parent
        self.activity = int(activity)
        self.entry = entry

    def __getitem__(self, activity: int) -> ScheduleEntry:
        if activity == self.activity:
            return self.entry
        return self.parent[activity]

    def __iter__(self) -> Iterator[int]:
        yield from self.parent
        if self.activity not in self.parent:
            yield self.activity

    def __len__(self) -> int:
        return len(self.parent) + int(self.activity not in self.parent)


@dataclass(slots=True)
class CompactBBNode:
    """Live-frontier node storing only one decision plus compact activity masks."""

    node_id: int
    parent: Optional["CompactBBNode"]
    selected_activity: Optional[int]
    selected_start: int
    scheduled_mask: int
    unscheduled_mask: int
    ready_mask: int
    start_times_code: int
    lower_bound: int
    depth: int
    makespan: int = 0
    path_best_lb: int = 0
    stagnation_depth: int = 0


@dataclass(slots=True)
class ExpandedNodeView:
    """Temporary BBNode-compatible view used only while one node is expanded."""

    node_id: int
    scheduled: Dict[int, ScheduleEntry]
    ready: ActivityMaskSet
    unscheduled: ActivityMaskSet
    lower_bound: int
    parent_id: Optional[int]
    action: Optional[str]
    depth: int
    path_best_lb: int = 0
    stagnation_depth: int = 0
    est_map: Optional[Dict[int, Optional[int]]] = None


@dataclass
class IncumbentEvent:
    """Records a single improvement to the best known makespan."""
    rank: int               # 1-based index of this improvement
    makespan: int
    nodes_expanded: int     # how many nodes had been expanded when this was found
    depth: int              # depth of the solution node in the tree
    elapsed_s: float = 0.0
    prefix: DebugPrefix = ()


@dataclass
class DebugInfo:
    incumbent_history: List[IncumbentEvent] = field(default_factory=list)
    all_makespans: List[int] = field(default_factory=list)   # every complete schedule seen
    lb_pruned: int = 0
    dominance_pruned: int = 0
    checkpoints: List[DebugCheckpoint] = field(default_factory=list)


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
    done_reason: str = "search_exhausted"  # search_exhausted | time_limit | node_limit | external_bound_matched
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
        external_lower_bound: Optional[int] = None,
        memory_bounded: bool = False,
        debug_checkpoint_nodes: Optional[int] = None,
        debug_prefix_depth: int = 3,
        debug_checkpoint_callback: Optional[Callable[[DebugCheckpoint], None]] = None,
    ) -> SolverResult:
        """Solve the instance with depth-first branch and bound.

        ``external_lower_bound`` is an optional certified lower bound for the
        complete instance.  Because every partial node restricts the root
        feasible set, the same value is a valid (static) lower-bound floor for
        every descendant.  It is therefore combined with the configured
        state-dependent bound as ``max(internal_lb, external_lower_bound)``.

        The argument is opt-in and defaults to ``None`` so existing training
        and evaluation behavior is unchanged.  With ``memory_bounded=True``,
        live DFS nodes use parent-linked decisions and activity bitmasks rather
        than copied schedules and sets. Completed nodes and tree edges are not
        retained in the returned result. This mode is intended for long-running
        evaluation, where callers consume aggregate solver statistics rather
        than reconstructing the full tree.

        Checkpoint diagnostics are opt-in via ``debug_checkpoint_nodes`` and
        require ``debug=True`` with full-tree storage. They observe the pending
        stack after complete expansion, on incumbent updates, and at termination;
        they never reorder or remove nodes. Callback time counts toward the run
        time limit. Use a node budget for debug-on/off search equivalence checks.
        """
        collect_checkpoints = debug_checkpoint_nodes is not None
        if collect_checkpoints:
            if not debug or memory_bounded:
                raise ValueError("Checkpoint diagnostics require debug=True and memory_bounded=False.")
            if debug_checkpoint_nodes <= 0 or debug_prefix_depth <= 0:
                raise ValueError("Debug checkpoint interval and prefix depth must be positive.")
        elif debug_checkpoint_callback is not None:
            raise ValueError("A debug checkpoint callback requires debug_checkpoint_nodes.")
        if external_lower_bound is not None:
            external_lower_bound = int(external_lower_bound)
            if external_lower_bound < 0:
                raise ValueError("external_lower_bound must be >= 0 when provided.")

        def _internal_lower_bound(
            node_unscheduled,
            node_scheduled: Dict[int, ScheduleEntry],
        ) -> int:
            return lower_bound(
                self.instance,
                node_unscheduled,
                node_scheduled,
                lb_id=lb_spec,
            )

        def _effective_lower_bound(internal: int) -> int:
            return internal if external_lower_bound is None else max(internal, external_lower_bound)

        unscheduled = set(self.instance.activities.keys())
        ready = compute_ready_set(unscheduled, set(), self.predecessors)
        activity_ids = tuple(sorted(self.instance.activities))
        activity_to_bit = {
            activity_id: bit_index
            for bit_index, activity_id in enumerate(activity_ids)
        }
        full_activity_mask = (1 << len(activity_ids)) - 1
        horizon_hint = sum(
            int(activity.duration) for activity in self.instance.activities.values()
        )
        start_time_bits = max(1, int(horizon_hint).bit_length())

        def _activity_mask(activities) -> int:
            mask = 0
            for activity_id in activities:
                mask |= 1 << activity_to_bit[activity_id]
            return mask

        predecessor_masks = {
            activity_id: _activity_mask(self.predecessors.get(activity_id, set()))
            for activity_id in activity_ids
        }

        def _mask_set(mask: int) -> ActivityMaskSet:
            return ActivityMaskSet(mask, activity_to_bit, activity_ids)

        def _ready_mask(unscheduled_mask: int, scheduled_mask: int) -> int:
            result = 0
            remaining = unscheduled_mask
            while remaining:
                least_bit = remaining & -remaining
                bit_index = least_bit.bit_length() - 1
                activity_id = activity_ids[bit_index]
                if predecessor_masks[activity_id] & ~scheduled_mask == 0:
                    result |= least_bit
                remaining ^= least_bit
            return result

        def _reconstruct_schedule(
            compact_node: CompactBBNode,
        ) -> Dict[int, ScheduleEntry]:
            decisions: List[Tuple[int, int]] = []
            cursor: Optional[CompactBBNode] = compact_node
            while cursor is not None and cursor.selected_activity is not None:
                decisions.append((cursor.selected_activity, cursor.selected_start))
                cursor = cursor.parent

            schedule: Dict[int, ScheduleEntry] = {}
            for activity_id, start in reversed(decisions):
                duration = int(self.instance.activities[activity_id].duration)
                schedule[activity_id] = ScheduleEntry(
                    start=start,
                    finish=start + duration,
                    duration=duration,
                )
            return schedule

        def _compact_state_key(compact_node: CompactBBNode):
            # The scheduled mask distinguishes unscheduled activities from
            # activities fixed at start zero. Fixed-width packed starts make
            # this an exact, order-independent dominance key using two ints.
            return compact_node.scheduled_mask, compact_node.start_times_code
        dominance_cfg = normalize_dominance_spec(dominance)
        dominance_engine = build_dominance_engine(
            instance=self.instance,
            predecessors=self.predecessors,
            dominance=dominance_cfg,
            retain_state_history=not memory_bounded,
        )

        root_id = self._new_node_id()
        root_internal_lb = _internal_lower_bound(unscheduled, {})
        root_lb = _effective_lower_bound(root_internal_lb)
        if memory_bounded:
            root = CompactBBNode(
                node_id=root_id,
                parent=None,
                selected_activity=None,
                selected_start=0,
                scheduled_mask=0,
                unscheduled_mask=full_activity_mask,
                ready_mask=_activity_mask(ready),
                start_times_code=0,
                lower_bound=root_lb,
                depth=0,
                path_best_lb=root_lb,
                stagnation_depth=0,
            )
        else:
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
            if collect_checkpoints:
                root.internal_lower_bound = root_internal_lb
        if memory_bounded:
            assert isinstance(root, CompactBBNode)
            dominance_engine.register_state(
                _mask_set(root.unscheduled_mask),
                {},
                root.lower_bound,
                state_key=_compact_state_key(root),
            )
        else:
            assert isinstance(root, BBNode)
            dominance_engine.register_state(
                root.unscheduled,
                root.scheduled,
                root.lower_bound,
            )

        # Full-tree mode stores integer node ids in ``self.nodes``.  The
        # memory-bounded mode stores the live BBNode objects directly on the
        # DFS stack and never appends historical nodes to ``self.nodes``.
        stack: List[object] = [root] if memory_bounded else [root_id]
        if target_makespan is not None and target_makespan < 0:
            raise ValueError("target_makespan must be >= 0 when provided.")

        incumbent_bound: Optional[int]
        if target_makespan is None:
            incumbent_bound = None
        else:
            incumbent_bound = int(target_makespan) + 1

        best_makespan: Optional[int] = None
        best_schedule: Optional[Dict[int, ScheduleEntry]] = None
        best_compact_node: Optional[CompactBBNode] = None
        nodes_expanded = 0
        nodes_pruned = 0
        nodes_expanded_after_incumbent = 0
        nodes_pruned_after_incumbent = 0
        first_incumbent_expanded: Optional[int] = None
        seen_incumbent = False
        # A memory-bounded run deliberately cannot provide a historical debug
        # tree.  Ignore debug in that mode even if a caller accidentally passes
        # debug=True, so the mode's memory contract is explicit.
        debug_info: Optional[DebugInfo] = (
            None if memory_bounded else (DebugInfo() if debug else None)
        )
        external_bound_matched = False

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

        def _stack_node(ref: object):
            """Resolve a live frontier reference in either storage mode."""
            if memory_bounded:
                return ref
            return self.nodes[int(ref)]

        def _compute_proof_burden() -> int:
            """Sum of (incumbent - node.lb) over open stack nodes with lb < incumbent."""
            if best_makespan is None:
                return 0
            total = 0
            for ref in stack:
                lb = _stack_node(ref).lower_bound
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
            for ref in stack:
                lb = _stack_node(ref).lower_bound
                if best is None or lb < best:
                    best = lb
            return best

        def _make_step_context(current_node=None) -> StepContext:
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

        def _wrapped_order_fn(node, incumbent: Optional[int]) -> List[int]:
            """Inject StepContext into order_ready_fn when it accepts it."""
            if order_ready_fn is None:
                from rcpsp_bb_rl.bnb.branching_order import order_by_activity_id
                return order_by_activity_id(node, incumbent)
            import inspect
            sig = inspect.signature(order_ready_fn)
            if len(sig.parameters) >= 3:
                return list(order_ready_fn(node, incumbent, _make_step_context(node)))
            return list(order_ready_fn(node, incumbent))

        # These aggregates are touched only when the debug runner opts in.
        diagnostic_totals: Counter = Counter({
            key: 0 for key in (
                "visited", "expanded", "lb_pruned", "dominance_pruned",
                "candidate_attempts", "infeasible_candidates", "bounded_children",
                "dead_ends", "solutions", "improvements",
            )
        }) if collect_checkpoints else Counter()
        previous_totals: Counter = diagnostic_totals.copy()
        recent_prefixes: Counter = Counter()
        previous_checkpoint_elapsed = 0.0
        last_expanded_node: Optional[BBNode] = None

        def _record_checkpoint(reason: str, termination: Optional[str] = None) -> None:
            nonlocal previous_totals, previous_checkpoint_elapsed
            assert debug_info is not None
            # Called between expansions: no currently popped node is omitted
            # from an unfinished subtree, and newly generated children are included.
            pending = [self.nodes[int(ref)] for ref in stack]
            internal_lbs = []
            for queued in pending:
                assert queued.internal_lower_bound is not None
                internal_lbs.append(queued.internal_lower_bound)
            effective_lbs = [queued.lower_bound for queued in pending]
            size = len(pending)
            elapsed = time.perf_counter() - start_time_monotonic
            interval_elapsed = elapsed - previous_checkpoint_elapsed
            interval = {key: diagnostic_totals[key] - previous_totals[key]
                        for key in diagnostic_totals}
            below = (sum(lb < best_makespan for lb in effective_lbs)
                     if best_makespan is not None else None)
            above = size - below if below is not None else None
            floor_count = (sum(lb <= external_lower_bound for lb in internal_lbs)
                           if external_lower_bound is not None else None)
            last_improvement = (debug_info.incumbent_history[-1]
                                if debug_info.incumbent_history else None)

            def pct(count: Optional[int], total: int) -> Optional[float]:
                return 100.0 * count / total if count is not None and total else None

            def groups(counts: Counter) -> List[DebugPrefixCount]:
                return [DebugPrefixCount(prefix, count) for prefix, count in counts.most_common()]

            checkpoint = DebugCheckpoint(
                reason=reason,
                termination=termination,
                nodes_expanded=nodes_expanded,
                elapsed_s=elapsed,
                incumbent=best_makespan,
                pruning_cutoff=incumbent_bound,
                external_lower_bound=external_lower_bound,
                nodes_since_improvement=(nodes_expanded - last_improvement.nodes_expanded
                                         if last_improvement is not None else None),
                seconds_since_improvement=(elapsed - last_improvement.elapsed_s
                                           if last_improvement is not None else None),
                incumbent_prefix=(last_improvement.prefix if last_improvement else None),
                incumbent_depth=(last_improvement.depth if last_improvement else None),
                frontier_size=size,
                internal_lb=_debug_bound_distribution(internal_lbs),
                effective_lb=_debug_bound_distribution(effective_lbs),
                below_incumbent=below,
                below_incumbent_pct=pct(below, size),
                at_or_above_incumbent=above,
                at_or_above_incumbent_pct=pct(above, size),
                internal_at_or_below_external=floor_count,
                internal_at_or_below_external_pct=pct(floor_count, size),
                proof_burden=(_compute_proof_burden() if best_makespan is not None else None),
                last_expanded_node_id=(last_expanded_node.node_id if last_expanded_node else None),
                last_expanded_depth=(last_expanded_node.depth if last_expanded_node else None),
                last_expanded_prefix=(last_expanded_node.debug_prefix if last_expanded_node else ()),
                next_pending_prefix=(pending[-1].debug_prefix if pending else None),
                recent_expansions_by_prefix=groups(recent_prefixes),
                frontier_by_first_branch=groups(Counter(n.debug_prefix[:1] for n in pending)),
                interval=interval,
                totals=dict(diagnostic_totals),
                interval_elapsed_s=interval_elapsed,
                nodes_per_second=(interval["expanded"] / interval_elapsed if interval_elapsed > 0 else None),
                lb_pruned_pct_of_visited=pct(interval["lb_pruned"], interval["visited"]),
                dominance_pruned_pct_of_bounded_children=pct(
                    interval["dominance_pruned"], interval["bounded_children"]),
            )
            debug_info.checkpoints.append(checkpoint)
            recent_prefixes.clear()
            previous_totals = diagnostic_totals.copy()
            previous_checkpoint_elapsed = elapsed
            if debug_checkpoint_callback is not None:
                debug_checkpoint_callback(checkpoint)

        while stack and ((max_nodes is None) or (nodes_expanded < max_nodes)) and not time_exceeded():
            if stop_on_first_solution and best_makespan is not None:
                break

            node_ref = stack.pop()
            stored_node = _stack_node(node_ref)
            node_id = stored_node.node_id
            if collect_checkpoints:
                diagnostic_totals["visited"] += 1
            compact_node: Optional[CompactBBNode] = None
            if memory_bounded:
                assert isinstance(stored_node, CompactBBNode)
                compact_node = stored_node
                dominance_engine.release_state_key(_compact_state_key(compact_node))
            else:
                assert isinstance(stored_node, BBNode)
                dominance_engine.release_state(
                    stored_node.unscheduled,
                    stored_node.scheduled,
                )

            incumbent = incumbent_bound
            if incumbent is not None and stored_node.lower_bound >= incumbent:
                if not memory_bounded:
                    stored_node.status = "pruned"
                nodes_pruned += 1
                _step_lb_pruned += 1
                if debug_info is not None:
                    debug_info.lb_pruned += 1
                if collect_checkpoints:
                    diagnostic_totals["lb_pruned"] += 1
                if seen_incumbent:
                    nodes_pruned_after_incumbent += 1
                continue

            node_is_complete = (
                compact_node.unscheduled_mask == 0
                if compact_node is not None
                else not stored_node.unscheduled
            )
            if node_is_complete:
                if not memory_bounded:
                    stored_node.status = "solution"
                makespan = (
                    compact_node.makespan
                    if compact_node is not None
                    else current_makespan(stored_node.scheduled)
                )
                if debug_info is not None:
                    debug_info.all_makespans.append(makespan)
                if collect_checkpoints:
                    diagnostic_totals["solutions"] += 1
                if incumbent_bound is None or makespan < incumbent_bound:
                    incumbent_bound = makespan
                    best_makespan = makespan
                    if compact_node is not None:
                        # Retain only the decision path. The full mapping is
                        # reconstructed once after the search terminates.
                        best_compact_node = compact_node
                    else:
                        best_schedule = stored_node.scheduled
                    if debug_info is not None:
                        debug_info.incumbent_history.append(IncumbentEvent(
                            rank=len(debug_info.incumbent_history) + 1,
                            makespan=makespan,
                            nodes_expanded=nodes_expanded,
                            depth=stored_node.depth,
                            elapsed_s=time.perf_counter() - start_time_monotonic,
                            prefix=stored_node.debug_prefix,
                        ))
                    if not seen_incumbent:
                        first_incumbent_expanded = nodes_expanded
                        seen_incumbent = True
                    if collect_checkpoints:
                        diagnostic_totals["improvements"] += 1
                        _record_checkpoint("incumbent")
                    # A feasible solution equal to a certified external lower
                    # bound closes the primal/dual gap. No remaining branch can
                    # improve it, so the optimality proof is complete without
                    # draining and individually pruning the open stack.
                    if (
                        external_lower_bound is not None
                        and makespan == external_lower_bound
                    ):
                        external_bound_matched = True
                        break
                continue

            node_has_ready = (
                compact_node.ready_mask != 0
                if compact_node is not None
                else bool(stored_node.ready)
            )
            if not node_has_ready:
                if collect_checkpoints:
                    diagnostic_totals["dead_ends"] += 1
                if not memory_bounded:
                    stored_node.status = "pruned"
                nodes_pruned += 1
                _step_lb_pruned += 1
                if seen_incumbent:
                    nodes_pruned_after_incumbent += 1
                continue

            if compact_node is not None:
                scheduled = _reconstruct_schedule(compact_node)
                node = ExpandedNodeView(
                    node_id=compact_node.node_id,
                    scheduled=scheduled,
                    ready=_mask_set(compact_node.ready_mask),
                    unscheduled=_mask_set(compact_node.unscheduled_mask),
                    lower_bound=compact_node.lower_bound,
                    parent_id=(
                        None if compact_node.parent is None else compact_node.parent.node_id
                    ),
                    action=(
                        None
                        if compact_node.selected_activity is None
                        else (
                            f"act {compact_node.selected_activity}"
                            f"@{compact_node.selected_start}"
                        )
                    ),
                    depth=compact_node.depth,
                    path_best_lb=compact_node.path_best_lb,
                    stagnation_depth=compact_node.stagnation_depth,
                )
            else:
                node = stored_node

            # Build the resource profile and earliest-feasible-start map ONCE,
            # before ordering. The ordering callback (e.g. the branching
            # policy) needs earliest-starts to featurise candidates, and the
            # child loop below needs them to place activities. Computing them
            # here and attaching to the node lets both share one computation
            # instead of each recomputing profile + earliest_feasible_start.
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

            if not memory_bounded:
                node.status = "expanded"
            nodes_expanded += 1
            if collect_checkpoints:
                diagnostic_totals["expanded"] += 1
                recent_prefixes[node.debug_prefix] += 1
                last_expanded_node = node
            if seen_incumbent:
                nodes_expanded_after_incumbent += 1

            # Reverse push for DFS/LIFO.
            for reversed_rank, act_id in enumerate(reversed(acts)):
                if collect_checkpoints:
                    diagnostic_totals["candidate_attempts"] += 1
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
                    if collect_checkpoints:
                        diagnostic_totals["infeasible_candidates"] += 1
                    continue

                duration = self.instance.activities[act_id].duration
                finish = est_start + duration

                child_entry = ScheduleEntry(
                    start=est_start,
                    finish=finish,
                    duration=duration,
                )
                if compact_node is not None:
                    activity_bit = 1 << activity_to_bit[act_id]
                    child_scheduled_mask = compact_node.scheduled_mask | activity_bit
                    child_unscheduled_mask = compact_node.unscheduled_mask & ~activity_bit
                    child_ready_mask = _ready_mask(
                        child_unscheduled_mask,
                        child_scheduled_mask,
                    )
                    child_scheduled = ScheduleOverlay(
                        node.scheduled,
                        act_id,
                        child_entry,
                    )
                    child_unscheduled = _mask_set(child_unscheduled_mask)
                else:
                    child_scheduled = dict(node.scheduled)
                    child_scheduled[act_id] = child_entry
                    child_unscheduled = set(node.unscheduled)
                    child_unscheduled.discard(act_id)
                    child_ready = compute_ready_set(
                        child_unscheduled,
                        set(child_scheduled.keys()),
                        self.predecessors,
                    )

                child_internal_lb = _internal_lower_bound(
                    child_unscheduled,
                    child_scheduled,
                )
                child_lb = _effective_lower_bound(child_internal_lb)
                if collect_checkpoints:
                    diagnostic_totals["bounded_children"] += 1

                compact_child_key = None
                if compact_node is not None:
                    child_start_times_code = (
                        compact_node.start_times_code
                        | (int(est_start) << (activity_to_bit[act_id] * start_time_bits))
                    )
                    compact_child_key = child_scheduled_mask, child_start_times_code

                pruned_rule = dominance_engine.prune_child(
                    parent_scheduled=node.scheduled,
                    child_scheduled=child_scheduled,
                    child_unscheduled=child_unscheduled,
                    child_lb=child_lb,
                    act_id=act_id,
                    child_start=est_start,
                    state_key=compact_child_key,
                )
                if pruned_rule is not None:
                    if collect_checkpoints:
                        diagnostic_totals["dominance_pruned"] += 1
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
                if compact_node is not None:
                    child_node = CompactBBNode(
                        node_id=child_id,
                        parent=compact_node,
                        selected_activity=act_id,
                        selected_start=est_start,
                        scheduled_mask=child_scheduled_mask,
                        unscheduled_mask=child_unscheduled_mask,
                        ready_mask=child_ready_mask,
                        start_times_code=child_start_times_code,
                        lower_bound=child_lb,
                        depth=node.depth + 1,
                        makespan=max(compact_node.makespan, finish),
                        path_best_lb=child_path_best_lb,
                        stagnation_depth=child_stagnation_depth,
                    )
                    stack.append(child_node)
                else:
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
                    if collect_checkpoints:
                        child_node.internal_lower_bound = child_internal_lb
                        prefix = node.debug_prefix
                        if len(acts) > 1 and len(prefix) < debug_prefix_depth:
                            prefix += (DebugBranchChoice(
                                decision_node_id=node_id,
                                activity=act_id,
                                start=est_start,
                                rank=len(acts) - reversed_rank,
                                candidate_count=len(acts),
                            ),)
                        child_node.debug_prefix = prefix
                    self.edges.append((node_id, child_id))
                    stack.append(child_id)

            # The feasibility cache has served both consumers (ordering callback
            # and the child loop above); free it so long runs don't retain one
            # dict per expanded node in self.nodes.
            node.est_map = None
            if collect_checkpoints and nodes_expanded % debug_checkpoint_nodes == 0:
                _record_checkpoint("periodic")

        if external_bound_matched:
            solver_done_reason = "external_bound_matched"
        elif stack and time_exceeded():
            solver_done_reason = "time_limit"
        elif stack and max_nodes is not None and nodes_expanded >= max_nodes:
            solver_done_reason = "node_limit"
        else:
            solver_done_reason = "search_exhausted"
        final_proof_burden = _compute_proof_burden()
        # At termination the frontier is exactly the remaining stack (no node is
        # being expanded). On an exhausted search the stack is empty -> None,
        # which the env treats as a fully closed gap (proof complete).
        final_frontier_min_lb = _compute_frontier_min_lb(None)
        if collect_checkpoints:
            _record_checkpoint("termination", solver_done_reason)

        if best_compact_node is not None:
            best_schedule = _reconstruct_schedule(best_compact_node)

        return SolverResult(
            best_makespan=best_makespan,
            best_schedule=best_schedule,
            nodes=[] if memory_bounded else self.nodes,
            edges=[] if memory_bounded else self.edges,
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
    external_lower_bound: Optional[int] = None,
    memory_bounded: bool = False,
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
        external_lower_bound=external_lower_bound,
        dominance=dominance,
        target_makespan=target_makespan,
        stop_on_first_solution=stop_on_first_solution,
        debug=debug,
        memory_bounded=memory_bounded,
    )
