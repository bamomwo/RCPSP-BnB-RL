from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, Iterable, Mapping, MutableMapping, Optional, Sequence, Set, Tuple

from rcpsp_bb_rl.bnb.scheduling import build_profile, earliest_feasible_start, entry_finish, entry_start

if TYPE_CHECKING:
    from rcpsp_bb_rl.data.parsing import RCPSPInstance

DominanceRuleId = str
SerialStateSignature = Tuple[frozenset[int], Tuple[Tuple[int, int, int], ...]]
ParallelCutsetSignature = Tuple[int, frozenset[int], Tuple[Tuple[int, int, int], ...]]

RULE_SET_BASED = "set_based"
RULE_CONTRADICTION = "contradiction"
RULE_EXTENDED_GLOBAL_SHIFT = "extended_global_shift"
RULE_PAR_CUTSET = "par_cutset"
RULE_PAR_LEFT_SHIFT = "par_left_shift"
SERIAL_RULE_IDS = (
    RULE_SET_BASED,
    RULE_CONTRADICTION,
    RULE_EXTENDED_GLOBAL_SHIFT,
)
PARALLEL_RULE_IDS = (
    RULE_PAR_CUTSET,
    RULE_PAR_LEFT_SHIFT,
)
ALL_RULE_IDS = (
    *SERIAL_RULE_IDS,
    *PARALLEL_RULE_IDS,
)


@dataclass(frozen=True)
class DominanceConfig:
    enabled: bool = False
    rules: Tuple[DominanceRuleId, ...] = ()


@dataclass
class DominanceStats:
    pruned_children: int = 0
    pruned_by_rule: Dict[DominanceRuleId, int] = field(default_factory=dict)

    def record_prune(self, rule_id: DominanceRuleId) -> None:
        self.pruned_children += 1
        self.pruned_by_rule[rule_id] = self.pruned_by_rule.get(rule_id, 0) + 1


def normalize_dominance_spec(spec: object = False) -> DominanceConfig:
    """
    Normalize user dominance spec into a validated config.

    Accepted forms:
    - False / None / "off" / "none" / "0"
    - True / "on" / "1" / "serial"
    - "parallel"
    - "all"
    - "set_based,contradiction"
    - ["set_based", "extended_global_shift"]
    """
    if isinstance(spec, DominanceConfig):
        return spec

    if spec is None:
        return DominanceConfig(enabled=False, rules=())

    if isinstance(spec, bool):
        return DominanceConfig(enabled=spec, rules=SERIAL_RULE_IDS if spec else ())

    if isinstance(spec, str):
        raw = spec.strip().lower()
        if raw in {"", "off", "none", "false", "0", "no"}:
            return DominanceConfig(enabled=False, rules=())
        if raw in {"on", "true", "1", "yes", "serial"}:
            return DominanceConfig(enabled=True, rules=SERIAL_RULE_IDS)
        if raw == "parallel":
            return DominanceConfig(enabled=True, rules=PARALLEL_RULE_IDS)
        if raw == "all":
            return DominanceConfig(enabled=True, rules=ALL_RULE_IDS)
        parts = tuple(part.strip().lower() for part in raw.split(",") if part.strip())
        if not parts:
            return DominanceConfig(enabled=False, rules=())
        _validate_rules(parts)
        return DominanceConfig(enabled=True, rules=parts)

    if isinstance(spec, Sequence):
        parts = tuple(str(item).strip().lower() for item in spec if str(item).strip())
        if not parts:
            return DominanceConfig(enabled=False, rules=())
        if len(parts) == 1 and parts[0] == "serial":
            return DominanceConfig(enabled=True, rules=SERIAL_RULE_IDS)
        if len(parts) == 1 and parts[0] == "parallel":
            return DominanceConfig(enabled=True, rules=PARALLEL_RULE_IDS)
        if len(parts) == 1 and parts[0] == "all":
            return DominanceConfig(enabled=True, rules=ALL_RULE_IDS)
        _validate_rules(parts)
        return DominanceConfig(enabled=True, rules=parts)

    raise TypeError("dominance spec must be bool, string, list/tuple of strings, or None.")


def format_dominance_spec(spec: object = False) -> str:
    cfg = normalize_dominance_spec(spec)
    if not cfg.enabled:
        return "off"
    if cfg.rules == SERIAL_RULE_IDS:
        return "serial"
    if cfg.rules == PARALLEL_RULE_IDS:
        return "parallel"
    if cfg.rules == ALL_RULE_IDS:
        return "all"
    return ",".join(cfg.rules)


def _validate_rules(rules: Iterable[str]) -> None:
    unknown = [rule for rule in rules if rule not in ALL_RULE_IDS]
    if unknown:
        valid = ", ".join(ALL_RULE_IDS)
        raise ValueError(f"Unknown dominance rule(s): {', '.join(unknown)}. Available: {valid}")


def _schedule_signature(
    unscheduled: Set[int],
    scheduled: Mapping[int, object],
) -> SerialStateSignature:
    """
    Canonical signature for duplicate-state elimination.

    This does not approximate: it keys on exact scheduled starts/finishes and
    exact unscheduled set.
    """
    entries = tuple(
        sorted(
            (int(act_id), int(entry_start(entry)), int(entry_finish(entry)))
            for act_id, entry in scheduled.items()
        )
    )
    return frozenset(int(a) for a in unscheduled), entries


def _parallel_cutset_signature(
    unscheduled: Set[int],
    scheduled: Mapping[int, object],
    current_time: int,
) -> ParallelCutsetSignature:
    t = int(current_time)
    entries = tuple(
        sorted(
            (int(act_id), int(entry_start(entry)), int(entry_finish(entry)))
            for act_id, entry in scheduled.items()
        )
    )
    return int(t), frozenset(int(a) for a in unscheduled), entries


class DominanceEngine:
    """
    Modular dominance-rule dispatcher for B&B child pruning.

    Rules are intentionally conservative in this first implementation:
    - set_based: exact duplicate-state elimination via canonical signatures.
    - contradiction: prune explicit precedence contradictions in scheduled edges.
    - extended_global_shift: prune if a child is not globally left-shifted
      relative to the parent's schedule.
    """

    def __init__(
        self,
        instance: RCPSPInstance,
        predecessors: Mapping[int, Set[int]],
        config: DominanceConfig,
    ) -> None:
        self.instance = instance
        self.predecessors = predecessors
        self.config = config
        self.stats = DominanceStats()
        self._best_lb_by_signature: MutableMapping[SerialStateSignature, int] = {}
        self._seen_cutset_signatures: Set[ParallelCutsetSignature] = set()

    def register_state(
        self,
        unscheduled: Set[int],
        scheduled: Mapping[int, object],
        lower_bound: int,
        current_time: int = 0,
    ) -> None:
        if not self.config.enabled:
            return

        if RULE_SET_BASED in self.config.rules:
            signature = _schedule_signature(unscheduled, scheduled)
            prev = self._best_lb_by_signature.get(signature)
            if prev is None or int(lower_bound) < prev:
                self._best_lb_by_signature[signature] = int(lower_bound)

        if RULE_PAR_CUTSET in self.config.rules:
            signature = _parallel_cutset_signature(
                unscheduled,
                scheduled,
                int(current_time),
            )
            self._seen_cutset_signatures.add(signature)

    def prune_child(
        self,
        *,
        parent_scheduled: Mapping[int, object],
        child_scheduled: Mapping[int, object],
        child_unscheduled: Set[int],
        child_lb: int,
        act_id: int,
        child_start: int,
        parent_time: int = 0,
        child_time: int = 0,
    ) -> Optional[DominanceRuleId]:
        if not self.config.enabled:
            return None

        for rule_id in self.config.rules:
            if rule_id == RULE_SET_BASED:
                if self._set_based_dominated(child_unscheduled, child_scheduled, child_lb):
                    self.stats.record_prune(RULE_SET_BASED)
                    return RULE_SET_BASED
            elif rule_id == RULE_CONTRADICTION:
                if self._contradiction_dominated(child_scheduled, act_id):
                    self.stats.record_prune(RULE_CONTRADICTION)
                    return RULE_CONTRADICTION
            elif rule_id == RULE_EXTENDED_GLOBAL_SHIFT:
                if self._extended_global_shift_dominated(parent_scheduled, act_id, child_start):
                    self.stats.record_prune(RULE_EXTENDED_GLOBAL_SHIFT)
                    return RULE_EXTENDED_GLOBAL_SHIFT
            elif rule_id == RULE_PAR_CUTSET:
                if self._par_cutset_dominated(
                    child_unscheduled,
                    child_scheduled,
                    child_time,
                ):
                    self.stats.record_prune(RULE_PAR_CUTSET)
                    return RULE_PAR_CUTSET
            elif rule_id == RULE_PAR_LEFT_SHIFT:
                if self._par_left_shift_dominated(
                    parent_scheduled,
                    act_id,
                    child_start,
                    parent_time,
                ):
                    self.stats.record_prune(RULE_PAR_LEFT_SHIFT)
                    return RULE_PAR_LEFT_SHIFT

        self.register_state(
            child_unscheduled,
            child_scheduled,
            child_lb,
            current_time=child_time,
        )
        return None

    def _set_based_dominated(
        self,
        child_unscheduled: Set[int],
        child_scheduled: Mapping[int, object],
        child_lb: int,
    ) -> bool:
        signature = _schedule_signature(child_unscheduled, child_scheduled)
        prev_best = self._best_lb_by_signature.get(signature)
        if prev_best is None:
            return False
        return int(child_lb) >= int(prev_best)

    def _contradiction_dominated(
        self,
        child_scheduled: Mapping[int, object],
        act_id: int,
    ) -> bool:
        """
        Incremental precedence-consistency check for the newly scheduled activity.
        """
        if act_id not in child_scheduled:
            return True

        new_start = entry_start(child_scheduled[act_id])
        new_finish = entry_finish(child_scheduled[act_id])

        for pred in self.predecessors.get(act_id, set()):
            if pred not in child_scheduled:
                continue
            if entry_finish(child_scheduled[pred]) > new_start:
                return True

        for succ in self.instance.activities[act_id].successors:
            if succ not in child_scheduled:
                continue
            if new_finish > entry_start(child_scheduled[succ]):
                return True

        return False

    def _extended_global_shift_dominated(
        self,
        parent_scheduled: Mapping[int, object],
        act_id: int,
        child_start: int,
    ) -> bool:
        horizon_hint = sum(act.duration for act in self.instance.activities.values())
        parent_profile = build_profile(
            self.instance.activities,
            self.instance.resource_caps,
            parent_scheduled,
            horizon=horizon_hint,
        )
        est = earliest_feasible_start(
            instance=self.instance,
            predecessors=self.predecessors,
            scheduled=parent_scheduled,
            act_id=act_id,
            incumbent=None,
            profile=parent_profile,
        )
        if est is None:
            return True
        return int(child_start) > int(est)

    def _par_cutset_dominated(
        self,
        child_unscheduled: Set[int],
        child_scheduled: Mapping[int, object],
        child_time: int,
    ) -> bool:
        signature = _parallel_cutset_signature(
            child_unscheduled,
            child_scheduled,
            int(child_time),
        )
        return signature in self._seen_cutset_signatures

    def _par_left_shift_dominated(
        self,
        parent_scheduled: Mapping[int, object],
        act_id: int,
        child_start: int,
        parent_time: int,
    ) -> bool:
        if int(child_start) < int(parent_time):
            return True
        horizon_hint = sum(act.duration for act in self.instance.activities.values())
        parent_profile = build_profile(
            self.instance.activities,
            self.instance.resource_caps,
            parent_scheduled,
            horizon=horizon_hint,
        )
        est = earliest_feasible_start(
            instance=self.instance,
            predecessors=self.predecessors,
            scheduled=parent_scheduled,
            act_id=act_id,
            incumbent=None,
            profile=parent_profile,
        )
        if est is None:
            return True
        return int(child_start) > int(est)


def build_dominance_engine(
    *,
    instance: RCPSPInstance,
    predecessors: Mapping[int, Set[int]],
    dominance: object = False,
) -> DominanceEngine:
    cfg = normalize_dominance_spec(dominance)
    return DominanceEngine(
        instance=instance,
        predecessors=predecessors,
        config=cfg,
    )
