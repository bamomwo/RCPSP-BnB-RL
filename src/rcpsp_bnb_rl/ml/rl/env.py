"""Single-dive scheduling environment and greedy sibling rollout collection."""
from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Dict, Optional

import torch

from rcpsp_bnb_rl.bnb.lower_bounds import DEFAULT_LOWER_BOUND_ID, lower_bound
from rcpsp_bnb_rl.bnb.precedence import compute_ready_set
from rcpsp_bnb_rl.bnb.scheduling import ResourceProfile, earliest_feasible_start
from rcpsp_bnb_rl.data.parsing import RCPSPInstance
from rcpsp_bnb_rl.ml.il.featurize import InstanceStatics, build_instance_statics
from rcpsp_bnb_rl.ml.observations import batch_observations, build_actor_observation


@dataclass(frozen=True)
class DiveScheduleEntry:
    start: int
    finish: int
    duration: int


def _real_activity_ids(instance: RCPSPInstance, statics: InstanceStatics) -> frozenset[int]:
    """Exclude unambiguous zero-demand dummy endpoints in the validated DAG."""
    ids = set(instance.activities)
    sources = [aid for aid in ids if not statics.predecessors[aid]]
    sinks = [aid for aid in ids if not statics.successors[aid]]
    for endpoints in (sources, sinks):
        if len(endpoints) == 1:
            aid = endpoints[0]
            activity = instance.activities[aid]
            if activity.duration == 0 and not any(activity.resources):
                ids.discard(aid)
    return frozenset(ids)


class DiveState:
    """Cloneable serial SGS state with no search stack or pruning history."""

    def __init__(
        self, instance: RCPSPInstance, statics: Optional[InstanceStatics] = None,
    ) -> None:
        if not instance.activities:
            raise ValueError("A dive needs a nonempty RCPSP instance")
        for activity in instance.activities.values():
            if activity.duration < 0 or len(activity.resources) != instance.num_resources:
                raise ValueError("Invalid activity duration or resource width")
            if any(req < 0 or req > cap for req, cap in zip(activity.resources, instance.resource_caps)):
                raise ValueError("Activity demand exceeds resource capacity")
        self.instance = instance
        self.statics = statics if statics is not None else build_instance_statics(instance)
        self.real_activity_ids = _real_activity_ids(instance, self.statics)
        self.scheduled: Dict[int, DiveScheduleEntry] = {}
        self.unscheduled = set(instance.activities)
        self.ready = compute_ready_set(self.unscheduled, set(), self.statics.predecessors)
        self.profile = ResourceProfile(
            instance.num_resources, sum(a.duration for a in instance.activities.values()),
        )
        self._makespan = 0

    @property
    def depth(self) -> int:
        return len(self.scheduled)

    @property
    def done(self) -> bool:
        return not self.unscheduled

    @property
    def makespan(self) -> int:
        return self._makespan

    @property
    def completion_ratio(self) -> float:
        return len(self.real_activity_ids.intersection(self.scheduled)) / max(1, len(self.real_activity_ids))

    def clone(self) -> "DiveState":
        clone = object.__new__(DiveState)
        clone.instance = self.instance
        clone.statics = self.statics
        clone.real_activity_ids = self.real_activity_ids
        clone.scheduled = dict(self.scheduled)
        clone.unscheduled = set(self.unscheduled)
        clone.ready = set(self.ready)
        clone.profile = self.profile.copy()
        clone._makespan = self._makespan
        return clone

    def observation(self, lb_spec: object = DEFAULT_LOWER_BOUND_ID) -> Dict[str, torch.Tensor]:
        if self.done:
            raise ValueError("A completed dive has no actor observation")
        return build_actor_observation(
            self.instance, self.scheduled, self.unscheduled, self.ready, self.depth,
            statics=self.statics, profile=self.profile, lb_spec=lb_spec,
        )

    def step(self, activity_id: int) -> None:
        if activity_id not in self.ready:
            raise ValueError(f"Activity {activity_id} is not ready")
        start = earliest_feasible_start(
            self.instance, self.statics.predecessors, self.scheduled, activity_id,
            incumbent=None, profile=self.profile,
        )
        if start is None:
            raise ValueError(f"Activity {activity_id} has no feasible serial placement")
        activity = self.instance.activities[activity_id]
        finish = start + activity.duration
        self.scheduled[activity_id] = DiveScheduleEntry(start, finish, activity.duration)
        self.unscheduled.remove(activity_id)
        self.profile.add_activity(activity.resources, start, finish)
        self.ready = compute_ready_set(
            self.unscheduled, set(self.scheduled), self.statics.predecessors,
        )
        self._makespan = max(self._makespan, finish)


@dataclass
class ForkRecord:
    observation: Dict[str, torch.Tensor]
    old_logits: torch.Tensor
    makespans: torch.Tensor
    depth: int
    completion_ratio: float
    selected_index: int


@dataclass
class ForkEpisode:
    records: list[ForkRecord]
    diagnostics: dict
    stem_schedule: Dict[int, DiveScheduleEntry]
    stem_makespan: int
    root_lower_bound: int


def _select_fork_indices(completion_ratios: list[float], budget: int | None) -> list[int]:
    if budget is None or len(completion_ratios) <= budget:
        return list(range(len(completion_ratios)))
    if budget == 1:
        return [min(range(len(completion_ratios)), key=lambda i: abs(completion_ratios[i] - 0.5))]
    selected = []
    for slot in range(budget):
        target = completion_ratios[0] + (completion_ratios[-1] - completion_ratios[0]) * slot / (budget - 1)
        first = selected[-1] + 1 if selected else 0
        last = len(completion_ratios) - (budget - slot)
        selected.append(min(range(first, last + 1), key=lambda i: abs(completion_ratios[i] - target)))
    return selected


def collect_fork_episode(
    actor: torch.nn.Module,
    instance: RCPSPInstance,
    device: torch.device | str,
    lb_spec: object = DEFAULT_LOWER_BOUND_ID,
    *,
    max_forks_per_dive: int | None = 32,
    max_active_forks: int = 128,
    inference_chunk_size: int = 128,
) -> ForkEpisode:
    """Record a greedy stem and complete every sibling at selected fork states.

    All fork states share the collection actor without intervening updates. The
    stem outcome is reused for its chosen child because its suffix is greedy too.
    A null fork budget evaluates every split; finite budgets cover the stem depth.
    """
    for name, value in (("max_forks_per_dive", max_forks_per_dive),
                        ("max_active_forks", max_active_forks),
                        ("inference_chunk_size", inference_chunk_size)):
        if value is None and name == "max_forks_per_dive":
            continue
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be a positive integer" + (" or null" if name == "max_forks_per_dive" else ""))
    device = torch.device(device)
    start_time = time.perf_counter()
    stem = DiveState(instance)
    root_lb = lower_bound(instance, stem.unscheduled, {}, lb_id=lb_spec, incumbent=None)
    records: list[ForkRecord] = []
    snapshots: list[DiveState | None] = []
    # Small projects can retain every split immediately; larger ones replay only
    # the selected prefixes so full snapshots never exceed the fork budget.
    save_all = max_forks_per_dive is None or max_forks_per_dive >= len(instance.activities) - 1
    stem_decisions: list[tuple[int, float, torch.Tensor, int]] = []
    action_trace: list[int] = []
    batch_sizes: list[int] = []
    stem_forward_calls = 0
    env_steps = 0
    replay_steps = 0
    peak_active_forks = 0
    was_training = actor.training
    actor.eval()
    try:
        with torch.no_grad():
            while not stem.done:
                if not stem.ready:
                    raise ValueError("An unfinished dive has no ready activities")
                if len(stem.ready) == 1:
                    aid = next(iter(stem.ready))
                else:
                    observation = stem.observation(lb_spec)
                    if int(observation["action_mask"].sum()) < 2:
                        raise ValueError("A valid serial SGS split must have at least two feasible actions")
                    logits = actor.forward_batch(**batch_observations([observation], device))[0]
                    selected = int(logits.argmax().item())
                    old_logits = logits.detach().cpu().clone()
                    if save_all:
                        snapshots.append(stem.clone())
                        records.append(ForkRecord(
                            observation=observation, old_logits=old_logits,
                            makespans=torch.full((len(observation["ready_ids"]),), -1, dtype=torch.long),
                            depth=stem.depth, completion_ratio=stem.completion_ratio,
                            selected_index=selected,
                        ))
                    else:
                        stem_decisions.append((stem.depth, stem.completion_ratio, old_logits, selected))
                    aid = int(observation["ready_ids"][selected])
                    stem_forward_calls += 1
                action_trace.append(aid)
                stem.step(aid)
                env_steps += 1

            stem_wall_s = time.perf_counter() - start_time
            n_eligible_fork_states = len(records) if save_all else len(stem_decisions)
            if not save_all and stem_decisions:
                indices = _select_fork_indices([decision[1] for decision in stem_decisions], max_forks_per_dive)
                selected_decisions = {stem_decisions[i][0]: stem_decisions[i] for i in indices}
                last_depth = max(selected_decisions)
                replay = DiveState(instance, statics=stem.statics)
                for depth, aid in enumerate(action_trace):
                    if depth in selected_decisions:
                        _, ratio, old_logits, selected = selected_decisions[depth]
                        observation = replay.observation(lb_spec)
                        records.append(ForkRecord(
                            observation=observation, old_logits=old_logits,
                            makespans=torch.full((len(observation["ready_ids"]),), -1, dtype=torch.long),
                            depth=depth, completion_ratio=ratio, selected_index=selected,
                        ))
                        snapshots.append(replay.clone())
                    if depth == last_depth:
                        break
                    replay.step(aid)
                    replay_steps += 1
                    env_steps += 1
                del replay

            for record in records:
                record.makespans[record.selected_index] = stem.makespan
            pending_counts = [len(record.makespans) - 1 for record in records]
            jobs = iter((row, slot, aid)
                        for row, record in enumerate(records)
                        for slot, aid in enumerate(record.observation["ready_ids"].tolist())
                        if slot != record.selected_index)
            pending = next(jobs, None)
            pool: list[tuple[DiveState, int, int]] = []
            while pool or pending is not None:
                while len(pool) < max_active_forks and pending is not None:
                    row, slot, aid = pending
                    snapshot = snapshots[row]
                    assert snapshot is not None
                    fork = snapshot.clone()
                    del snapshot
                    pending_counts[row] -= 1
                    if pending_counts[row] == 0:
                        snapshots[row] = None
                    peak_active_forks = max(peak_active_forks, len(pool) + 1)
                    fork.step(aid)
                    env_steps += 1
                    if fork.done:
                        records[row].makespans[slot] = fork.makespan
                    else:
                        pool.append((fork, row, slot))
                    del fork
                    pending = next(jobs, None)
                split_indices = [k for k, (state, _, _) in enumerate(pool) if len(state.ready) > 1]
                choices: dict[int, int] = {}
                for start in range(0, len(split_indices), inference_chunk_size):
                    indices = split_indices[start:start + inference_chunk_size]
                    observations = [pool[k][0].observation(lb_spec) for k in indices]
                    logits = actor.forward_batch(**batch_observations(observations, device))
                    selected = logits.argmax(dim=1).cpu().tolist()
                    choices.update({index: int(observation["ready_ids"][slot])
                                    for index, observation, slot in zip(indices, observations, selected)})
                    batch_sizes.append(len(observations))
                    del observations, logits
                remaining = []
                for k, (state, row, slot) in enumerate(pool):
                    if not state.ready:
                        raise ValueError("An unfinished fork has no ready activities")
                    aid = choices[k] if k in choices else next(iter(state.ready))
                    state.step(aid)
                    env_steps += 1
                    if state.done:
                        records[row].makespans[slot] = state.makespan
                    else:
                        remaining.append((state, row, slot))
                if pool:
                    del state
                pool = remaining
    finally:
        actor.train(was_training)

    wall_s = time.perf_counter() - start_time
    n_pairs = sum(len(record.makespans) * (len(record.makespans) - 1) // 2 for record in records)
    n_non_tied_pairs = 0
    depth_counts = {band: {"states": 0, "pairs": 0, "non_tied_pairs": 0} for band in ("shallow", "mid", "deep")}
    for record in records:
        costs = record.makespans.tolist()
        pairs = len(costs) * (len(costs) - 1) // 2
        non_tied = sum(costs[i] != costs[j] for i in range(len(costs)) for j in range(i + 1, len(costs)))
        n_non_tied_pairs += non_tied
        band = ("shallow", "mid", "deep")[min(2, int(3 * record.completion_ratio))]
        depth_counts[band]["states"] += 1
        depth_counts[band]["pairs"] += pairs
        depth_counts[band]["non_tied_pairs"] += non_tied
    n_non_tied_groups = sum(bool((record.makespans != record.makespans[0]).any()) for record in records)
    n_beat_states = sum(bool((record.makespans < stem.makespan).any()) for record in records)
    n_fork_dives = sum(len(record.makespans) - 1 for record in records)
    diagnostics = {
        "n_real_activities": len(stem.real_activity_ids),
        "n_eligible_fork_states": n_eligible_fork_states,
        "fork_selection": "all" if len(records) == n_eligible_fork_states else "stratified",
        "n_saved_snapshots": len(records),
        "peak_active_forks": peak_active_forks,
        "replay_steps": replay_steps,
        "n_fork_states": len(records),
        "n_fork_dives": n_fork_dives,
        "n_sibling_pairs": n_pairs,
        "n_non_tied_pairs": n_non_tied_pairs,
        "n_non_tied_groups": n_non_tied_groups,
        "n_beat_states": n_beat_states,
        "signal_density": n_non_tied_groups / max(1, len(records)),
        "environment_steps": env_steps,
        "stem_forward_calls": stem_forward_calls,
        "phase2_forward_calls": len(batch_sizes),
        "phase2_batch_sizes": batch_sizes,
        "mean_phase2_batch_size": sum(batch_sizes) / max(1, len(batch_sizes)),
        "max_phase2_batch_size": max(batch_sizes, default=0),
        "stem_makespan": stem.makespan,
        "best_makespan": min((int(record.makespans.min()) for record in records), default=stem.makespan),
        "root_lower_bound": root_lb,
        "stem_gap_to_root_bound": (stem.makespan - root_lb) / max(1, root_lb),
        "stem_wall_s": stem_wall_s,
        "fork_wall_s": wall_s - stem_wall_s,
        "wall_s": wall_s,
        "dives_per_s": (1 + n_fork_dives) / max(wall_s, 1e-9),
        "depth_counts": depth_counts,
    }
    return ForkEpisode(records, diagnostics, dict(stem.scheduled), stem.makespan, root_lb)
