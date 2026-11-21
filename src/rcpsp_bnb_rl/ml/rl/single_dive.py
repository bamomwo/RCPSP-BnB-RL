"""Greedy serial schedules and direct sibling counterfactuals for training."""
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
            resource_encoder="set",
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


def collect_fork_episode(
    actor: torch.nn.Module,
    instance: RCPSPInstance,
    device: torch.device | str,
    lb_spec: object = DEFAULT_LOWER_BOUND_ID,
) -> ForkEpisode:
    """Record one greedy stem, then greedily complete every alternative sibling.

    All fork states share the collection actor without intervening updates. The
    stem outcome is reused for its chosen child because its suffix is greedy too.
    """
    device = torch.device(device)
    start_time = time.perf_counter()
    stem = DiveState(instance)
    root_lb = lower_bound(instance, stem.unscheduled, {}, lb_id=lb_spec, incumbent=None)
    records: list[ForkRecord] = []
    snapshots: list[DiveState] = []
    batch_sizes: list[int] = []
    stem_forward_calls = 0
    env_steps = 0
    was_training = actor.training
    actor.eval()
    try:
        with torch.no_grad():
            while not stem.done:
                if not stem.ready:
                    raise ValueError("An unfinished dive has no ready activities")
                if len(stem.ready) == 1:
                    stem.step(next(iter(stem.ready)))
                else:
                    observation = stem.observation(lb_spec)
                    if int(observation["action_mask"].sum()) < 2:
                        raise ValueError("A valid serial SGS split must have at least two feasible actions")
                    logits = actor.forward_batch(**batch_observations([observation], device))[0][0]
                    selected = int(logits.argmax().item())
                    snapshots.append(stem.clone())
                    records.append(ForkRecord(
                        observation=observation,
                        old_logits=logits.detach().cpu().clone(),
                        makespans=torch.full((len(observation["ready_ids"]),), -1, dtype=torch.long),
                        depth=stem.depth,
                        completion_ratio=stem.completion_ratio,
                        selected_index=selected,
                    ))
                    stem.step(int(observation["ready_ids"][selected]))
                    stem_forward_calls += 1
                env_steps += 1

            stem_wall_s = time.perf_counter() - start_time
            pool: list[tuple[DiveState, int, int]] = []
            for row, (snapshot, record) in enumerate(zip(snapshots, records)):
                record.makespans[record.selected_index] = stem.makespan
                for slot, aid in enumerate(record.observation["ready_ids"].tolist()):
                    if slot == record.selected_index:
                        continue
                    fork = snapshot.clone()
                    fork.step(aid)
                    env_steps += 1
                    if fork.done:
                        record.makespans[slot] = fork.makespan
                    else:
                        pool.append((fork, row, slot))

            while pool:
                split_work = [(k, state.observation(lb_spec)) for k, (state, _, _) in enumerate(pool)
                              if len(state.ready) > 1]
                choices: dict[int, int] = {}
                if split_work:
                    observations = [observation for _, observation in split_work]
                    logits = actor.forward_batch(**batch_observations(observations, device))[0]
                    selected = logits.argmax(dim=1).cpu().tolist()
                    choices = {
                        index: int(observation["ready_ids"][slot])
                        for (index, observation), slot in zip(split_work, selected)
                    }
                    batch_sizes.append(len(observations))
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
