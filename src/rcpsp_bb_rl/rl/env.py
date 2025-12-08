from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import torch

from rcpsp_bb_rl.bnb.core import (
    BBNode,
    ScheduleEntry,
    build_predecessors,
    compute_ready_set,
    current_makespan,
    earliest_feasible_start,
    lower_bound,
)
from rcpsp_bb_rl.bnb.policy_guidance import _build_record_for_node  # re-use feature construction
from rcpsp_bb_rl.data.featurize import candidate_features, global_features
from rcpsp_bb_rl.data.parsing import RCPSPInstance, load_instance
from rcpsp_bb_rl.data.trajectory_dataset import TrajectoryRecord


@dataclass
class StepOutput:
    """Container for env.step results."""

    observation: Dict[str, torch.Tensor]
    reward: float
    done: bool
    info: Dict


class BranchingEnv:
    """
    Lightweight environment wrapper to step through RCPSP B&B decisions.

    Each step selects one ready activity to schedule next. Observations mirror
    the imitation-learning featurization so PPO (or similar) can be warm-started
    from the behavior-cloned policy.
    """

    def __init__(
        self,
        instance_source: RCPSPInstance | Path | str,
        max_resources: int = 4,
        step_cost: float = 1.0,
        terminal_makespan_coeff: float = 0.0,
        max_steps: Optional[int] = None,
    ) -> None:
        """
        Args:
            instance_source: RCPSPInstance or path to load for rollouts.
            max_resources: Pad/truncate resource dims for features.
            step_cost: Reward penalty applied each expansion (negative encourages shallow trees).
            terminal_makespan_coeff: Optional coefficient to add terminal reward of
                -terminal_makespan_coeff * makespan when the schedule completes.
            max_steps: Optional hard cap on steps (episode ends if exceeded).
        """
        self.instance_source = instance_source
        self.max_resources = max_resources
        self.step_cost = step_cost
        self.terminal_makespan_coeff = terminal_makespan_coeff
        self.max_steps = max_steps

        self.instance: Optional[RCPSPInstance] = None
        self.predecessors: Dict[int, Sequence[int]] = {}
        self.node: Optional[BBNode] = None
        self.steps = 0

    def _load_instance(self) -> RCPSPInstance:
        if isinstance(self.instance_source, RCPSPInstance):
            return self.instance_source
        return load_instance(Path(self.instance_source))

    def _build_node(self, scheduled: Dict[int, ScheduleEntry], unscheduled: Iterable[int], depth: int) -> BBNode:
        unscheduled_set = set(unscheduled)
        ready = compute_ready_set(unscheduled_set, set(scheduled.keys()), self.predecessors)
        return BBNode(
            node_id=depth,
            scheduled=scheduled,
            ready=ready,
            unscheduled=unscheduled_set,
            lower_bound=lower_bound(self.instance, unscheduled_set, scheduled),
            parent_id=None,
            action=None,
            depth=depth,
        )

    def _observation_for_node(self, node: BBNode, incumbent: Optional[int] = None) -> Dict[str, torch.Tensor]:
        """Return tensors compatible with the PolicyMLP: candidate/global features plus a mask."""
        record: TrajectoryRecord = _build_record_for_node(
            self.instance,
            node,
            self.predecessors,
            incumbent=incumbent,
        )
        glob = global_features(record, max_resources=self.max_resources)
        ready_sorted = record.ready
        cand = [candidate_features(record, rid, max_resources=self.max_resources) for rid in ready_sorted]

        glob_tensor = torch.tensor(glob, dtype=torch.float32)
        cand_tensor = torch.tensor(cand, dtype=torch.float32) if cand else torch.zeros((0, 0), dtype=torch.float32)
        mask = torch.ones(len(ready_sorted), dtype=torch.bool)

        return {
            "global_feats": glob_tensor,
            "candidate_feats": cand_tensor,
            "ready_ids": torch.tensor(ready_sorted, dtype=torch.long),
            "action_mask": mask,
        }

    def reset(self, instance: Optional[RCPSPInstance | Path | str] = None) -> Dict[str, torch.Tensor]:
        """
        Start a fresh episode on the provided instance (or the default source).

        Returns:
            observation dict with tensors: global_feats [Fg], candidate_feats [R, Fc],
            ready_ids [R], action_mask [R].
        """
        if instance is not None:
            self.instance_source = instance
        self.instance = self._load_instance()
        self.predecessors = build_predecessors(self.instance)
        self.steps = 0

        unscheduled = set(self.instance.activities.keys())
        self.node = self._build_node(scheduled={}, unscheduled=unscheduled, depth=0)
        return self._observation_for_node(self.node, incumbent=None)

    def step(self, action_index: int) -> StepOutput:
        """
        Schedule the selected ready activity and advance the search.

        Args:
            action_index: index into the current ready set (sorted internally).
        """
        if self.node is None:
            raise RuntimeError("Call reset() before step().")
        ready_sorted = sorted(self.node.ready)

        info = {"ready_ids": ready_sorted}
        done = False
        # TODO: incorporate a terminal makespan-based penalty/bonus when optimizing for final performance.
        reward = -float(self.step_cost)

        if action_index < 0 or action_index >= len(ready_sorted):
            info["error"] = "invalid_action_index"
            info["done_reason"] = "invalid_action"
            return StepOutput(self._observation_for_node(self.node), reward, True, info)

        act_id = ready_sorted[action_index]
        start_time = earliest_feasible_start(
            self.instance,
            self.predecessors,
            self.node.scheduled,
            act_id,
            incumbent=None,
        )
        if start_time is None:
            info["error"] = "infeasible_action"
            info["done_reason"] = "infeasible_action"
            return StepOutput(self._observation_for_node(self.node), reward, True, info)

        duration = self.instance.activities[act_id].duration
        finish = start_time + duration
        scheduled = dict(self.node.scheduled)
        scheduled[act_id] = ScheduleEntry(start=start_time, finish=finish, duration=duration)
        unscheduled = set(self.node.unscheduled)
        unscheduled.discard(act_id)

        self.steps += 1
        next_node = self._build_node(scheduled=scheduled, unscheduled=unscheduled, depth=self.node.depth + 1)
        self.node = next_node

        if not next_node.unscheduled:
            done = True
            makespan = current_makespan(next_node.scheduled)
            reward -= self.terminal_makespan_coeff * float(makespan)
            info["makespan"] = makespan
            info["done_reason"] = "all_scheduled"

        if self.max_steps is not None and self.steps >= self.max_steps and not done:
            done = True
            info["done_reason"] = "max_steps"

        obs = self._observation_for_node(next_node)
        info["action"] = {"task": act_id, "start": start_time}
        info["steps"] = self.steps
        return StepOutput(obs, reward, done, info)
