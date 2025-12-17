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
    Environment wrapper around the B&B search.

    Each step expands a B&B node by picking the next ready activity to branch on.
    The environment maintains a DFS stack, an incumbent makespan, and automatically
    prunes nodes whose lower bound exceeds the incumbent. Observations mirror the
    imitation-learning featurization so PPO (or similar) can be warm-started from
    the behavior-cloned policy.
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
        self.stack: List[BBNode] = []
        self.best_makespan: Optional[int] = None
        self.best_schedule: Optional[Dict[int, ScheduleEntry]] = None
        self.nodes_expanded = 0
        self._next_node_id = 0

    def _load_instance(self) -> RCPSPInstance:
        if isinstance(self.instance_source, RCPSPInstance):
            return self.instance_source
        return load_instance(Path(self.instance_source))

    def _new_node_id(self) -> int:
        nid = self._next_node_id
        self._next_node_id += 1
        return nid

    def _build_node(
        self,
        scheduled: Dict[int, ScheduleEntry],
        unscheduled: Iterable[int],
        depth: int,
        parent_id: Optional[int],
        action: Optional[str],
    ) -> BBNode:
        unscheduled_set = set(unscheduled)
        ready = compute_ready_set(unscheduled_set, set(scheduled.keys()), self.predecessors)
        return BBNode(
            node_id=self._new_node_id(),
            scheduled=scheduled,
            ready=ready,
            unscheduled=unscheduled_set,
            lower_bound=lower_bound(self.instance, unscheduled_set, scheduled),
            parent_id=parent_id,
            action=action,
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
        expected_ready = sorted(node.ready)
        if record.ready != expected_ready:
            raise RuntimeError(
                f"Ready order mismatch: record.ready={record.ready} vs sorted(node.ready)={expected_ready}"
            )
        ready_sorted = expected_ready
        cand = [candidate_features(record, rid, max_resources=self.max_resources) for rid in ready_sorted]
        mask = [record.earliest_start.get(rid) is not None for rid in ready_sorted]

        glob_tensor = torch.tensor(glob, dtype=torch.float32)
        cand_tensor = torch.tensor(cand, dtype=torch.float32) if cand else torch.zeros((0, 0), dtype=torch.float32)
        mask_tensor = torch.tensor(mask, dtype=torch.bool)

        return {
            "global_feats": glob_tensor,
            "candidate_feats": cand_tensor,
            "ready_ids": torch.tensor(ready_sorted, dtype=torch.long),
            "action_mask": mask_tensor,
        }

    def _advance_to_next_expandable(self) -> bool:
        """
        Pop from the DFS stack until we find a node that is expandable (has ready tasks
        and is not pruned by the incumbent). Updates incumbents on solutions encountered.
        """
        while self.stack:
            candidate = self.stack.pop()

            if self.best_makespan is not None and candidate.lower_bound >= self.best_makespan:
                candidate.status = "pruned"
                continue

            if not candidate.unscheduled:
                candidate.status = "solution"
                makespan = current_makespan(candidate.scheduled)
                if self.best_makespan is None or makespan < self.best_makespan:
                    self.best_makespan = makespan
                    self.best_schedule = candidate.scheduled
                continue

            if not candidate.ready:
                candidate.status = "pruned"
                continue

            # Skip nodes whose ready set is entirely infeasible w.r.t. resources/incumbent.
            feasibles = [
                earliest_feasible_start(
                    self.instance,
                    self.predecessors,
                    candidate.scheduled,
                    rid,
                    incumbent=self.best_makespan,
                )
                for rid in candidate.ready
            ]
            if all(st is None for st in feasibles):
                candidate.status = "pruned"
                continue

            self.node = candidate
            return True

        self.node = None
        return False

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
        self.stack = []
        self.best_makespan = None
        self.best_schedule = None
        self.nodes_expanded = 0
        self._next_node_id = 0

        unscheduled = set(self.instance.activities.keys())
        root = self._build_node(
            scheduled={},
            unscheduled=unscheduled,
            depth=0,
            parent_id=None,
            action=None,
        )
        self.stack.append(root)
        found = self._advance_to_next_expandable()
        if not found:
            raise RuntimeError("No expandable nodes found during reset; instance may be trivial.")
        return self._observation_for_node(self.node, incumbent=self.best_makespan)

    def step(self, action_index: int) -> StepOutput:
        """
        Expand the current B&B node by selecting the next ready activity to branch on.

        Args:
            action_index: index into the current ready set (sorted internally).
        """
        if self.node is None:
            raise RuntimeError("Call reset() before step().")
        ready_sorted = sorted(self.node.ready)

        info = {"ready_ids": ready_sorted}
        done = False
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
            incumbent=self.best_makespan,
        )
        if start_time is None:
            info["error"] = "infeasible_action"
            info["done_reason"] = "infeasible_action"
            return StepOutput(self._observation_for_node(self.node), reward, True, info)

        # Order children so the chosen action is explored first; others follow the default ordering.
        ordered_acts = [act_id] + [aid for aid in ready_sorted if aid != act_id]

        for aid in reversed(ordered_acts):
            st = earliest_feasible_start(
                self.instance,
                self.predecessors,
                self.node.scheduled,
                aid,
                incumbent=self.best_makespan,
            )
            if st is None:
                continue
            dur = self.instance.activities[aid].duration
            fin = st + dur
            child_sched = dict(self.node.scheduled)
            child_sched[aid] = ScheduleEntry(start=st, finish=fin, duration=dur)
            child_unsched = set(self.node.unscheduled)
            child_unsched.discard(aid)
            child_node = self._build_node(
                scheduled=child_sched,
                unscheduled=child_unsched,
                depth=self.node.depth + 1,
                parent_id=self.node.node_id,
                action=f"act {aid}@{st}",
            )
            self.stack.append(child_node)

        self.steps += 1
        self.nodes_expanded += 1

        best_before = self.best_makespan
        found_next = self._advance_to_next_expandable()
        if not found_next:
            done = True
            info["done_reason"] = "search_exhausted"
            obs = self._observation_for_node(self.node, incumbent=self.best_makespan) if self.node else {}
        else:
            obs = self._observation_for_node(self.node, incumbent=self.best_makespan)

        # Reward bonus if we improved the incumbent during this expansion.
        if self.best_makespan is not None and self.best_makespan != best_before:
            reward -= self.terminal_makespan_coeff * float(self.best_makespan)
            info["best_makespan"] = self.best_makespan

        if self.max_steps is not None and self.steps >= self.max_steps and not done:
            done = True
            info["done_reason"] = "max_steps"

        info["action"] = {"task": act_id, "start": start_time}
        info["steps"] = self.steps
        info["stack_size"] = len(self.stack)
        return StepOutput(obs, reward, done, info)
