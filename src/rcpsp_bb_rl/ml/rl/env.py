from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import torch

from rcpsp_bb_rl.bnb.lower_bounds import lower_bound
from rcpsp_bb_rl.bnb.precedence import build_predecessors, compute_ready_set
from rcpsp_bb_rl.bnb.scheduling import build_profile, earliest_feasible_start
from rcpsp_bb_rl.bnb.solver import BBNode, ScheduleEntry, current_makespan
from rcpsp_bb_rl.data.parsing import RCPSPInstance, load_instance
from rcpsp_bb_rl.ml.il.featurize import (
    NodeContext,
    candidate_features,
    global_features,
)


@dataclass
class StepOutput:
    observation: Dict[str, torch.Tensor]
    reward: float
    done: bool
    info: Dict


class BranchingEnv:
    """
    RL environment wrapping the serial B&B search.

    Each step corresponds to one branching decision: the agent picks which
    ready activity to schedule next. The environment maintains a DFS stack,
    tracks the incumbent makespan, and prunes nodes whose lower bound meets
    or exceeds the incumbent.

    Observations use the same NodeContext featurisation as the IL pipeline,
    so a BC-pretrained BranchingTransformer can be used as the initial policy
    without any adaptation.
    """

    def __init__(
        self,
        instance_source: RCPSPInstance | Path | str,
        max_resources: int = 4,
        step_cost: float = 0.0,
        terminal_makespan_coeff: float = 0.0,
        max_steps: Optional[int] = None,
    ) -> None:
        self.instance_source = instance_source
        self.max_resources = max_resources
        self.step_cost = step_cost
        self.terminal_makespan_coeff = terminal_makespan_coeff
        self.max_steps = max_steps

        self.instance: Optional[RCPSPInstance] = None
        self.predecessors: Dict = {}
        self.node: Optional[BBNode] = None
        self.steps: int = 0
        self.stack: List[BBNode] = []
        self.best_makespan: Optional[int] = None
        self.best_schedule: Optional[Dict[int, ScheduleEntry]] = None
        self.nodes_expanded: int = 0
        self._next_node_id: int = 0

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_instance(self) -> RCPSPInstance:
        if isinstance(self.instance_source, RCPSPInstance):
            return self.instance_source
        return load_instance(Path(self.instance_source))

    def _new_node_id(self) -> int:
        nid = self._next_node_id
        self._next_node_id += 1
        return nid

    def _make_node(
        self,
        scheduled: Dict[int, ScheduleEntry],
        unscheduled: set,
        depth: int,
        parent_id: Optional[int],
        action: Optional[str],
    ) -> BBNode:
        ready = compute_ready_set(unscheduled, set(scheduled.keys()), self.predecessors)
        lb = lower_bound(self.instance, unscheduled, scheduled)
        return BBNode(
            node_id=self._new_node_id(),
            scheduled=scheduled,
            ready=ready,
            unscheduled=unscheduled,
            lower_bound=lb,
            parent_id=parent_id,
            action=action,
            depth=depth,
        )

    def _earliest_starts_for_node(self, node: BBNode) -> Dict[int, Optional[int]]:
        horizon = sum(a.duration for a in self.instance.activities.values())
        profile = build_profile(
            self.instance.activities,
            self.instance.resource_caps,
            node.scheduled,
            horizon=self.best_makespan if self.best_makespan is not None else horizon,
        )
        return {
            rid: earliest_feasible_start(
                self.instance,
                self.predecessors,
                node.scheduled,
                rid,
                incumbent=self.best_makespan,
                profile=profile,
            )
            for rid in sorted(node.ready)
        }

    def _observe(self, node: BBNode) -> Dict[str, torch.Tensor]:
        earliest_starts = self._earliest_starts_for_node(node)
        ctx = NodeContext(
            instance=self.instance,
            scheduled=node.scheduled,
            unscheduled=node.unscheduled,
            ready=node.ready,
            lower_bound=node.lower_bound,
            incumbent=self.best_makespan,
            earliest_starts=earliest_starts,
        )
        ready_sorted = sorted(node.ready)
        glob = torch.tensor(
            global_features(ctx, self.max_resources, depth=node.depth),
            dtype=torch.float32,
        )
        cand = torch.tensor(
            [candidate_features(ctx, rid, self.max_resources) for rid in ready_sorted],
            dtype=torch.float32,
        )
        mask = torch.tensor(
            [earliest_starts.get(rid) is not None for rid in ready_sorted],
            dtype=torch.bool,
        )
        return {
            "global_feats": glob,
            "candidate_feats": cand,
            "ready_ids": torch.tensor(ready_sorted, dtype=torch.long),
            "action_mask": mask,
        }

    def _advance(self) -> bool:
        """
        Pop the DFS stack until we land on an expandable node.
        Handles pruning and incumbent updates along the way.
        """
        while self.stack:
            node = self.stack.pop()

            # LB pruning
            if self.best_makespan is not None and node.lower_bound >= self.best_makespan:
                node.status = "pruned"
                continue

            # Complete schedule — update incumbent
            if not node.unscheduled:
                node.status = "solution"
                ms = current_makespan(node.scheduled)
                if self.best_makespan is None or ms < self.best_makespan:
                    self.best_makespan = ms
                    self.best_schedule = node.scheduled
                continue

            # No ready activities (shouldn't happen in serial B&B, but guard anyway)
            if not node.ready:
                node.status = "pruned"
                continue

            # Check at least one ready activity is resource-feasible
            horizon = sum(a.duration for a in self.instance.activities.values())
            profile = build_profile(
                self.instance.activities,
                self.instance.resource_caps,
                node.scheduled,
                horizon=self.best_makespan if self.best_makespan is not None else horizon,
            )
            any_feasible = any(
                earliest_feasible_start(
                    self.instance, self.predecessors, node.scheduled,
                    rid, incumbent=self.best_makespan, profile=profile,
                ) is not None
                for rid in node.ready
            )
            if not any_feasible:
                node.status = "pruned"
                continue

            self.node = node
            return True

        self.node = None
        return False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(
        self,
        instance: Optional[RCPSPInstance | Path | str] = None,
    ) -> Dict[str, torch.Tensor]:
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
        root = self._make_node(
            scheduled={}, unscheduled=unscheduled,
            depth=0, parent_id=None, action=None,
        )
        self.stack.append(root)

        if not self._advance():
            raise RuntimeError("No expandable nodes at reset — instance may be trivial.")
        return self._observe(self.node)

    def step(self, action_index: int) -> StepOutput:
        """
        Branch on the activity at position action_index in the sorted ready set.

        The chosen activity is pushed first onto the DFS stack (explored first).
        All other ready activities are pushed after it (explored later).
        """
        if self.node is None:
            raise RuntimeError("Call reset() before step().")

        ready_sorted = sorted(self.node.ready)
        reward = -float(self.step_cost)
        info: Dict = {}

        if action_index < 0 or action_index >= len(ready_sorted):
            info["done_reason"] = "invalid_action"
            return StepOutput(self._observe(self.node), reward, True, info)

        chosen = ready_sorted[action_index]

        # Build profile once for all children
        horizon = sum(a.duration for a in self.instance.activities.values())
        profile = build_profile(
            self.instance.activities,
            self.instance.resource_caps,
            self.node.scheduled,
            horizon=self.best_makespan if self.best_makespan is not None else horizon,
        )

        # Push children in reverse order so chosen is on top of the stack
        ordered = [chosen] + [a for a in ready_sorted if a != chosen]
        for act_id in reversed(ordered):
            est = earliest_feasible_start(
                self.instance, self.predecessors, self.node.scheduled,
                act_id, incumbent=self.best_makespan, profile=profile,
            )
            if est is None:
                continue
            dur = self.instance.activities[act_id].duration
            child_sched = dict(self.node.scheduled)
            child_sched[act_id] = ScheduleEntry(start=est, finish=est + dur, duration=dur)
            child_unsched = set(self.node.unscheduled)
            child_unsched.discard(act_id)
            child = self._make_node(
                scheduled=child_sched,
                unscheduled=child_unsched,
                depth=self.node.depth + 1,
                parent_id=self.node.node_id,
                action=f"act {act_id}@{est}",
            )
            self.stack.append(child)

        self.steps += 1
        self.nodes_expanded += 1
        best_before = self.best_makespan

        found = self._advance()
        done = not found

        # Incumbent improvement reward
        if (
            self.best_makespan is not None
            and self.best_makespan != best_before
        ):
            improvement = (
                float(best_before - self.best_makespan)
                if best_before is not None
                else 0.0
            )
            reward += self.terminal_makespan_coeff * improvement
            info["best_makespan"] = self.best_makespan
            info["makespan_improvement"] = improvement

        if not done and self.max_steps is not None and self.steps >= self.max_steps:
            done = True
            info["done_reason"] = "max_steps"
        elif done:
            info["done_reason"] = "search_exhausted"

        info["steps"] = self.steps
        info["stack_size"] = len(self.stack)
        info["action"] = {"task": chosen}

        obs = self._observe(self.node) if (not done and self.node is not None) else {}
        return StepOutput(obs, reward, done, info)
