from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch

from rcpsp_bb_rl.bnb.dominance import normalize_dominance_spec
from rcpsp_bb_rl.bnb.lower_bounds import DEFAULT_LOWER_BOUND_ID, lower_bound
from rcpsp_bb_rl.bnb.scheduling import build_profile, earliest_feasible_start
from rcpsp_bb_rl.bnb.solver import BBNode, BnBSolver, ScheduleEntry, StepContext
from rcpsp_bb_rl.data.parsing import RCPSPInstance, load_instance
from rcpsp_bb_rl.ml.il.featurize import (
    NodeContext,
    candidate_features,
    global_features,
)


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class StepOutput:
    observation: Dict[str, torch.Tensor]
    reward: float
    done: bool
    info: Dict


@dataclass
class RewardConfig:
    """
    All reward coefficients in one place so train_ppo.py can pass them cleanly.

    Reward at each branching step:
        r = -step_cost
          + prune_coeff * (lb_pruned + dom_pruned since last step)
          + [if incumbent improved]:
                inc_coeff * (old_inc - new_inc)
              + gap_coeff * (old_gap - new_gap) / old_gap
          + [if done AND search_exhausted]:
                exhausted_coeff / nodes_expanded
    """
    step_cost: float = 0.01
    prune_coeff: float = 0.1
    inc_coeff: float = 1.0
    gap_coeff: float = 2.0
    exhausted_coeff: float = 10.0


@dataclass
class EpisodeStats:
    """Tracks per-episode statistics for logging in train_ppo.py."""
    nodes_expanded: int = 0
    nodes_pruned: int = 0
    dominance_pruned: int = 0
    incumbent_improvements: int = 0
    first_incumbent_node: Optional[int] = None
    first_incumbent_makespan: Optional[int] = None
    best_makespan: Optional[int] = None
    final_gap: Optional[float] = None
    done_reason: str = "unknown"
    total_reward: float = 0.0


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

class BranchingEnv:
    """
    RL environment for the serial B&B branching policy.

    Each episode solves one RCPSP instance. Each step is one branching
    decision — the agent picks which ready activity to schedule next.

    The environment wraps BnBSolver directly, so the B&B mechanics
    (dominance, LB pruning, child generation) are identical to evaluation.
    The solver is paused at each branching decision via a callback; the
    agent provides the ordering, then the solver resumes.

    Observations use NodeContext featurisation — identical to the IL
    pipeline — so a BC-pretrained BranchingTransformer warm-starts
    without any adaptation.
    """

    def __init__(
        self,
        instance_source: RCPSPInstance | Path | str,
        max_resources: int = 4,
        time_limit_s: float = 60.0,
        reward_cfg: Optional[RewardConfig] = None,
        dominance: object = "set_based",
        lb_spec: object = DEFAULT_LOWER_BOUND_ID,
    ) -> None:
        self.instance_source = instance_source
        self.max_resources = max_resources
        self.time_limit_s = time_limit_s
        self.reward_cfg = reward_cfg or RewardConfig()
        self.dominance_spec = normalize_dominance_spec(dominance)
        self.lb_spec = lb_spec

        # Set at reset()
        self.instance: Optional[RCPSPInstance] = None
        self._episode_stats: EpisodeStats = EpisodeStats()

        # Step-level state written by the callback, read by step()
        self._pending_node: Optional[BBNode] = None
        self._pending_ctx: Optional[StepContext] = None
        self._pending_incumbent: Optional[int] = None

        # Synchronisation between step() and the solver thread
        self._solver_gen = None
        self._done: bool = True
        self._done_reason: str = "unknown"
        self._steps: int = 0

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_instance(self) -> RCPSPInstance:
        if isinstance(self.instance_source, RCPSPInstance):
            return self.instance_source
        return load_instance(Path(self.instance_source))

    def _observe(self, node: BBNode, incumbent: Optional[int]) -> Dict[str, torch.Tensor]:
        horizon = sum(a.duration for a in self.instance.activities.values())
        profile = build_profile(
            self.instance.activities,
            self.instance.resource_caps,
            node.scheduled,
            horizon=horizon,
        )
        from rcpsp_bb_rl.bnb.precedence import build_predecessors
        predecessors = build_predecessors(self.instance)
        ready_sorted = sorted(node.ready)
        earliest_starts: Dict[int, Optional[int]] = {
            rid: earliest_feasible_start(
                self.instance, predecessors, node.scheduled,
                rid, incumbent=incumbent, profile=profile,
            )
            for rid in ready_sorted
        }
        ctx = NodeContext(
            instance=self.instance,
            scheduled=node.scheduled,
            unscheduled=node.unscheduled,
            ready=node.ready,
            lower_bound=node.lower_bound,
            incumbent=incumbent,
            earliest_starts=earliest_starts,
        )
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

    def _compute_reward(
        self,
        ctx: StepContext,
        node_lb: int,
        done: bool,
        done_reason: str,
        nodes_expanded: int,
    ) -> Tuple[float, Dict]:
        cfg = self.reward_cfg
        reward = 0.0
        breakdown: Dict[str, float] = {}

        # 1. Step cost
        r_step = -cfg.step_cost
        reward += r_step
        breakdown["step"] = r_step

        # 2. Pruning signal (dense)
        total_pruned = ctx.lb_pruned + ctx.dom_pruned
        r_prune = cfg.prune_coeff * total_pruned
        reward += r_prune
        breakdown["prune"] = r_prune

        # 3 & 4. Incumbent improvement + gap closure
        r_inc = 0.0
        r_gap = 0.0
        old_inc = ctx.incumbent_before
        new_inc = ctx.incumbent_after
        if old_inc is not None and new_inc is not None and new_inc < old_inc:
            improvement = float(old_inc - new_inc)
            r_inc = cfg.inc_coeff * improvement

            old_gap = float(old_inc - node_lb)
            new_gap = float(new_inc - node_lb)
            if old_gap > 0:
                r_gap = cfg.gap_coeff * (old_gap - new_gap) / old_gap

            reward += r_inc + r_gap

        breakdown["incumbent"] = r_inc
        breakdown["gap_closure"] = r_gap

        # 5. Search exhaustion bonus
        r_exhausted = 0.0
        if done and done_reason == "search_exhausted" and nodes_expanded > 0:
            r_exhausted = cfg.exhausted_coeff / nodes_expanded
            reward += r_exhausted
        breakdown["exhausted"] = r_exhausted

        return reward, breakdown

    # ------------------------------------------------------------------
    # Generator-based solver coroutine
    # ------------------------------------------------------------------

    def _run_solver(self, instance: RCPSPInstance):
        """
        Generator that drives BnBSolver step by step.

        BnBSolver is synchronous, so we run it in a daemon thread and
        communicate via two single-slot queues:
          to_env   : solver → env  ("branch", node, incumbent, ctx) or ("done", result)
          to_solver: env → solver  (chosen ordering list)

        The generator yields ("branch", node, incumbent, ctx) each time the
        solver needs a branching decision, and ("done", result) when finished.
        The caller sends back the chosen ordering via generator.send().
        """
        import queue
        import threading

        solver = BnBSolver(instance=instance)
        to_solver: queue.Queue = queue.Queue(maxsize=1)
        to_env: queue.Queue = queue.Queue(maxsize=1)

        def _order_fn(node: BBNode, incumbent: Optional[int], step_ctx: StepContext) -> List[int]:
            to_env.put(("branch", node, incumbent, step_ctx))
            return to_solver.get()

        def _run():
            try:
                result = solver.solve(
                    order_ready_fn=_order_fn,
                    lb_spec=self.lb_spec,
                    dominance=self.dominance_spec,
                    time_limit_s=self.time_limit_s,
                )
                to_env.put(("done", result))
            except Exception as exc:
                to_env.put(("error", exc))

        threading.Thread(target=_run, daemon=True).start()

        while True:
            msg = to_env.get()
            if msg[0] == "branch":
                _, node, incumbent, step_ctx = msg
                ordering = yield ("branch", node, incumbent, step_ctx)
                to_solver.put(ordering)
            elif msg[0] == "done":
                yield ("done", msg[1])
                return
            else:
                raise msg[1]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(
        self,
        instance: Optional[RCPSPInstance | Path | str] = None,
    ) -> Dict[str, torch.Tensor]:
        """Start a fresh episode."""
        if instance is not None:
            if isinstance(instance, (Path, str)):
                self.instance_source = instance
                self.instance = load_instance(Path(instance))
            else:
                self.instance_source = instance
                self.instance = instance
        else:
            self.instance = self._load_instance()

        self._episode_stats = EpisodeStats()
        self._steps = 0
        self._done = False
        self._done_reason = "unknown"

        self._solver_gen = self._run_solver(self.instance)
        msg = next(self._solver_gen)

        if msg[0] == "done":
            # Trivial instance — solved without any branching decision
            _, result = msg
            self._done = True
            self._done_reason = "search_exhausted"
            self._episode_stats.done_reason = self._done_reason
            self._episode_stats.best_makespan = result.best_makespan
            raise RuntimeError("Instance solved at root — no branching decisions needed.")

        _, node, incumbent, step_ctx = msg
        self._pending_node = node
        self._pending_ctx = step_ctx
        self._pending_incumbent = incumbent

        return self._observe(node, incumbent)

    def step(self, action_index: int) -> StepOutput:
        """
        Branch on the activity at position action_index in sorted(node.ready).

        The chosen activity is placed first in the ordering passed to the
        solver; the solver explores it first (DFS/LIFO push order).
        """
        if self._done or self._solver_gen is None:
            raise RuntimeError("Call reset() before step().")

        node = self._pending_node
        ctx = self._pending_ctx
        incumbent = self._pending_incumbent
        ready_sorted = sorted(node.ready)
        info: Dict = {}

        if action_index < 0 or action_index >= len(ready_sorted):
            info["done_reason"] = "invalid_action"
            self._episode_stats.done_reason = "invalid_action"
            self._done = True
            return StepOutput({}, 0.0, True, info)

        chosen = ready_sorted[action_index]
        # Put chosen first; solver pushes in reversed order so chosen is
        # explored first (LIFO stack).
        ordering = [chosen] + [a for a in ready_sorted if a != chosen]

        # Resume the solver with the chosen ordering.
        try:
            msg = self._solver_gen.send(ordering)
        except StopIteration:
            msg = ("done_implicit", None)

        self._steps += 1

        # Update episode stats from StepContext
        old_inc = ctx.incumbent_before
        new_inc = ctx.incumbent_after
        self._episode_stats.nodes_expanded = ctx.nodes_expanded
        self._episode_stats.nodes_pruned += ctx.lb_pruned
        self._episode_stats.dominance_pruned += ctx.dom_pruned

        if old_inc is not None and new_inc is not None and new_inc < old_inc:
            self._episode_stats.incumbent_improvements += 1
            if self._episode_stats.first_incumbent_node is None:
                self._episode_stats.first_incumbent_node = ctx.nodes_expanded
                self._episode_stats.first_incumbent_makespan = new_inc
        elif old_inc is None and new_inc is not None:
            self._episode_stats.incumbent_improvements += 1
            if self._episode_stats.first_incumbent_node is None:
                self._episode_stats.first_incumbent_node = ctx.nodes_expanded
                self._episode_stats.first_incumbent_makespan = new_inc

        # Determine done
        done = False
        done_reason = "running"

        if msg[0] in ("done", "done_implicit"):
            done = True
            if msg[0] == "done":
                result = msg[1]
                self._episode_stats.best_makespan = result.best_makespan
                self._episode_stats.nodes_expanded = result.nodes_expanded
                done_reason = result.done_reason
            else:
                done_reason = "search_exhausted"

        # Compute reward using the StepContext from the *previous* branching
        # decision (ctx), which accumulated pruning/incumbent data up to now.
        reward, breakdown = self._compute_reward(
            ctx=ctx,
            node_lb=node.lower_bound,
            done=done,
            done_reason=done_reason,
            nodes_expanded=ctx.nodes_expanded,
        )

        if done:
            self._done = True
            self._done_reason = done_reason
            self._episode_stats.done_reason = done_reason
            self._episode_stats.total_reward += reward
            if self._episode_stats.best_makespan is not None:
                instance_lb = lower_bound(
                    self.instance,
                    set(self.instance.activities.keys()),
                    {},
                    lb_id=self.lb_spec,
                )
                self._episode_stats.final_gap = float(
                    self._episode_stats.best_makespan - instance_lb
                )

        info.update({
            "done_reason": done_reason,
            "steps": self._steps,
            "nodes_expanded": ctx.nodes_expanded,
            "best_makespan": new_inc,
            "action_task": chosen,
            "lb_pruned": ctx.lb_pruned,
            "dom_pruned": ctx.dom_pruned,
            "reward_breakdown": breakdown,
        })

        if not done and msg[0] == "branch":
            _, next_node, next_incumbent, next_ctx = msg
            self._pending_node = next_node
            self._pending_ctx = next_ctx
            self._pending_incumbent = next_incumbent
            obs = self._observe(next_node, next_incumbent)
        else:
            obs = {}

        return StepOutput(obs, reward, done, info)

    @property
    def episode_stats(self) -> EpisodeStats:
        return self._episode_stats
