from __future__ import annotations

from dataclasses import dataclass, field
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
    Proof-oriented reward coefficients (see reward.txt for full design).

    Reward at each branching step:
        r = -step_cost
          + [if first incumbent appeared at this step]:
                first_inc_coeff * (cp_lb / first_incumbent)
          + [if an incumbent exists at the start of the step]:
                proof_coeff *
                (proof_burden_before - proof_burden_after)
                / max(1, nodes_expanded_delta)
          + [if done AND search_exhausted AND best_makespan is not None]:
                proof_bonus

    where
        cp_lb              = root critical-path lower bound (computed at reset)
        first_incumbent    = makespan of the first complete schedule found
        proof_burden       = sum over open stack nodes of max(0, incumbent - lb)
        nodes_expanded_delta = nodes expanded between this branching decision
                               and the next decision (or termination)
    """
    step_cost: float = 0.01
    first_inc_coeff: float = 3.0
    proof_coeff: float = 1.0
    proof_bonus: float = 30.0


@dataclass
class EpisodeStats:
    """Tracks per-episode statistics for logging in train_ppo.py."""
    nodes_expanded: int = 0
    nodes_pruned: int = 0
    dominance_pruned: int = 0
    incumbent_improvements: int = 0
    first_incumbent_node: Optional[int] = None
    first_incumbent_makespan: Optional[int] = None
    last_incumbent_node: Optional[int] = None
    last_incumbent_makespan: Optional[int] = None
    best_makespan: Optional[int] = None
    final_gap: Optional[float] = None
    done_reason: str = "unknown"
    total_reward: float = 0.0
    reward_breakdown: Dict[str, float] = field(default_factory=lambda: {
        "step": 0.0,
        "first_incumbent": 0.0,
        "proof_burden": 0.0,
        "proof_bonus": 0.0,
    })


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
        self._n_activities: int = 0
        self._cp_lb: int = 0

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
        *,
        pre_inc: Optional[int],
        post_inc: Optional[int],
        pre_burden: int,
        post_burden: int,
        nodes_delta: int,
        done: bool,
        done_reason: str,
    ) -> Tuple[float, Dict[str, float]]:
        """
        Proof-oriented reward (see RewardConfig docstring and reward.txt).

        All quantities reflect the advance triggered by the action just taken:
            pre_*  : state at the branching decision the agent acted on
            post_* : state at the next branching decision (or termination)
        """
        cfg = self.reward_cfg
        reward = 0.0
        breakdown: Dict[str, float] = {}

        # 1. Step cost
        r_step = -cfg.step_cost
        reward += r_step
        breakdown["step"] = r_step

        # 2. First-incumbent reward — only on the step where the first
        #    complete feasible schedule appears.
        r_first_inc = 0.0
        if pre_inc is None and post_inc is not None:
            if post_inc > 0 and self._cp_lb > 0:
                quality = float(self._cp_lb) / float(post_inc)
            else:
                quality = 0.0
            r_first_inc = cfg.first_inc_coeff * quality
            reward += r_first_inc
        breakdown["first_incumbent"] = r_first_inc

        # 3. Proof-burden progress — only when an incumbent already existed
        #    at the start of the step (i.e., we are already in proof mode).
        r_proof = 0.0
        if pre_inc is not None:
            denom = max(1, int(nodes_delta))
            r_proof = cfg.proof_coeff * float(pre_burden - post_burden) / float(denom)
            reward += r_proof
        breakdown["proof_burden"] = r_proof

        # 4. Terminal proof bonus — only when the search is fully exhausted
        #    and an incumbent exists (i.e., optimality is proven).
        r_proof_bonus = 0.0
        proved_optimal = (
            done
            and done_reason == "search_exhausted"
            and self._episode_stats.best_makespan is not None
        )
        if proved_optimal:
            r_proof_bonus = cfg.proof_bonus
            reward += r_proof_bonus
        breakdown["proof_bonus"] = r_proof_bonus

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
        self._n_activities = len(self.instance.activities)
        self._cp_lb = int(lower_bound(
            self.instance,
            set(self.instance.activities.keys()),
            {},
            lb_id=self.lb_spec,
        ))

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

        # ---- Snapshot pre-action state ----
        # ctx was created when the solver paused for THIS branching decision,
        # so ctx.incumbent_after / ctx.proof_burden / ctx.nodes_expanded reflect
        # the moment the agent is about to act.
        pre_inc: Optional[int] = ctx.incumbent_after
        pre_burden: int = ctx.proof_burden
        pre_nodes: int = ctx.nodes_expanded

        # Resume the solver with the chosen ordering.
        try:
            msg = self._solver_gen.send(ordering)
        except StopIteration:
            msg = ("done_implicit", None)

        self._steps += 1

        # ---- Snapshot post-action state (after the advance triggered by
        #      the chosen ordering) ----
        done = False
        done_reason = "running"
        next_ctx: Optional[StepContext] = None
        result_obj = None

        if msg[0] == "branch":
            _, next_node, next_incumbent, next_ctx = msg
            post_inc: Optional[int] = next_ctx.incumbent_after
            post_burden: int = next_ctx.proof_burden
            post_nodes: int = next_ctx.nodes_expanded
            advance_lb_pruned = next_ctx.lb_pruned
            advance_dom_pruned = next_ctx.dom_pruned
        elif msg[0] == "done":
            _, result_obj = msg
            done = True
            done_reason = result_obj.done_reason
            post_inc = result_obj.best_makespan
            post_burden = result_obj.final_proof_burden
            post_nodes = result_obj.nodes_expanded
            advance_lb_pruned = 0
            advance_dom_pruned = 0
        else:  # done_implicit — generator already exhausted
            done = True
            done_reason = "search_exhausted"
            post_inc = pre_inc
            post_burden = 0
            post_nodes = pre_nodes
            advance_lb_pruned = 0
            advance_dom_pruned = 0

        nodes_delta = max(0, post_nodes - pre_nodes)

        # ---- Update episode stats ----
        if msg[0] == "branch":
            self._episode_stats.nodes_expanded = post_nodes
            self._episode_stats.nodes_pruned += advance_lb_pruned
            self._episode_stats.dominance_pruned += advance_dom_pruned
        elif result_obj is not None:
            self._episode_stats.nodes_expanded = result_obj.nodes_expanded
            self._episode_stats.nodes_pruned = result_obj.nodes_pruned
            self._episode_stats.dominance_pruned = result_obj.dominance_pruned_children
            self._episode_stats.best_makespan = result_obj.best_makespan

        # Incumbent-improvement tracking: pre_inc -> post_inc during this advance.
        if pre_inc is None and post_inc is not None:
            if self._episode_stats.first_incumbent_node is None:
                self._episode_stats.first_incumbent_node = post_nodes
                self._episode_stats.first_incumbent_makespan = post_inc
            self._episode_stats.last_incumbent_node = post_nodes
            self._episode_stats.last_incumbent_makespan = post_inc
        elif pre_inc is not None and post_inc is not None and post_inc < pre_inc:
            self._episode_stats.incumbent_improvements += 1
            self._episode_stats.last_incumbent_node = post_nodes
            self._episode_stats.last_incumbent_makespan = post_inc

        # ---- Compute reward (proof-oriented) ----
        reward, breakdown = self._compute_reward(
            pre_inc=pre_inc,
            post_inc=post_inc,
            pre_burden=pre_burden,
            post_burden=post_burden,
            nodes_delta=nodes_delta,
            done=done,
            done_reason=done_reason,
        )

        for key, value in breakdown.items():
            self._episode_stats.reward_breakdown[key] = (
                self._episode_stats.reward_breakdown.get(key, 0.0) + value
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
            "nodes_expanded": post_nodes,
            "nodes_delta": nodes_delta,
            "best_makespan": post_inc,
            "action_task": chosen,
            "lb_pruned": advance_lb_pruned,
            "dom_pruned": advance_dom_pruned,
            "proof_burden_before": pre_burden,
            "proof_burden_after": post_burden,
            "reward_breakdown": breakdown,
        })

        if not done and msg[0] == "branch":
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
