"""Inspect one frozen rollout and its truncation targets.

This is a read-only debugging tool: it runs one policy rollout, builds both
the legacy ``drop`` targets and the ``critic_bootstrap`` targets from that
*same* search tree, and reports the difference.  In particular, it answers
the mechanical question behind the unfinished-search credit fix:

    which visited decisions were invalid with ``drop``, but become trainable
    after the remaining DFS frontier is completed exactly or bootstrapped?

The debugger follows the PPO training reward-scale rules.  When an oracle
node-count file is configured, pass ``--episode``: alpha is then
``clip(c_target / N_ref, alpha_min, alpha_max)`` using the same
``(episode, instance filename)`` lookup as training.  A missing oracle record
uses the configured estimator fallback, again matching training.

Examples
--------
Use the configuration saved in a PPO checkpoint::

    python3 scripts/debug_critic.py \
        --checkpoint models/train-full-bootstrap/policy_ppo_best.pt \
        --instance data/train/1kNetRes/RU_inst180001.rcp \
        --episode 1 --sample --seed 42

Or explicitly supply the configuration used for a run::

    python3 scripts/debug_critic.py \
        --checkpoint <checkpoint.pt> --config <train-config.json> \
        --instance <instance.rcp> --episode 1 --time-limit-s 120 \
        --sample --seed 42

``--sample --seed`` makes the fresh debug rollout reproducible.  It cannot
reconstruct a historical training trajectory exactly: training's RNG has
already advanced through prior episodes.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from rcpsp_bb_rl.data.parsing import load_instance  # noqa: E402
from rcpsp_bb_rl.ml.estimator import load_estimator_checkpoint, predict_difficulty  # noqa: E402
from rcpsp_bb_rl.ml.models import load_policy_checkpoint  # noqa: E402
from rcpsp_bb_rl.ml.rl import BranchingEnv  # noqa: E402
from rcpsp_bb_rl.ml.rl.ranking_policy import make_ranking_action  # noqa: E402
from rcpsp_bb_rl.ml.rl.tree_return import (  # noqa: E402
    TreeAdvantages,
    compute_episode_advantages_decoupled,
    make_bonus_reward_fn,
    make_cost_reward_fn,
)

# Import the actual training sampler and oracle parser rather than maintaining
# a second, almost-identical implementation in a diagnostic script.  This is
# safe: train_ppo_gpu only enters main() under its __name__ == "__main__" guard.
from train_ppo_gpu import (  # noqa: E402
    StagedEpisode,
    load_oracle_node_counts,
    subsample_critic_singletons,
    subsample_episode,
)


DEFAULTS: Dict[str, Any] = {
    "max_resources": 4,
    "dominance": "set_based",
    "time_limit_s": 60.0,
    "alpha": 0.001,
    "beta1": 1.0,
    "beta2": 1.0,
    "tree_gamma": 1.0,
    "tree_gamma_cost": None,
    "tree_gamma_bonus": None,
    "estimator_path": None,
    "oracle_node_counts_path": None,
    "c_target": 1.0,
    "alpha_min": 1e-6,
    "alpha_max": 0.5,
    "episode_transition_cap": 8192,
    "critic_singleton_transition_cap": 2048,
    "stratify_time_bands": 3,
    "stratify_depth_bands": 4,
    "incumbent_window": 25,
    "seed": 42,
}


@dataclass
class Rollout:
    """The parts of one rollout needed for backup and sampler inspection."""

    node_ids: List[Optional[int]]
    values: List[Tuple[float, float]]  # independently normalized cost/bonus values
    depths: List[int]
    feasible_counts: List[int]
    incumbent_steps: List[int]
    first_incumbent_step: Optional[int]


@dataclass
class FrontierReport:
    """How a time-limit DFS frontier was completed for bootstrap."""

    total: int = 0
    exact_incumbent_pruned: int = 0
    exact_dead_end: int = 0
    exact_all_infeasible: int = 0
    critic_estimated: int = 0
    unresolved_complete_schedule: int = 0
    pending_not_on_frontier: int = 0


@dataclass
class RewardScale:
    alpha: float
    beta1: float
    beta2: float
    gamma_cost: float
    gamma_bonus: float
    n_ref: Optional[float]
    source: str
    oracle_path: Optional[Path]


# ---------------------------------------------------------------------------
# CLI and configuration
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Frozen one-rollout comparison of drop and critic-bootstrap targets.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--checkpoint", required=True, help="Two-head PPO checkpoint (.pt).")
    p.add_argument("--instance", required=True, help="One RCPSP instance file.")
    p.add_argument(
        "--config",
        help="Optional training JSON. It overrides the train_config stored in the checkpoint.",
    )
    p.add_argument(
        "--episode",
        type=int,
        help="Training episode number for the oracle (required when an oracle is configured).",
    )
    p.add_argument("--time-limit-s", type=float, help="Per-solve time limit.")
    p.add_argument("--dominance", help="Dominance specification.")
    p.add_argument("--max-resources", type=int, help="Resource feature dimension.")
    p.add_argument("--device", default="cpu", help="Torch device (cpu|cuda).")

    # Defaults are None so a checkpoint/config's values survive.  --alpha is
    # an intentional manual/static override of dynamic reward scaling.
    p.add_argument("--alpha", type=float, help="Fixed alpha; bypasses oracle/estimator scaling.")
    p.add_argument("--beta1", type=float, help="First-incumbent bonus weight.")
    p.add_argument("--beta2", type=float, help="Incumbent-improvement bonus weight.")
    p.add_argument("--gamma-cost", type=float, help="Cost-channel subtree discount.")
    p.add_argument("--gamma-bonus", type=float, help="Bonus-channel subtree discount.")
    p.add_argument("--oracle-node-counts", help="Oracle JSONL path; overrides config.")
    p.add_argument("--estimator-path", help="Estimator fallback path; overrides config.")
    p.add_argument("--c-target", type=float, help="Dynamic alpha numerator.")
    p.add_argument("--alpha-min", type=float, help="Dynamic alpha lower clip.")
    p.add_argument("--alpha-max", type=float, help="Dynamic alpha upper clip.")

    p.add_argument("--sample", action="store_true", help="Sample full rankings (default: greedy).")
    p.add_argument("--seed", type=int, help="Fresh rollout and sampler RNG seed when sampling.")
    p.add_argument("--no-subsample", action="store_true", help="Skip retained-row table.")
    p.add_argument("--subsample-seed", type=int, help="Fresh RNG seed for sampler inspection.")
    p.add_argument("--max-print", type=int, default=12, help="Rows to print per depth bucket.")
    return p.parse_args()


def _load_checkpoint_payload(path: str, device: str) -> Mapping[str, Any]:
    payload = torch.load(path, map_location=device)
    if not isinstance(payload, Mapping):
        raise ValueError(f"Checkpoint at {path} is not a mapping payload.")
    return payload


def resolve_config(args: argparse.Namespace, checkpoint: Mapping[str, Any]) -> Dict[str, Any]:
    """Merge defaults, checkpoint train_config, optional JSON, then CLI values."""
    config = dict(DEFAULTS)
    saved = checkpoint.get("train_config")
    if saved is not None:
        if not isinstance(saved, Mapping):
            raise ValueError("Checkpoint train_config must be a JSON-like mapping.")
        config.update(saved)
    if args.config:
        with Path(args.config).open() as handle:
            external = json.load(handle)
        if not isinstance(external, Mapping):
            raise ValueError("--config must contain a JSON object.")
        config.update(external)

    overrides = {
        "max_resources": args.max_resources,
        "dominance": args.dominance,
        "time_limit_s": args.time_limit_s,
        "beta1": args.beta1,
        "beta2": args.beta2,
        "estimator_path": args.estimator_path,
        "oracle_node_counts_path": args.oracle_node_counts,
        "c_target": args.c_target,
        "alpha_min": args.alpha_min,
        "alpha_max": args.alpha_max,
    }
    for key, value in overrides.items():
        if value is not None:
            config[key] = value
    if args.gamma_cost is not None:
        config["tree_gamma_cost"] = args.gamma_cost
    if args.gamma_bonus is not None:
        config["tree_gamma_bonus"] = args.gamma_bonus
    return config


def load_value_norm(checkpoint: Mapping[str, Any]) -> tuple[Tuple[float, float], Tuple[float, float]]:
    """Recover ``(cost, bonus)`` mean/std pairs from a two-head PPO checkpoint."""
    vn = checkpoint.get("value_norm")
    if not isinstance(vn, Mapping) or not isinstance(vn.get("cost"), Mapping) or not isinstance(vn.get("bonus"), Mapping):
        raise ValueError(
            "This debugger expects a two-head PPO checkpoint with "
            "value_norm={'cost': ..., 'bonus': ...}."
        )

    def one_channel(channel: Mapping[str, Any]) -> Tuple[float, float]:
        mean = float(channel.get("mean", 0.0))
        var = channel.get("var")
        std = float(channel["std"]) if "std" in channel else float(var) ** 0.5 if var is not None else 1.0
        return mean, max(std, 1e-8)

    return one_channel(vn["cost"]), one_channel(vn["bonus"])


def resolve_reward_scale(
    *,
    args: argparse.Namespace,
    config: Mapping[str, Any],
    instance,
    instance_name: str,
    device: torch.device,
) -> RewardScale:
    """Apply the training oracle -> estimator fallback -> static alpha policy."""
    beta1 = float(config["beta1"])
    beta2 = float(config["beta2"])
    tree_gamma = float(config["tree_gamma"])
    gamma_cost = float(
        config["tree_gamma_cost"] if config["tree_gamma_cost"] is not None else tree_gamma
    )
    gamma_bonus = float(
        config["tree_gamma_bonus"] if config["tree_gamma_bonus"] is not None else tree_gamma
    )
    if args.alpha is not None:
        return RewardScale(
            alpha=float(args.alpha), beta1=beta1, beta2=beta2,
            gamma_cost=gamma_cost, gamma_bonus=gamma_bonus,
            n_ref=None, source="static_override", oracle_path=None,
        )

    oracle_raw = config.get("oracle_node_counts_path")
    oracle_path = None if not oracle_raw else Path(str(oracle_raw))
    estimator_path = config.get("estimator_path")
    c_target = float(config["c_target"])
    alpha_min = float(config["alpha_min"])
    alpha_max = float(config["alpha_max"])
    n_ref: Optional[float] = None
    source = "static"

    if oracle_path is not None:
        # Match train_ppo_gpu's contract: an oracle run must be able to fall
        # back to the estimator when a record is missing.  Falling all the way
        # back to static alpha here would make the diagnostic misleading.
        if not estimator_path:
            raise ValueError(
                "oracle_node_counts_path requires estimator_path for a missing-record fallback, "
                "matching PPO training."
            )
        if args.episode is None:
            raise ValueError(
                "This run configures oracle alpha. Pass --episode so the debugger can "
                "look up the same (episode, instance) reference as PPO training."
            )
        oracle = load_oracle_node_counts(oracle_path)
        oracle_nodes = oracle.get((args.episode, instance_name))
        if oracle_nodes is not None:
            n_ref = float(oracle_nodes)
            source = "oracle"
        else:
            source = "estimator_fallback"

    if n_ref is None and estimator_path:
        estimator, scaler, _ = load_estimator_checkpoint(str(estimator_path), device=device)
        n_ref = predict_difficulty(estimator, scaler, instance, device=device)
        if oracle_path is None:
            source = "estimator"

    if n_ref is not None:
        alpha = float(np.clip(c_target / n_ref, alpha_min, alpha_max))
    else:
        alpha = float(config["alpha"])

    return RewardScale(
        alpha=alpha, beta1=beta1, beta2=beta2,
        gamma_cost=gamma_cost, gamma_bonus=gamma_bonus,
        n_ref=n_ref, source=source, oracle_path=oracle_path,
    )


# ---------------------------------------------------------------------------
# Frozen rollout and frontier completion
# ---------------------------------------------------------------------------

def rollout(model, env: BranchingEnv, instance, device: torch.device, sample: bool) -> Rollout:
    """Run one frozen episode, retaining only data needed by the debugger."""
    node_ids: List[Optional[int]] = []
    values: List[Tuple[float, float]] = []
    depths: List[int] = []
    feasible_counts: List[int] = []
    incumbent_steps: List[int] = []
    first_incumbent_step: Optional[int] = None
    previous_incumbent: Optional[int] = None

    obs = env.reset(instance=instance)
    model.eval()
    while True:
        cand = obs["candidate_feats"].to(device)
        glob = obs["global_feats"].to(device)
        mask = obs["action_mask"].to(device)
        critic = obs.get("critic_feats")
        if critic is not None:
            critic = critic.to(device)

        with torch.no_grad():
            logits, value = model(cand, glob, mask, critic)
            ranking_action = make_ranking_action(logits, mask, sample=sample)

        if value.shape != (2,):
            raise RuntimeError(f"Expected two critic heads [2], got {tuple(value.shape)}")
        values.append((float(value[0].item()), float(value[1].item())))
        feasible_counts.append(int(obs["action_mask"].sum().item()))

        step_out = env.step(
            ranking_action.solver_order_indices[0],
            action_order_indices=ranking_action.solver_order_indices,
        )
        node_ids.append(step_out.info.get("node_id"))
        depths.append(int(step_out.info.get("depth", -1)))

        best = step_out.info.get("best_makespan")
        if best is not None and (previous_incumbent is None or best < previous_incumbent):
            step = len(node_ids) - 1
            incumbent_steps.append(step)
            if first_incumbent_step is None:
                first_incumbent_step = step
            previous_incumbent = int(best)

        if step_out.done:
            break
        obs = step_out.observation

    return Rollout(
        node_ids=node_ids,
        values=values,
        depths=depths,
        feasible_counts=feasible_counts,
        incumbent_steps=incumbent_steps,
        first_incumbent_step=first_incumbent_step,
    )


def make_bootstrap_boundaries(
    *,
    model,
    env: BranchingEnv,
    tree: Mapping[str, Any],
    alpha: float,
    cost_mean: float,
    cost_std: float,
    bonus_mean: float,
    bonus_std: float,
    device: torch.device,
) -> tuple[Dict[int, float], Dict[int, float], FrontierReport]:
    """Mirror the training loop's exact/critic frontier-completion logic."""
    cost_values: Dict[int, float] = {}
    bonus_values: Dict[int, float] = {}
    report = FrontierReport()
    frontier_nodes = list(tree.get("frontier_nodes", []))
    report.total = len(frontier_nodes)
    final_incumbent = tree.get("final_incumbent")
    final_frontier_lb = tree.get("final_frontier_min_lb")

    pending_ids = {
        int(node["id"]) for node in tree.get("nodes", [])
        if node.get("status") == "pending"
    }
    frontier_ids = {int(node.node_id) for node in frontier_nodes}
    report.pending_not_on_frontier = len(pending_ids - frontier_ids)

    for node in frontier_nodes:
        nid = int(node.node_id)
        if final_incumbent is not None and node.lower_bound >= int(final_incumbent):
            cost_values[nid] = 0.0
            bonus_values[nid] = 0.0
            report.exact_incumbent_pruned += 1
            continue

        # A complete schedule remains intentionally unresolved: its potential
        # incumbent effect on later DFS siblings cannot be a local boundary.
        if not node.unscheduled:
            report.unresolved_complete_schedule += 1
            continue

        # No ready task means solver continuation is an exact prune without an
        # expansion and has zero continuation return in both heads.
        if not node.ready:
            cost_values[nid] = 0.0
            bonus_values[nid] = 0.0
            report.exact_dead_end += 1
            continue

        bootstrap_obs = env.observation_for_bootstrap(
            node,
            incumbent=None if final_incumbent is None else int(final_incumbent),
            frontier_min_lb=None if final_frontier_lb is None else int(final_frontier_lb),
            stack_size=len(frontier_nodes),
        )
        if int(bootstrap_obs["action_mask"].sum().item()) == 0:
            # Solver will expand the node once, discover no feasible child, and
            # terminate that local branch. This is an exact one-node cost.
            cost_values[nid] = -float(alpha)
            bonus_values[nid] = 0.0
            report.exact_all_infeasible += 1
            continue

        with torch.no_grad():
            _, normalized_values = model(
                bootstrap_obs["candidate_feats"].to(device),
                bootstrap_obs["global_feats"].to(device),
                bootstrap_obs["action_mask"].to(device),
                bootstrap_obs["critic_feats"].to(device),
            )
        if normalized_values.shape != (2,):
            raise RuntimeError("Expected two-head critic values at bootstrap frontier.")
        cost_values[nid] = float(normalized_values[0].item()) * cost_std + cost_mean
        bonus_values[nid] = float(normalized_values[1].item()) * bonus_std + bonus_mean
        report.critic_estimated += 1

    return cost_values, bonus_values, report


def build_targets(
    *,
    tree: Mapping[str, Any],
    rollout_data: Rollout,
    scale: RewardScale,
    cost_boundary_values: Optional[Mapping[int, float]] = None,
    bonus_boundary_values: Optional[Mapping[int, float]] = None,
) -> TreeAdvantages:
    cost_fn = make_cost_reward_fn(alpha=scale.alpha)
    bonus_fn = make_bonus_reward_fn(
        tree,
        beta1=scale.beta1,
        beta2=scale.beta2,
        root_lb=tree.get("root_lb"),
    )
    return compute_episode_advantages_decoupled(
        tree=tree,
        node_ids=rollout_data.node_ids,
        cost_reward_fn=cost_fn,
        bonus_reward_fn=bonus_fn,
        gamma_cost=scale.gamma_cost,
        gamma_bonus=scale.gamma_bonus,
        keep_open=False,
        cost_boundary_values=cost_boundary_values,
        bonus_boundary_values=bonus_boundary_values,
    )


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def explained_variance(targets: np.ndarray, predictions: np.ndarray) -> float:
    if targets.size == 0 or np.var(targets) == 0.0:
        return float("nan")
    return float(1.0 - np.var(targets - predictions) / np.var(targets))


def target_metrics(
    targets: TreeAdvantages,
    value_cost_raw: np.ndarray,
    value_bonus_raw: np.ndarray,
) -> tuple[int, float, float, float]:
    if targets.cost_returns is None or targets.bonus_returns is None:
        raise RuntimeError("Expected separate cost and bonus returns.")
    valid = np.asarray(targets.valid, dtype=bool)
    cost = np.asarray(targets.cost_returns, dtype=np.float64)[valid]
    bonus = np.asarray(targets.bonus_returns, dtype=np.float64)[valid]
    pred_cost = value_cost_raw[valid]
    pred_bonus = value_bonus_raw[valid]
    return (
        int(valid.sum()),
        explained_variance(cost, pred_cost),
        explained_variance(bonus, pred_bonus),
        explained_variance(cost + bonus, pred_cost + pred_bonus),
    )


def depth_band(depth: int, n_activities: int, n_bands: int) -> int:
    return min(
        n_bands - 1,
        int(max(0.0, depth / float(max(1, n_activities))) * n_bands),
    )


def count_by_depth(
    indices: List[int], rollout_data: Rollout, n_activities: int, n_bands: int
) -> List[int]:
    counts = [0] * n_bands
    for i in indices:
        counts[depth_band(rollout_data.depths[i], n_activities, n_bands)] += 1
    return counts


def print_depth_table(
    *,
    drop_targets: TreeAdvantages,
    bootstrap_targets: TreeAdvantages,
    rollout_data: Rollout,
    n_activities: int,
    n_bands: int,
    actor_retained: Optional[List[int]],
    singleton_retained: Optional[List[int]],
) -> None:
    drop_valid = [i for i, ok in enumerate(drop_targets.valid) if ok]
    bootstrap_valid = [i for i, ok in enumerate(bootstrap_targets.valid) if ok]
    recovered = [
        i for i, ok in enumerate(bootstrap_targets.valid)
        if ok and not drop_targets.valid[i]
    ]
    actor = [
        i for i, ok in enumerate(bootstrap_targets.valid)
        if ok and rollout_data.feasible_counts[i] > 1
    ]
    singleton = [
        i for i, ok in enumerate(bootstrap_targets.valid)
        if ok and rollout_data.feasible_counts[i] == 1
    ]
    rows: List[Tuple[str, List[int]]] = [
        ("drop valid", drop_valid),
        ("bootstrap valid", bootstrap_valid),
        ("recovered by bootstrap", recovered),
        ("actor eligible (>1)", actor),
        ("singleton critic (=1)", singleton),
    ]
    if actor_retained is not None and singleton_retained is not None:
        rows.extend([
            ("actor retained", actor_retained),
            ("singleton retained", singleton_retained),
        ])

    print("  [Eligibility / retained rows by relative depth]")
    print("  " + f"{'row':<27}" + "".join(f"d{d:>7}" for d in range(n_bands)) + f"{'total':>9}")
    print("  " + "-" * (27 + n_bands * 9 + 9))
    for name, indices in rows:
        counts = count_by_depth(indices, rollout_data, n_activities, n_bands)
        print("  " + f"{name:<27}" + "".join(f"{count:>9}" for count in counts) + f"{sum(counts):>9}")
    print()


def bucket_indices(depths: List[int], valid: List[bool], k: int) -> Dict[str, List[int]]:
    valid_idx = [i for i, is_valid in enumerate(valid) if is_valid]
    if not valid_idx:
        return {"top": [], "mid": [], "bottom": []}
    ordered = sorted(valid_idx, key=lambda i: depths[i])
    n = len(ordered)
    mid_start = max(0, n // 2 - k // 2)
    return {"top": ordered[:k], "mid": ordered[mid_start:mid_start + k], "bottom": ordered[-k:]}


def print_node_examples(
    *,
    rollout_data: Rollout,
    drop_targets: TreeAdvantages,
    bootstrap_targets: TreeAdvantages,
    value_cost_raw: np.ndarray,
    value_bonus_raw: np.ndarray,
    max_print: int,
) -> None:
    if bootstrap_targets.cost_returns is None or bootstrap_targets.bonus_returns is None:
        return
    buckets = bucket_indices(rollout_data.depths, bootstrap_targets.valid, max_print)
    header = (
        f"  {'node':>7} {'depth':>5} {'F':>4} {'target':>10} "
        f"{'Vc(raw)':>10} {'Vb(raw)':>10} | {'Gc':>10} {'Gb':>10} {'G':>10}"
    )
    print("  [Bootstrap-valid node examples]")
    for label in ("top", "mid", "bottom"):
        print(f"  {label.upper()} ({len(buckets[label])} rows)")
        print(header)
        print("  " + "-" * (len(header) - 2))
        for i in buckets[label]:
            target_origin = "closed" if drop_targets.valid[i] else "recovered"
            gc = bootstrap_targets.cost_returns[i]
            gb = bootstrap_targets.bonus_returns[i]
            print(
                f"  {str(rollout_data.node_ids[i]):>7} {rollout_data.depths[i]:>5} "
                f"{rollout_data.feasible_counts[i]:>4} {target_origin:>10} "
                f"{value_cost_raw[i]:>+10.3f} {value_bonus_raw[i]:>+10.3f} | "
                f"{gc:>+10.3f} {gb:>+10.3f} {gc + gb:>+10.3f}"
            )
        print()


def main() -> None:
    args = parse_args()
    device = torch.device(
        "cpu" if (args.device == "cuda" and not torch.cuda.is_available()) else args.device
    )
    checkpoint = _load_checkpoint_payload(args.checkpoint, str(device))
    config = resolve_config(args, checkpoint)
    seed = int(args.seed if args.seed is not None else config["seed"])
    if args.sample:
        torch.manual_seed(seed)
        np.random.seed(seed)

    inst_path = Path(args.instance)
    instance = load_instance(inst_path)
    scale = resolve_reward_scale(
        args=args,
        config=config,
        instance=instance,
        instance_name=inst_path.name,
        device=device,
    )
    model = load_policy_checkpoint(args.checkpoint, device=device)
    model.eval()
    (cost_mean, cost_std), (bonus_mean, bonus_std) = load_value_norm(checkpoint)
    env = BranchingEnv(
        instance_source=instance,
        max_resources=int(config["max_resources"]),
        time_limit_s=float(config["time_limit_s"]),
        dominance=str(config["dominance"]),
    )

    rollout_data = rollout(model, env, instance, device, args.sample)
    stats = env.episode_stats
    tree = env.search_tree()
    if tree is None:
        raise RuntimeError("Rollout finished without a search tree.")

    # Both constructions use exactly the same rollout values, tree, and rewards.
    drop_targets = build_targets(tree=tree, rollout_data=rollout_data, scale=scale)
    cost_boundaries: Dict[int, float] = {}
    bonus_boundaries: Dict[int, float] = {}
    frontier_report = FrontierReport()
    if stats.done_reason == "time_limit":
        cost_boundaries, bonus_boundaries, frontier_report = make_bootstrap_boundaries(
            model=model, env=env, tree=tree, alpha=scale.alpha,
            cost_mean=cost_mean, cost_std=cost_std,
            bonus_mean=bonus_mean, bonus_std=bonus_std, device=device,
        )
    bootstrap_targets = build_targets(
        tree=tree, rollout_data=rollout_data, scale=scale,
        cost_boundary_values=cost_boundaries,
        bonus_boundary_values=bonus_boundaries,
    )

    values = np.asarray(rollout_data.values, dtype=np.float64)
    value_cost_raw = values[:, 0] * cost_std + cost_mean
    value_bonus_raw = values[:, 1] * bonus_std + bonus_mean
    d_valid, d_ec, d_eb, d_ecomb = target_metrics(drop_targets, value_cost_raw, value_bonus_raw)
    b_valid, b_ec, b_eb, b_ecomb = target_metrics(bootstrap_targets, value_cost_raw, value_bonus_raw)
    recovered = sum(b and not d for b, d in zip(bootstrap_targets.valid, drop_targets.valid))
    unresolved = sum(not ok for ok in bootstrap_targets.valid)

    actor_retained: Optional[List[int]] = None
    singleton_retained: Optional[List[int]] = None
    if not args.no_subsample:
        staged = StagedEpisode(
            values=[(0.0, 0.0)] * len(rollout_data.node_ids),
            depths=list(rollout_data.depths),
            feasible_counts=list(rollout_data.feasible_counts),
            incumbent_steps=list(rollout_data.incumbent_steps),
            first_incumbent_step=rollout_data.first_incumbent_step,
        )
        sampler_seed = int(args.subsample_seed if args.subsample_seed is not None else seed + 1)
        sampler_rng = random.Random(sampler_seed)
        actor_retained, _ = subsample_episode(
            staged, bootstrap_targets.valid,
            cap=(None if config["episode_transition_cap"] is None else int(config["episode_transition_cap"])),
            n_activities=len(instance.activities),
            time_bands=int(config["stratify_time_bands"]),
            depth_bands=int(config["stratify_depth_bands"]),
            incumbent_window=int(config["incumbent_window"]),
            rng=sampler_rng,
        )
        singleton_retained = subsample_critic_singletons(
            staged, bootstrap_targets.valid,
            cap=(None if config.get("critic_singleton_transition_cap") is None else int(config["critic_singleton_transition_cap"])),
            n_activities=len(instance.activities),
            time_bands=int(config["stratify_time_bands"]),
            depth_bands=int(config["stratify_depth_bands"]),
            rng=sampler_rng,
        )

    sep = "=" * 88
    print(f"\n{sep}\n  CRITIC-BOOTSTRAP DEBUG (frozen, one rollout) — {inst_path.name}\n{sep}")
    print(f"  done_reason       : {stats.done_reason}")
    print(f"  nodes_expanded    : {stats.nodes_expanded}")
    print(f"  best_makespan     : {stats.best_makespan}")
    print(f"  transitions       : {len(rollout_data.node_ids)}")
    print(f"  action mode       : {'sampled (fresh seed=%d)' % seed if args.sample else 'greedy'}")
    print(f"  alpha             : {scale.alpha:.8g}  ({scale.source})")
    if scale.n_ref is not None:
        n_label = "N_ref" if scale.source == "oracle" else "N_est"
        print(f"  {n_label:<18}: {scale.n_ref:.0f}")
    if scale.oracle_path is not None:
        print(f"  oracle            : {scale.oracle_path}  episode={args.episode}")
    print(f"  rewards/gammas    : beta1={scale.beta1:g} beta2={scale.beta2:g}  "
          f"cost={scale.gamma_cost:g} bonus={scale.gamma_bonus:g}")
    print(f"  value_norm cost   : mean={cost_mean:.5f} std={cost_std:.5f}")
    print(f"  value_norm bonus  : mean={bonus_mean:.5f} std={bonus_std:.5f}\n")

    print("  [Truncation comparison — identical rollout tree]")
    print("  " + f"{'target':<20}{'valid':>9}{'ev_cost':>11}{'ev_bonus':>11}{'ev_combined':>14}")
    print("  " + "-" * 65)
    print(f"  {'drop':<20}{d_valid:>9}{d_ec:>+11.3f}{d_eb:>+11.3f}{d_ecomb:>+14.3f}")
    print(f"  {'critic_bootstrap':<20}{b_valid:>9}{b_ec:>+11.3f}{b_eb:>+11.3f}{b_ecomb:>+14.3f}")
    print(f"  {'recovered by bootstrap':<20}{recovered:>9}")
    print(f"  {'still unresolved':<20}{unresolved:>9}\n")

    print("  [Frontier completion]")
    print(f"  frontier total                 : {frontier_report.total}")
    print(f"  exact incumbent-pruned (zero) : {frontier_report.exact_incumbent_pruned}")
    print(f"  exact dead-end (zero)         : {frontier_report.exact_dead_end}")
    print(f"  exact all-infeasible (-alpha) : {frontier_report.exact_all_infeasible}")
    print(f"  critic-estimated               : {frontier_report.critic_estimated}")
    print(f"  unresolved complete schedules  : {frontier_report.unresolved_complete_schedule}")
    print(f"  pending not on final frontier  : {frontier_report.pending_not_on_frontier}\n")

    print_depth_table(
        drop_targets=drop_targets, bootstrap_targets=bootstrap_targets,
        rollout_data=rollout_data, n_activities=len(instance.activities),
        n_bands=int(config["stratify_depth_bands"]),
        actor_retained=actor_retained, singleton_retained=singleton_retained,
    )
    print_node_examples(
        rollout_data=rollout_data, drop_targets=drop_targets,
        bootstrap_targets=bootstrap_targets, value_cost_raw=value_cost_raw,
        value_bonus_raw=value_bonus_raw, max_print=args.max_print,
    )

    if recovered:
        print("  Interpretation: bootstrap mechanically restored targets for the "
              "recovered rows above. Their target quality still depends on the "
              "two critic continuation estimates; the EVs diagnose that separately.")
    elif stats.done_reason != "time_limit":
        print("  Interpretation: search exhausted, so there was no truncation credit to recover.")
    else:
        print("  Interpretation: no rows were recovered. Inspect unresolved complete schedules "
              "and pending-not-on-frontier above.")
    print(f"{sep}\n")


if __name__ == "__main__":
    main()
