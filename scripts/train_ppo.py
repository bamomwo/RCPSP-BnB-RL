"""Actor-only censored pairwise fine-tuning (historical CLI filename retained)."""
from __future__ import annotations

import argparse
import copy
import json
import math
import random
import sys
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch.distributions import Categorical

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rcpsp_bnb_rl.bnb.branching_order import make_order_fn
from rcpsp_bnb_rl.bnb.lower_bounds import lower_bound
from rcpsp_bnb_rl.bnb.solver import BnBSolver
from rcpsp_bnb_rl.data.dataset import list_instance_paths
from rcpsp_bnb_rl.data.parsing import load_instance
from rcpsp_bnb_rl.ml.action_order import selected_first_policy_order
from rcpsp_bnb_rl.ml.il.featurize import (
    candidate_feature_dim, global_feature_dim, set_candidate_feature_dim,
    set_global_feature_dim, set_interaction_feature_dim, set_resource_feature_dim,
)
from rcpsp_bnb_rl.ml.models import (
    BranchingTransformer, load_actor_checkpoint, save_actor_checkpoint,
)
from rcpsp_bnb_rl.ml.rl import BranchingEnv
from rcpsp_bnb_rl.ml.rl.tree_return import (
    PreferencePair, categorical_policy_kl, extract_valid_preference_pairs,
    ranknet_pairwise_loss,
)


@dataclass
class SplitRecord:
    observation: dict[str, torch.Tensor]
    old_logits: torch.Tensor
    pairs: list[PreferencePair]
    winner_slots: list[int]
    loser_slots: list[int]
    depth: int


def required_contributors(config: Mapping[str, Any], n_train_instances: int) -> int:
    return min(int(config["min_episodes"]), n_train_instances)


@dataclass
class BatchAccumulator:
    records: list[SplitRecord] = field(default_factory=list)
    diagnostics: Counter = field(default_factory=Counter)
    contributor_instances: set[Path] = field(default_factory=set)
    episodes: int = 0
    contributing_episodes: int = 0

    def add(self, path: Path, records: list[SplitRecord], diagnostics: Mapping) -> None:
        self.episodes += 1
        self.diagnostics.update({key: value for key, value in diagnostics.items()
                                 if isinstance(value, (int, float))})
        if records:
            self.records.extend(records)
            self.contributor_instances.add(path)
            self.contributing_episodes += 1
        depth_counts = Counter(record.depth for record in self.records)
        reported_depths = [int(key[1:-len("_total_split")]) for key in self.diagnostics
                           if key.startswith("d") and key.endswith("_total_split")
                           and key[1:-len("_total_split")].isdigit()]
        max_depth = max([*depth_counts, *reported_depths], default=-1)
        for depth in range(max_depth + 1):
            self.diagnostics.setdefault(f"d{depth}_total_split", 0)
            self.diagnostics[f"d{depth}_active_split"] = depth_counts[depth]

    def ready(self, config: Mapping[str, Any], n_train_instances: int) -> bool:
        return (len(self.records) >= config["min_batch_size"]
                and len(self.contributor_instances) >= required_contributors(config, n_train_instances))


def balanced_minibatches(n: int, size: int, *, shuffle: bool = False) -> list[list[int]]:
    """Keep batches below the size limit without a tiny remainder batch."""
    if n == 0:
        return []
    indices = list(range(n))
    if shuffle:
        random.shuffle(indices)
    count = math.ceil(n / size)
    width, extra = divmod(n, count)
    batches = []
    start = 0
    for index in range(count):
        end = start + width + int(index < extra)
        batches.append(indices[start:end])
        start = end
    return batches


def batch_observations(observations: Sequence[dict], device: torch.device) -> dict:
    """Pad candidates and resource sets independently, keeping feasibility masks."""
    batch_size = len(observations)
    max_candidates = max(len(obs["action_mask"]) for obs in observations)
    first = observations[0]
    kwargs = {
        "candidate_feats": torch.zeros(batch_size, max_candidates, first["candidate_feats"].shape[-1]),
        "global_feats": torch.stack([obs["global_feats"] for obs in observations]),
        "action_mask": torch.zeros(batch_size, max_candidates, dtype=torch.bool),
        "pad_mask": torch.zeros(batch_size, max_candidates, dtype=torch.bool),
    }
    if "resource_feats" in first:
        max_resources = max(len(obs["resource_feats"]) for obs in observations)
        kwargs.update(
            resource_feats=torch.zeros(batch_size, max_resources, first["resource_feats"].shape[-1]),
            candidate_resource_feats=torch.zeros(
                batch_size, max_candidates, max_resources, first["candidate_resource_feats"].shape[-1]
            ),
            resource_mask=torch.zeros(batch_size, max_resources, dtype=torch.bool),
        )
    for row, obs in enumerate(observations):
        count = len(obs["action_mask"])
        kwargs["candidate_feats"][row, :count] = obs["candidate_feats"]
        kwargs["action_mask"][row, :count] = obs["action_mask"]
        kwargs["pad_mask"][row, :count] = True
        if "resource_feats" in first:
            resources = len(obs["resource_feats"])
            kwargs["resource_feats"][row, :resources] = obs["resource_feats"]
            kwargs["candidate_resource_feats"][row, :count, :resources] = obs["candidate_resource_feats"]
            kwargs["resource_mask"][row, :resources] = obs["resource_mask"]
    return {name: tensor.to(device) for name, tensor in kwargs.items()}


def batch_pairs(records: Sequence[SplitRecord], device: torch.device) -> tuple:
    max_pairs = max(len(record.pairs) for record in records)
    winners = torch.zeros(len(records), max_pairs, dtype=torch.long, device=device)
    losers = torch.zeros_like(winners)
    weights = torch.zeros(len(records), max_pairs, device=device)
    mask = torch.zeros_like(winners, dtype=torch.bool)
    for row, record in enumerate(records):
        count = len(record.pairs)
        winners[row, :count] = torch.tensor(record.winner_slots, device=device)
        losers[row, :count] = torch.tensor(record.loser_slots, device=device)
        weights[row, :count] = torch.tensor([pair.weight for pair in record.pairs], device=device)
        mask[row, :count] = True
    return winners, losers, weights, mask


def old_logits_batch(records: Sequence[SplitRecord], width: int, device: torch.device) -> torch.Tensor:
    result = torch.full((len(records), width), -1e9, device=device)
    for row, record in enumerate(records):
        result[row, :len(record.old_logits)] = record.old_logits.to(device)
    return result


def collect_episode(actor: BranchingTransformer, env: BranchingEnv, device: torch.device,
                    config: Mapping[str, Any]) -> tuple[list[SplitRecord], dict, int]:
    """Collect complete dives, then label explored siblings from the exported tree."""
    actor.eval()
    observation = env.reset()
    choices = {}
    steps = 0
    while True:
        feasible = observation["action_mask"]
        if int(feasible.sum()) > 1:
            with torch.no_grad():
                logits = actor.forward_batch(**batch_observations([observation], device))[0][0]
                action = int(Categorical(logits=logits).sample())
            order = selected_first_policy_order(logits.cpu().tolist(), action)
            old_logits = logits.detach().cpu()
        else:
            action = int(torch.nonzero(feasible)[0]) if bool(feasible.any()) else 0
            order = [action] + [i for i in range(len(feasible)) if i != action]
            old_logits = None
        output = env.step(action, action_order_indices=order)
        steps += 1
        if old_logits is not None:
            choices[int(output.info["node_id"])] = (observation, old_logits)
        if output.done:
            break
        observation = output.observation

    tree = env.search_tree()
    if tree is None:
        raise RuntimeError("Finished episode did not export its search tree")
    preferences = extract_valid_preference_pairs(
        tree, preference_mode=config["preference_mode"],
        pair_gap_alpha=float(config["pair_gap_alpha"]),
        tie_break_weight=float(config["tie_break_weight"]),
    )
    nodes = {int(node["id"]): node for node in tree["nodes"]}
    records = []
    for node_id, pairs in preferences.split_pairs.items():
        if not pairs:
            continue
        observation, old_logits = choices[node_id]
        slots = {int(activity): slot for slot, activity in enumerate(observation["ready_ids"])}
        winners = [slots[pair.winner_activity] for pair in pairs]
        losers = [slots[pair.loser_activity] for pair in pairs]
        if not all(bool(observation["action_mask"][slot]) for slot in winners + losers):
            raise RuntimeError("Preference refers to an infeasible action")
        records.append(SplitRecord(observation, old_logits, pairs, winners, losers, int(nodes[node_id]["depth"])))
    diagnostics = asdict(preferences.diagnostics)
    # Count genuine split opportunities, including those yielding no usable pair.
    max_depth = max(len(env.instance.activities) - 1,
                    max((int(nodes[node_id]["depth"]) for node_id in choices), default=-1))
    for depth in range(max_depth + 1):
        diagnostics[f"d{depth}_total_split"] = 0
        diagnostics[f"d{depth}_active_split"] = 0
    for node_id, (obs, _) in choices.items():
        depth = int(nodes[node_id]["depth"])
        diagnostics[f"d{depth}_total_split"] += 1
        diagnostics[f"d{depth}_active_split"] += int(bool(preferences.split_pairs.get(node_id)))
    diagnostics["n_split_nodes"] = len(choices)
    diagnostics["n_active_split_nodes"] = len(records)
    diagnostics["nodes_expanded"] = env.episode_stats.nodes_expanded
    return records, diagnostics, steps


def policy_diagnostics(actor: BranchingTransformer, records: list[SplitRecord],
                       bc_actor: BranchingTransformer | None, device: torch.device,
                       minibatch_size: int) -> dict[str, float]:
    actor.eval()
    totals = Counter()
    correct = Counter()
    loss_by_type = Counter()
    kl_old = kl_bc = entropy = node_loss_sum = 0.0
    with torch.no_grad():
        for indices in balanced_minibatches(len(records), minibatch_size):
            minibatch = [records[index] for index in indices]
            kwargs = batch_observations([record.observation for record in minibatch], device)
            logits = actor.forward_batch(**kwargs)[0]
            old = old_logits_batch(minibatch, logits.shape[1], device)
            kl_old += float(categorical_policy_kl(old, logits, kwargs["action_mask"]).sum())
            entropy += float(Categorical(logits=logits).entropy().sum())
            if bc_actor is not None:
                bc_logits = bc_actor.forward_batch(**kwargs)[0]
                kl_bc += float(categorical_policy_kl(bc_logits, logits, kwargs["action_mask"]).sum())
            node_loss, _ = ranknet_pairwise_loss(logits, *batch_pairs(minibatch, device))
            node_loss_sum += float(node_loss) * len(minibatch)
            for row, record in enumerate(minibatch):
                for pair, winner, loser in zip(record.pairs, record.winner_slots, record.loser_slots):
                    difference = logits[row, winner] - logits[row, loser]
                    totals[pair.pair_type] += 1
                    correct[pair.pair_type] += int(difference > 0)
                    loss_by_type[pair.pair_type] += float(pair.weight * torch.nn.functional.softplus(-difference))
    count = len(records)
    result = {
        "rank_accuracy": sum(correct.values()) / max(1, sum(totals.values())),
        "ranknet_loss": node_loss_sum / max(1, count),
        "kl_old": kl_old / max(1, count),
        "entropy": entropy / max(1, count),
    }
    if bc_actor is not None:
        result["kl_bc"] = kl_bc / max(1, count)
    for kind in ("certified", "empirical", "tie_break_bound"):
        result[f"accuracy_{kind}"] = correct[kind] / max(1, totals[kind])
        result[f"loss_{kind}"] = loss_by_type[kind] / max(1, totals[kind])
    return result


def train_update(actor: BranchingTransformer, optimizer: torch.optim.Optimizer,
                 records: list[SplitRecord], bc_actor: BranchingTransformer | None,
                 device: torch.device, config: Mapping[str, Any]) -> dict[str, Any]:
    """Per-split RankNet with exact categorical KL monitoring against rollout logits."""
    if not records:
        return {"skipped_no_pairs": True, "optimizer_steps": 0}
    actor.eval()  # Dropout must not alter the replay distribution or KL reference.
    size = int(config["target_mb_size"])
    before = policy_diagnostics(actor, records, bc_actor, device, size)
    metrics = {f"before_{key}": value for key, value in before.items()}
    steps = 0
    stopped = False
    epochs_completed = 0
    max_minibatch_kl = 0.0
    batch_sizes = []
    for epoch in range(int(config["epochs"])):
        epoch_finished = True
        for indices in balanced_minibatches(len(records), size, shuffle=True):
            minibatch = [records[index] for index in indices]
            kwargs = batch_observations([record.observation for record in minibatch], device)
            logits = actor.forward_batch(**kwargs)[0]
            old = old_logits_batch(minibatch, logits.shape[1], device)
            kl = categorical_policy_kl(old, logits, kwargs["action_mask"]).mean()
            max_minibatch_kl = max(max_minibatch_kl, float(kl.detach()))
            if float(kl.detach()) > float(config["target_kl"]):
                stopped = True
                epoch_finished = False
                break
            rank_loss, _ = ranknet_pairwise_loss(logits, *batch_pairs(minibatch, device))
            entropy = Categorical(logits=logits).entropy().mean()
            loss = rank_loss - float(config["entropy_coef"]) * entropy
            if float(config["bc_kl_coef"]) > 0:
                with torch.no_grad():
                    reference = bc_actor.forward_batch(**kwargs)[0]
                loss = loss + float(config["bc_kl_coef"]) * categorical_policy_kl(
                    reference, logits, kwargs["action_mask"]
                ).mean()
            if not bool(torch.isfinite(loss)):
                raise RuntimeError("Non-finite RankNet objective")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(actor.parameters(), float(config["max_grad_norm"]))
            optimizer.step()
            steps += 1
            batch_sizes.append(len(minibatch))
            with torch.no_grad():
                updated = actor.forward_batch(**kwargs)[0]
                updated_kl = float(categorical_policy_kl(old, updated, kwargs["action_mask"]).mean())
                max_minibatch_kl = max(max_minibatch_kl, updated_kl)
                if updated_kl > float(config["target_kl"]):
                    stopped = True
                    epoch_finished = False
                    break
        epochs_completed += int(epoch_finished)
        after = policy_diagnostics(actor, records, bc_actor, device, size)
        if stopped or after["kl_old"] > float(config["target_kl"]):
            stopped = True
            break
    metrics.update({f"after_{key}": value for key, value in after.items()})
    metrics.update(optimizer_steps=steps, epochs_completed=epochs_completed,
                   epochs_attempted=epoch + 1, kl_early_stop=stopped,
                   max_minibatch_kl=max_minibatch_kl,
                   minibatch_size_min=min(batch_sizes, default=0),
                   minibatch_size_max=max(batch_sizes, default=0))
    return metrics


def evaluate(model: BranchingTransformer, instance_paths: Sequence[Path], *,
             config: Mapping[str, Any], device: torch.device,
             optimal_makespans: Mapping[str, int]) -> dict:
    """Measure primal progress and proof effort under identical solver budgets."""
    model.eval()
    rows = []
    for path in instance_paths:
        instance = load_instance(path)
        solver = BnBSolver(instance)
        order_fn = make_order_fn(
            "policy", instance=instance, model=model, max_resources=int(config["max_resources"]),
            resource_encoder="set" if model.resource_set_enabled else "legacy_flat",
            device=device, predecessors=solver.predecessors,
        )
        root_lb = lower_bound(instance, set(instance.activities), {})
        started = time.perf_counter()
        result = solver.solve(
            order_ready_fn=order_fn, dominance="set_based", memory_bounded=True,
            time_limit_s=config["eval_time_limit_s"], max_nodes=config["eval_max_nodes"],
            stop_at_dive_boundary=True, external_lower_bound=root_lb,
        )
        elapsed = time.perf_counter() - started
        optimum = optimal_makespans.get(path.stem.lower())
        targets = config["eval_target_makespans"]
        target = targets.get(path.stem.lower(), optimum)
        proven = result.best_makespan is not None and result.done_reason in {"search_exhausted", "external_bound_matched"}
        history = [
            {"nodes_expanded": event.nodes_expanded, "makespan": event.makespan,
             "elapsed_s": event.elapsed_s,
             "gap_to_root_bound": (event.makespan - root_lb) / max(1, root_lb),
             "primal_gap": ((event.makespan - optimum) / max(1, optimum) if optimum is not None else None)}
            for event in result.incumbent_history
        ]
        rows.append({
            "instance": str(path), "makespan": result.best_makespan,
            "root_lower_bound": root_lb,
            "gap_to_root_bound": ((result.best_makespan - root_lb) / max(1, root_lb)
                                  if result.best_makespan is not None else None),
            "optimal_makespan": optimum, "target_makespan": target,
            "primal_gap": ((result.best_makespan - optimum) / max(1, optimum)
                           if result.best_makespan is not None and optimum is not None else None),
            "nodes_expanded": result.nodes_expanded,
            "nodes_to_target": next((event["nodes_expanded"] for event in history
                                     if target is not None and event["makespan"] <= target), None),
            "nodes_to_proof": result.nodes_expanded if proven else None,
            "proven_optimal": proven, "done_reason": result.done_reason,
            "runtime_s": elapsed, "incumbent_history": history,
        })
    def mean(field):
        values = [row[field] for row in rows if row[field] is not None]
        return sum(values) / len(values) if values else None
    n = len(rows)
    target_count = sum(row["target_makespan"] is not None for row in rows)
    summary = {
        "n_instances": n, "proof_fraction": sum(row["proven_optimal"] for row in rows) / max(1, n),
        "feasible_fraction": sum(row["makespan"] is not None for row in rows) / max(1, n),
        "n_target_instances": target_count,
        "n_targets_reached": sum(row["nodes_to_target"] is not None for row in rows),
        "n_proofs": sum(row["proven_optimal"] for row in rows),
        "n_primal_gaps": sum(row["primal_gap"] is not None for row in rows),
        "target_fraction": (sum(row["nodes_to_target"] is not None for row in rows) / target_count
                            if target_count else None),
        "mean_primal_gap": mean("primal_gap"), "mean_nodes_expanded": mean("nodes_expanded"),
        "mean_gap_to_root_bound": mean("gap_to_root_bound"),
        "mean_nodes_to_target": mean("nodes_to_target"), "mean_nodes_to_proof": mean("nodes_to_proof"),
        "mean_runtime_s": mean("runtime_s"),
    }
    return {"summary": summary, "instances": rows, "dominance": "set_based"}


DEFAULT_CONFIG = {
    "root": "data/train/1kNetRes", "pattern": "*.rcp", "max_instances": None,
    "max_resources": 4, "resource_encoder": "set",
    "bc_checkpoint": "models/policy_bc.pt", "d_model": 64, "n_heads": 4,
    "n_layers": 2, "ffn_dim": 128,
    "total_env_steps": 1_000_000, "epochs": 2, "target_mb_size": 32,
    "min_batch_size": 64, "min_episodes": 4,
    "preference_mode": "all", "pair_gap_alpha": 1.0, "tie_break_weight": 0.5,
    "entropy_coef": 0.002, "bc_kl_coef": 0.0, "actor_lr": 5e-5,
    "max_grad_norm": 0.5, "target_kl": 0.02,
    "stop_at_root_lb": True,
    "time_limit_s": 300.0, "max_nodes": 20000,
    "eval_every_updates": 1, "eval_root": None, "eval_pattern": "*.rcp",
    "eval_time_limit_s": 60.0, "eval_max_nodes": None, "eval_optimal_json": None,
    "eval_target_makespans": {},
    "save_path": "models/pairwise/policy_ranknet.pt", "checkpoint_dir": "models/pairwise/checkpoints",
    "tensorboard": True, "seed": 42, "device": "cpu",
}


def validate_config(config: Mapping[str, Any]) -> None:
    unknown = set(config) - set(DEFAULT_CONFIG)
    if unknown:
        raise ValueError(f"Unknown configuration fields: {sorted(unknown)}")
    if config["preference_mode"] not in {"all", "certified_only"}:
        raise ValueError("preference_mode must be 'all' or 'certified_only'")
    if config["resource_encoder"] not in {"set", "legacy_flat"}:
        raise ValueError("resource_encoder must be 'set' or 'legacy_flat'")
    for name in ("total_env_steps", "epochs", "target_mb_size", "min_batch_size", "min_episodes", "eval_every_updates"):
        value = config[name]
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    for name in ("max_nodes", "eval_max_nodes", "max_instances"):
        value = config[name]
        if value is not None and (isinstance(value, bool) or not isinstance(value, int) or value <= 0):
            raise ValueError(f"{name} must be a positive integer or null")
    for name in ("pair_gap_alpha", "tie_break_weight", "entropy_coef", "bc_kl_coef", "actor_lr", "max_grad_norm", "target_kl"):
        value = config[name]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be finite and nonnegative")
    for name in ("actor_lr", "max_grad_norm", "target_kl"):
        if config[name] == 0:
            raise ValueError(f"{name} must be positive")
    if config["bc_kl_coef"] > 0 and not config["bc_checkpoint"]:
        raise ValueError("bc_kl_coef requires a BC checkpoint")
    for name in ("time_limit_s", "eval_time_limit_s"):
        value = config[name]
        if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0):
            raise ValueError(f"{name} must be positive or null")
    if config["max_nodes"] is None and config["time_limit_s"] is None:
        raise ValueError("Training requires a node or time budget")
    if config["eval_max_nodes"] is None and config["eval_time_limit_s"] is None:
        raise ValueError("Evaluation requires a node or time budget")


def _metric(value: float | None, *, percent: bool = False, precision: int = 4) -> str:
    if value is None:
        return "n/a"
    return f"{value * 100:.{precision}f}%" if percent else f"{value:.{precision}f}"


def _node_count(value: int | float | None) -> str:
    return "n/a" if value is None else f"{value:,.0f}"


def print_startup_summary(config: Mapping[str, Any], actor: BranchingTransformer,
                          train_paths: Sequence[Path], eval_paths: Sequence[Path],
                          device: torch.device) -> None:
    print("\n" + "=" * 80)
    print("RankNet Pairwise Training")
    print(f"  Device     : {device} (seed={config['seed']}, "
          f"params={sum(p.numel() for p in actor.parameters()):,}, "
          f"init={config['bc_checkpoint'] or 'from scratch'})")
    print(f"  Dataset    : {config['root']} ({len(train_paths):,} train) | "
          f"{config['eval_root'] or 'disabled'} ({len(eval_paths):,} eval)")
    print(f"  Search     : train_max_nodes={_node_count(config['max_nodes'])} | "
          f"eval_max_nodes={_node_count(config['eval_max_nodes'])} | dominance=set_based")
    print(f"  Collection : min_batch={config['min_batch_size']} splits "
          f"(min_episodes={required_contributors(config, len(train_paths))}, "
          f"total_steps={config['total_env_steps']:,})")
    print(f"  Optimizer  : lr={config['actor_lr']:g} | epochs={config['epochs']} | "
          f"mb_size={config['target_mb_size']} | target_kl={config['target_kl']:g} | "
          f"gap_alpha={config['pair_gap_alpha']}")
    print("=" * 80, flush=True)


def log_episode(episode: int, path: Path, *, env: BranchingEnv,
                global_step: int, elapsed_s: float, episode_records: list[SplitRecord],
                diagnostics: Mapping[str, Any],
                batch: BatchAccumulator, config: Mapping[str, Any], n_train_instances: int) -> None:
    print(f"[Ep {episode}] {path.name} | nodes={env.episode_stats.nodes_expanded:,} | "
          f"cmax={env.episode_stats.best_makespan} | end={env.episode_stats.done_reason} | "
          f"time={elapsed_s:.1f}s | step={global_step:,}/{config['total_env_steps']:,}")
    print(f"  splits: total={diagnostics['n_split_nodes']:,} "
          f"active={len(episode_records)} | "
          f"pairs: cert={diagnostics.get('n_certified', 0)} "
          f"emp={diagnostics.get('n_empirical', 0)} tie={diagnostics.get('n_tie_break', 0)} | "
          f"batch: splits={len(batch.records)}/{config['min_batch_size']} "
          f"inst={len(batch.contributor_instances)}/{required_contributors(config, n_train_instances)}\n",
          flush=True)


def log_update(report: Mapping[str, Any]) -> None:
    print("\n" + "-" * 80)
    print(f"[Update {report['update']}] splits={report['n_retained_split_nodes']} | "
          f"pairs={report['n_retained_pairs']} | instances={report['n_contributing_instances']} | "
          f"collect={report['collection_time_s']:.1f}s | update={report['update_time_s']:.2f}s")
    print(f"  loss: {_metric(report['before_ranknet_loss'])} -> {_metric(report['after_ranknet_loss'])} | "
          f"acc: {_metric(report['before_rank_accuracy'], percent=True, precision=1)} -> "
          f"{_metric(report['after_rank_accuracy'], percent=True, precision=1)} | "
          f"KL_old={_metric(report['after_kl_old'])} | KL_bc={_metric(report.get('after_kl_bc'))} | "
          f"ent={_metric(report['after_entropy'])}")
    types = []
    for short, kind in (("cert", "certified"), ("emp", "empirical"), ("tie", "tie_break_bound")):
        count = report[f"retained_{kind}_pairs"]
        before = _metric(report.get(f"before_accuracy_{kind}") if count else None, percent=True, precision=1)
        after = _metric(report.get(f"after_accuracy_{kind}") if count else None, percent=True, precision=1)
        types.append(f"{short}={count} ({before}->{after})")
    depths = [int(key[1:-len("_active_split")]) for key in report
              if key.startswith("d") and key.endswith("_active_split")
              and key[1:-len("_active_split")].isdigit()]
    depth_text = " ".join(f"d{depth}={report.get(f'd{depth}_active_split', 0)}"
                          for depth in range(max(depths, default=-1) + 1))
    print(f"  pairs: {' '.join(types)} | depth(act): {depth_text or 'none'}")
    print("-" * 80 + "\n", flush=True)


def summarize_evaluation(result: Mapping[str, Any], *, step: int,
                         update: int) -> dict[str, Any]:
    """Compact checkpoint metrics; mean_gap is a percentage, rates are fractions."""
    summary = result["summary"]
    known = [row for row in result["instances"] if row.get("optimal_makespan") is not None]
    matched = sum(row.get("makespan") == row["optimal_makespan"] for row in known)
    gap = summary["mean_primal_gap"]
    return {
        "step": step,
        "update": update,
        "solved": summary["proof_fraction"],
        "optimal_primal": matched / len(known) if known else None,
        "mean_gap": gap * 100 if gap is not None else None,
        "mean_nodes": summary["mean_nodes_expanded"],
    }


def log_evaluation(label: str, result: Mapping[str, Any], *, writer=None,
                   step: int = 0) -> None:
    summary = result["summary"]
    known = [row for row in result["instances"] if row.get("optimal_makespan") is not None]
    matched = sum(row.get("makespan") == row["optimal_makespan"] for row in known)
    matched_text = (f"{matched}/{len(known)} ({100 * matched / len(known):.1f}%)" if known else "n/a")
    print(f"{f'[Eval / {label}]':<22} opt_matched={matched_text} | "
          f"opt_gap={_metric(summary['mean_primal_gap'], percent=True, precision=2)} | "
          f"proofs={summary['n_proofs']}/{summary['n_instances']} | "
          f"nodes={_node_count(summary['mean_nodes_expanded'])} | "
          f"time={_metric(summary['mean_runtime_s'], precision=1)}s", flush=True)
    if writer:
        for key, value in summary.items():
            if isinstance(value, (int, float)) and math.isfinite(value):
                writer.add_scalar(f"eval/{key}", value, step)
        writer.flush()


class _Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            stream.write(data)
            stream.flush()
        return len(data)

    def flush(self):
        for stream in self.streams:
            stream.flush()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--log", action="store_true", help="Tee output next to the saved checkpoint")
    args = parser.parse_args()
    config = {**DEFAULT_CONFIG, **json.loads(Path(args.config).read_text())}
    validate_config(config)
    save_path = Path(config["save_path"])
    save_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = Path(config["checkpoint_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_handle = None
    original_stdout, original_stderr = sys.stdout, sys.stderr
    if args.log:
        log_handle = save_path.with_suffix(".log").open("w")
        sys.stdout = _Tee(sys.stdout, log_handle)
        sys.stderr = _Tee(sys.stderr, log_handle)
    random.seed(config["seed"])
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    rng = random.Random(config["seed"] + 1)
    device = torch.device("cpu" if config["device"] == "cuda" and not torch.cuda.is_available() else config["device"])
    train_paths = list_instance_paths(config["root"], patterns=(config["pattern"],))
    if config["max_instances"] is not None:
        train_paths = train_paths[:config["max_instances"]]
    if not train_paths:
        raise FileNotFoundError(f"No training instances found under {config['root']}")
    eval_paths = list_instance_paths(config["eval_root"], patterns=(config["eval_pattern"],)) if config["eval_root"] else []
    use_set = config["resource_encoder"] == "set"
    if config["bc_checkpoint"]:
        actor = load_actor_checkpoint(config["bc_checkpoint"], device=device, dropout=0.0)
    else:
        actor = BranchingTransformer(
            global_dim=set_global_feature_dim() if use_set else global_feature_dim(config["max_resources"]),
            candidate_dim=set_candidate_feature_dim() if use_set else candidate_feature_dim(config["max_resources"]),
            d_model=config["d_model"], n_heads=config["n_heads"], n_layers=config["n_layers"],
            ffn_dim=config["ffn_dim"], dropout=0.0,
            resource_feature_dim=set_resource_feature_dim() if use_set else 0,
            interaction_feature_dim=set_interaction_feature_dim() if use_set else 0,
        ).to(device)
    if actor.resource_set_enabled != use_set:
        raise ValueError("resource_encoder does not match checkpoint feature schema")
    bc_actor = copy.deepcopy(actor).eval().requires_grad_(False) if config["bc_checkpoint"] else None
    optimizer = torch.optim.AdamW(actor.parameters(), lr=float(config["actor_lr"]))
    optima = {}
    if config["eval_optimal_json"]:
        raw = json.loads(Path(config["eval_optimal_json"]).read_text())
        optima = {Path(name).stem.lower(): int(value["makespan"])
                  for name, value in raw.get("instances", {}).items()
                  if isinstance(value, dict) and "makespan" in value}
    writer = None
    if config["tensorboard"]:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(log_dir=str(save_path.parent / "tb"))
    print_startup_summary(config, actor, train_paths, eval_paths, device)
    run_started = time.perf_counter()
    evaluation = []
    evaluation_path = save_path.with_suffix(".eval.json")
    if eval_paths:
        evaluation_path.write_text(json.dumps(evaluation, indent=2))
    batch = BatchAccumulator()
    total_episodes = global_step = update = 0
    best_score = None
    queue = []
    collection_started = time.perf_counter()
    try:
        with save_path.with_suffix(".updates.jsonl").open("w") as update_log:
            while global_step < config["total_env_steps"]:
                if not queue:
                    queue = list(train_paths)
                    rng.shuffle(queue)
                path = queue.pop()
                total_episodes += 1
                episode_started = time.perf_counter()
                env = BranchingEnv(
                    path, max_resources=config["max_resources"], resource_encoder=config["resource_encoder"],
                    time_limit_s=config["time_limit_s"], max_nodes=config["max_nodes"], dominance="set_based",
                    stop_at_dive_boundary=True, stop_at_root_lb=config["stop_at_root_lb"],
                )
                episode_records, episode_diagnostics, steps = collect_episode(actor, env, device, config)
                global_step += steps
                batch.add(path, episode_records, episode_diagnostics)
                episode_time = time.perf_counter() - episode_started
                log_episode(total_episodes, path, env=env, global_step=global_step,
                            elapsed_s=episode_time, episode_records=episode_records,
                            diagnostics=episode_diagnostics, batch=batch, config=config,
                            n_train_instances=len(train_paths))
                at_end = global_step >= config["total_env_steps"]
                if not batch.ready(config, len(train_paths)) and not at_end:
                    continue
                did_update = bool(batch.records)
                if did_update:
                    update += 1
                    collection_time = time.perf_counter() - collection_started
                    update_started = time.perf_counter()
                    metrics = train_update(actor, optimizer, batch.records, bc_actor, device, config)
                    retained_pairs = Counter(pair.pair_type for record in batch.records for pair in record.pairs)
                    report = {"update": update, "global_step": global_step,
                              "n_retained_split_nodes": len(batch.records),
                              "n_retained_pairs": sum(retained_pairs.values()),
                              "n_episodes": batch.episodes, "total_episodes": total_episodes,
                              "n_contributing_episodes": batch.contributing_episodes,
                              "n_contributing_instances": len(batch.contributor_instances),
                              "collection_time_s": collection_time,
                              "update_time_s": time.perf_counter() - update_started,
                              **dict(batch.diagnostics), **metrics}
                    for kind in ("certified", "empirical", "tie_break_bound"):
                        report[f"retained_{kind}_pairs"] = retained_pairs[kind]
                    log_update(report)
                    update_log.write(json.dumps(report) + "\n")
                    update_log.flush()
                    if writer:
                        for key, value in report.items():
                            if isinstance(value, (int, float)) and math.isfinite(value):
                                writer.add_scalar(f"train/{key}", value, global_step)
                        writer.flush()
                else:
                    print("[Update] skipped: final collection has no active splits", flush=True)
                extra = {"training_method": "censored_pairwise_ranknet", "global_step": global_step,
                         "update": update, "training_config": config}
                save_actor_checkpoint(actor, save_path, extra)
                if (did_update and update % config["eval_every_updates"] == 0) or at_end:
                    checkpoint_path = checkpoint_dir / f"policy_ranknet_step{global_step}.pt"
                    save_actor_checkpoint(actor, checkpoint_path, extra)
                    if eval_paths:
                        label = f"update {update}" if did_update else "final actor"
                        result = evaluate(actor, eval_paths, config=config, device=device, optimal_makespans=optima)
                        evaluation.append(summarize_evaluation(result, step=global_step, update=update))
                        evaluation_path.write_text(json.dumps(evaluation, indent=2))
                        log_evaluation(label, result, writer=writer, step=global_step)
                        print(flush=True)
                        summary = result["summary"]
                        score = (-summary["feasible_fraction"], summary["mean_gap_to_root_bound"] if summary["mean_gap_to_root_bound"] is not None else float("inf"), -summary["proof_fraction"], summary["mean_nodes_expanded"])
                        if best_score is None or score < best_score:
                            best_score = score
                            best_path = save_path.with_name(save_path.stem + "_best.pt")
                            save_actor_checkpoint(actor, best_path, extra)
                batch = BatchAccumulator()
                collection_started = time.perf_counter()
        print(f"\nTraining complete | env_steps={global_step:,} episodes={total_episodes} "
              f"updates={update} elapsed={time.perf_counter() - run_started:.1f}s\n"
              f"  actor={save_path}\n  update_log={save_path.with_suffix('.updates.jsonl')}\n"
              f"  evaluation_log={evaluation_path if eval_paths else 'disabled'}", flush=True)
    finally:
        if writer:
            writer.close()
        if log_handle:
            sys.stdout, sys.stderr = original_stdout, original_stderr
            log_handle.close()


if __name__ == "__main__":
    main()
