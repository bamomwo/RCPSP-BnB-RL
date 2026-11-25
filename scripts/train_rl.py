"""Primal-policy training with greedy single dives and sibling RankNet."""
from __future__ import annotations

import argparse
from contextlib import redirect_stderr, redirect_stdout
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rcpsp_bnb_rl.data.dataset import list_instance_paths
from rcpsp_bnb_rl.data.parsing import load_instance
from rcpsp_bnb_rl.ml.models import BranchingTransformer, load_policy_checkpoint, save_policy_checkpoint
from rcpsp_bnb_rl.ml.observations import OBSERVATION_SCHEMA
from rcpsp_bnb_rl.ml.rl.evaluation import evaluate_actor
from rcpsp_bnb_rl.ml.rl.primal_training import train_update
from rcpsp_bnb_rl.ml.rl.reporting import format_evaluation, save_evaluation, summarize_evaluation
from rcpsp_bnb_rl.ml.rl.env import collect_fork_episode


DEFAULT_CONFIG = {
    "root": "data/train/1kNetRes",
    "pattern": "*.rcp",
    "max_instances": None,
    "bc_checkpoint": None,
    "d_model": 64,
    "n_heads": 4,
    "n_layers": 2,
    "ffn_dim": 128,
    "total_updates": 200,
    "episodes_per_update": 32,
    "epochs": 2,
    "target_mb_size": 128,
    "max_forks_per_dive": 32,
    "max_active_forks": 128,
    "inference_chunk_size": 128,
    "metrics_batch_size": 128,
    "actor_lr": 5e-5,
    "entropy_coef": 0.001,
    "target_kl": 0.02,
    "max_grad_norm": 0.5,
    "lower_bound": "lb_cp",
    "eval_every_updates": 10,
    "eval_root": "data/eval/intrain/J30_10",
    "eval_pattern": "*.rcp",
    "eval_limit": None,
    "eval_time_limit_s": None,
    "eval_max_nodes": 5000,
    "eval_optimal_json": "data/baseline/j30_opt.json",
    "save_path": "models/primal-policy/policy.pt",
    "checkpoint_dir": "models/primal-policy/checkpoints",
    "tensorboard": True,
    "seed": 42,
    "device": "cuda",
    "torch_threads": 1,
}


def load_config(path: Path) -> dict:
    supplied = json.loads(path.read_text())
    if not isinstance(supplied, dict):
        raise ValueError("Training config must be a JSON object")
    unknown = supplied.keys() - DEFAULT_CONFIG.keys()
    if unknown:
        raise ValueError(f"Unknown training config keys: {sorted(unknown)}")
    config = {**DEFAULT_CONFIG, **supplied}
    for name in ("total_updates", "episodes_per_update", "epochs", "target_mb_size",
                 "max_active_forks", "inference_chunk_size", "metrics_batch_size",
                 "eval_every_updates", "torch_threads", "d_model", "n_heads", "n_layers", "ffn_dim"):
        value = config[name]
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    for name in ("actor_lr", "max_grad_norm"):
        value = config[name]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be positive and finite")
    for name in ("entropy_coef",):
        value = config[name]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be non-negative and finite")
    for name in ("target_kl", "eval_time_limit_s"):
        value = config[name]
        if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0):
            raise ValueError(f"{name} must be positive and finite or null")
    for name in ("max_instances", "eval_limit", "eval_max_nodes", "max_forks_per_dive"):
        value = config[name]
        if value is not None and (isinstance(value, bool) or not isinstance(value, int) or value < 1):
            raise ValueError(f"{name} must be a positive integer or null")
    if config["eval_root"] and config["eval_max_nodes"] is None and config["eval_time_limit_s"] is None:
        raise ValueError("Evaluation requires eval_max_nodes or eval_time_limit_s")
    checkpoint = config["bc_checkpoint"]
    if checkpoint is not None and (not isinstance(checkpoint, str) or not checkpoint.strip()):
        raise ValueError("bc_checkpoint must be a nonempty path string or null for random initialization")
    if config["d_model"] % config["n_heads"]:
        raise ValueError("d_model must be divisible by n_heads")
    return config


def initialize_actor(config: dict, device: torch.device | str) -> BranchingTransformer:
    """Load an actor checkpoint or construct a random actor for the shared schema."""
    if config["bc_checkpoint"] is None:
        actor = BranchingTransformer(
            d_model=config["d_model"], n_heads=config["n_heads"],
            n_layers=config["n_layers"], ffn_dim=config["ffn_dim"], dropout=0.0,
        ).to(device)
    else:
        actor = load_policy_checkpoint(config["bc_checkpoint"], device=device)
    return actor.eval()


def load_optima(path: str | None) -> dict[str, int]:
    if path is None:
        return {}
    payload = json.loads(Path(path).read_text())
    return {Path(name).stem.lower(): int(value["makespan"])
            for name, value in payload["instances"].items() if value.get("makespan") is not None}


class _Tee:
    """Mirror console output to the optional training text log."""

    def __init__(self, console, logfile) -> None:
        self.console = console
        self.logfile = logfile

    def write(self, text: str) -> int:
        self.console.write(text)
        self.logfile.write(text)
        return len(text)

    def flush(self) -> None:
        self.console.flush()
        self.logfile.flush()


def run_training(config: dict) -> dict:
    torch.set_num_threads(config["torch_threads"])
    random.seed(config["seed"])
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    device = torch.device(config["device"])
    if device.type == "cuda" and not torch.cuda.is_available():
        print("CUDA unavailable; using CPU for this run.", flush=True)
        device = torch.device("cpu")
    paths = list_instance_paths(config["root"], (config["pattern"],))
    if config["max_instances"] is not None:
        paths = paths[:config["max_instances"]]
    if not paths:
        raise ValueError("No training instances found")
    eval_paths = list_instance_paths(config["eval_root"], (config["eval_pattern"],)) if config["eval_root"] else []
    if config["eval_limit"] is not None:
        eval_paths = eval_paths[:config["eval_limit"]]
    optima = load_optima(config["eval_optimal_json"])
    actor = initialize_actor(config, device)
    optimizer = torch.optim.AdamW(actor.parameters(), lr=config["actor_lr"], weight_decay=0.0)
    save_path = Path(config["save_path"])
    save_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = Path(config["checkpoint_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    evaluation_log = save_path.parent / "evaluation.json"
    writer = None
    if config["tensorboard"]:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(str(save_path.parent / "tb"))
    evaluation_options = dict(device=device, time_limit_s=config["eval_time_limit_s"],
                              max_nodes=config["eval_max_nodes"], lb_spec=config["lower_bound"],
                              optimal_makespans=optima)
    rng = random.Random(config["seed"])
    order = []
    env_steps = 0
    episodes = 0
    started = time.perf_counter()
    print(f"Actor-only sibling training: {len(paths)} instances, {device}; schema={OBSERVATION_SCHEMA}", flush=True)
    initialization = "random" if config["bc_checkpoint"] is None else f"checkpoint ({config['bc_checkpoint']})"
    print(f"  initialization={initialization}", flush=True)
    print(f"  instances/update={config['episodes_per_update']}  minibatch={config['target_mb_size']}  "
          f"epochs={config['epochs']}  lr={config['actor_lr']:g}  target_KL={config['target_kl']}", flush=True)
    fork_budget = config["max_forks_per_dive"]
    print(f"  fork_budget={fork_budget if fork_budget is not None else 'all'}  "
          f"active_forks={config['max_active_forks']}  inference_batch={config['inference_chunk_size']}  "
          f"metrics_batch={config['metrics_batch_size']}", flush=True)
    if eval_paths:
        node_limit = config["eval_max_nodes"]
        time_limit = config["eval_time_limit_s"]
        print(f"  evaluation: {len(eval_paths)} instances  "
              f"max_nodes/instance={node_limit if node_limit is not None else 'none'}  "
              f"time_limit/instance={f'{time_limit:g}s' if time_limit is not None else 'none'}", flush=True)
        print("  solved=optimality proved  primal_opt=known optimal makespan found", flush=True)
    try:
        for update in range(1, config["total_updates"] + 1):
            print(f"\n[Collect {update}/{config['total_updates']}] instances={config['episodes_per_update']}", flush=True)
            records = []
            episode_reports = []
            batch_started = time.perf_counter()
            for _ in range(config["episodes_per_update"]):
                if not order:
                    order = list(paths)
                    rng.shuffle(order)
                path = order.pop()
                instance = load_instance(path)
                episode = collect_fork_episode(
                    actor, instance, device, lb_spec=config["lower_bound"],
                    max_forks_per_dive=config["max_forks_per_dive"],
                    max_active_forks=config["max_active_forks"],
                    inference_chunk_size=config["inference_chunk_size"],
                )
                records.extend(episode.records)
                diagnostics = dict(episode.diagnostics)
                env_steps += int(diagnostics["environment_steps"])
                episodes += 1
                episode_reports.append({"instance": str(path), "stem_makespan": episode.stem_makespan,
                                        "root_lower_bound": episode.root_lower_bound, **diagnostics})
                fork_count = str(len(episode.records))
                if diagnostics["n_eligible_fork_states"] > len(episode.records):
                    fork_count += f"/{diagnostics['n_eligible_fork_states']}"
                print(f"  [Episode {episodes}] {path.name:<24}  stem={episode.stem_makespan:<4}  "
                      f"best={diagnostics['best_makespan']:<4}  lb={episode.root_lower_bound:<4}  "
                      f"forks={fork_count:<2}  signal={diagnostics['n_non_tied_groups']:<2}  "
                      f"beat={diagnostics['n_beat_states']}", flush=True)
            report = train_update(actor, optimizer, records, device, epochs=config["epochs"],
                                  minibatch_size=config["target_mb_size"], entropy_coef=config["entropy_coef"],
                                  target_kl=config["target_kl"], max_grad_norm=config["max_grad_norm"],
                                  metrics_batch_size=config["metrics_batch_size"])
            duration = time.perf_counter() - batch_started
            rollout_summary = {
                "mean_dive1_makespan": float(np.mean([e["stem_makespan"] for e in episode_reports])),
                "best_dive1_makespan": min(e["stem_makespan"] for e in episode_reports),
                "mean_dive1_gap_to_root_bound": float(np.mean([e["stem_gap_to_root_bound"] for e in episode_reports])),
                "mean_best_fork_makespan": float(np.mean([e["best_makespan"] for e in episode_reports])),
                "mean_improvement": float(np.mean([e["stem_makespan"] - e["best_makespan"] for e in episode_reports])),
                "dives_per_s": sum(1 + e["n_fork_dives"] for e in episode_reports) / max(duration, 1e-9),
                "depth_counts": {band: {key: sum(e["depth_counts"][band][key] for e in episode_reports)
                                        for key in ("states", "pairs", "non_tied_pairs")}
                                 for band in ("shallow", "mid", "deep")},
            }
            report.update(update=update, step=env_steps, episodes=episodes, episode_reports=episode_reports,
                          rollout_summary=rollout_summary, wall_s=duration,
                          elapsed_s=time.perf_counter() - started, updates_per_hour=3600 / max(duration, 1e-9))
            pair_accuracy = report["before"]["ranking_accuracy"]
            pair_accuracy_text = f"{pair_accuracy:.2f}" if pair_accuracy is not None else "n/a"
            print(flush=True)
            print(f"[Update {update}] states={len(records)} info={report['informative_states']} "
                  f"pair_acc={pair_accuracy_text} impr={rollout_summary['mean_improvement']:.1f} "
                  f"loss={report['before']['ranknet_loss']:.4f}->{report['after']['ranknet_loss']:.4f} "
                  f"KL={report['after']['kl_old']:.5f} steps={report['optimizer_steps']} wall={duration:.2f}s", flush=True)
            if writer is not None:
                for key, value in report['after'].items():
                    if isinstance(value, (float, int)):
                        writer.add_scalar(f"train/{key}", value, update)
                writer.add_scalar("train/signal_density", report["signal_density"], update)
                if pair_accuracy is not None:
                    writer.add_scalar("train/pair_acc", pair_accuracy, update)
                writer.add_scalar("throughput/updates_per_hour", report["updates_per_hour"], update)
                for key, value in rollout_summary.items():
                    if isinstance(value, (float, int)):
                        writer.add_scalar(f"rollout/{key}", value, update)
                for band, counts in rollout_summary["depth_counts"].items():
                    for key, value in counts.items():
                        writer.add_scalar(f"rollout/{band}/{key}", value, update)
            metadata = {"step": env_steps, "update": update, "training_config": config,
                        "training_method": "greedy_sibling_ranknet_v1", "observation_schema": OBSERVATION_SCHEMA,
                        "feature_schema": OBSERVATION_SCHEMA,
                        "actor_initialization": "random" if config["bc_checkpoint"] is None else "checkpoint",
                        "fork_selection": "stratified_depth", "fork_outcome_scope": "selected_fork_states"}
            save_policy_checkpoint(actor, str(save_path), metadata)
            if update % config["eval_every_updates"] == 0 or update == config["total_updates"]:
                save_policy_checkpoint(actor, str(checkpoint_dir / f"policy_update{update}.pt"), metadata)
                if eval_paths:
                    print("\n" + "." * 110, flush=True)
                    evaluation = evaluate_actor(actor, eval_paths, **evaluation_options)
                    entry = summarize_evaluation(evaluation, update)
                    save_evaluation(evaluation_log, entry)
                    print(format_evaluation(evaluation, entry), flush=True)
                    print("." * 110 + "\n", flush=True)
                    if writer is not None:
                        for key, value in entry.items():
                            if key != "update" and isinstance(value, (float, int)):
                                writer.add_scalar(f"eval/{key}", value, update)
    finally:
        if writer is not None:
            writer.close()
    return {"updates": config["total_updates"], "episodes": episodes, "environment_steps": env_steps,
            "save_path": str(save_path), "evaluation_log": str(evaluation_log)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/train_rl.json", type=Path)
    parser.add_argument("--log", action="store_true", help="Also save console output beside the policy as <policy_stem>_train_log.txt.")
    args = parser.parse_args()
    config = load_config(args.config)
    if args.log:
        path = Path(config["save_path"])
        path.parent.mkdir(parents=True, exist_ok=True)
        log_path = path.with_name(f"{path.stem}_train_log.txt")
        with log_path.open("w") as logfile:
            with redirect_stdout(_Tee(sys.stdout, logfile)), redirect_stderr(_Tee(sys.stderr, logfile)):
                print(f"Logging training output to: {log_path}", flush=True)
                run_training(config)
    else:
        run_training(config)


if __name__ == "__main__":
    main()
