from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import deque
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from rcpsp_bb_rl.data.dataset import list_instance_paths  # noqa: E402
from rcpsp_bb_rl.models import PolicyMLP, load_policy_checkpoint  # noqa: E402
from rcpsp_bb_rl.rl import BranchingEnv  # noqa: E402
from rcpsp_bb_rl.bnb.eval import evaluate_bnb_suite, list_eval_instances  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="PPO fine-tuning (budgeted episodes + shaping rewards).")
    p.add_argument("--config", required=True)
    return p.parse_args()


def load_json(path: Path) -> Dict[str, Any]:
    with path.open() as f:
        return json.load(f)


def load_json_list(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    try:
        with path.open() as f:
            data = json.load(f)
    except json.JSONDecodeError:
        return []
    return data if isinstance(data, list) else []


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _copy_policy_weights(src: PolicyMLP, dst: PolicyMLP) -> None:
    src_linears = [m for m in src.backbone if isinstance(m, nn.Linear)]
    dst_linears = [m for m in dst.backbone if isinstance(m, nn.Linear)]
    if len(src_linears) != len(dst_linears):
        raise RuntimeError("Mismatch in linear layer counts when copying BC weights.")
    for s, d in zip(src_linears, dst_linears):
        if s.weight.shape != d.weight.shape:
            raise RuntimeError("Linear shape mismatch when copying BC weights.")
        d.weight.data.copy_(s.weight.data)
        d.bias.data.copy_(s.bias.data)
    dst.head.load_state_dict(src.head.state_dict())


class ActorCritic(nn.Module):
    def __init__(self, global_dim: int, candidate_dim: int, hidden_sizes: Sequence[int], dropout: float) -> None:
        super().__init__()
        self.policy = PolicyMLP(global_dim, candidate_dim, hidden_sizes=hidden_sizes, dropout=dropout)
        self.value_head = nn.Sequential(nn.Linear(global_dim, 128), nn.ReLU(), nn.Linear(128, 1))

    def policy_logits(self, cand_feats: torch.Tensor, glob_feats_repeated: torch.Tensor) -> torch.Tensor:
        return self.policy(cand_feats, glob_feats_repeated)

    def value(self, glob_feats: torch.Tensor) -> torch.Tensor:
        return self.value_head(glob_feats).squeeze(-1)


def policy_dist_and_value(model: ActorCritic, obs: Dict[str, torch.Tensor], device: torch.device) -> Tuple[Categorical, torch.Tensor]:
    cand = obs["candidate_feats"].to(device)
    glob = obs["global_feats"].to(device)
    mask = obs["action_mask"].to(device)
    if cand.numel() == 0:
        raise RuntimeError("No available actions in the ready set.")
    glob_rep = glob.unsqueeze(0).repeat(len(cand), 1)
    logits = model.policy_logits(cand, glob_rep).masked_fill(~mask, -1e9)
    return Categorical(logits=logits), model.value(glob)


def compute_gae(
    rewards: torch.Tensor, dones: torch.Tensor, values: torch.Tensor, last_value: float, gamma: float, gae_lambda: float
) -> Tuple[torch.Tensor, torch.Tensor]:
    T = rewards.shape[0]
    adv = torch.zeros(T, dtype=torch.float32)
    lastgaelam = 0.0
    for t in reversed(range(T)):
        nextnonterminal = 1.0 - float(dones[t].item())
        nextvalue = last_value if t == T - 1 else float(values[t + 1].item())
        delta = float(rewards[t].item()) + gamma * nextvalue * nextnonterminal - float(values[t].item())
        lastgaelam = delta + gamma * gae_lambda * nextnonterminal * lastgaelam
        adv[t] = lastgaelam
    return adv, adv + values


def main() -> None:
    args = parse_args()
    cfg_path = Path(args.config)
    config = load_json(cfg_path)

    required = [
        "root","pattern","seed","device","save_path","max_resources","hidden_sizes","dropout","lr",
        "total_env_steps","rollout_horizon","ppo_epochs","clip_eps","gamma","gae_lambda","ent_coef",
        "vf_coef","max_grad_norm","minibatches","episode_budget",
        # new reward knobs
        "step_eps","prune_alpha","lb_beta","inc_gamma",
    ]
    missing = [k for k in required if k not in config or config[k] is None]
    if missing:
        raise ValueError(f"Config missing: {', '.join(missing)}")

    bc_checkpoint = config.get("bc_checkpoint", None)
    target_kl = config.get("target_kl", None)
    max_instances = config.get("max_instances", None)

    set_seed(int(config["seed"]))
    requested_device = str(config["device"])
    device = torch.device("cpu") if (requested_device == "cuda" and not torch.cuda.is_available()) else torch.device(requested_device)
    eval_requested_device = str(config.get("eval_policy_device", requested_device))
    eval_device = torch.device("cpu") if (eval_requested_device == "cuda" and not torch.cuda.is_available()) else torch.device(eval_requested_device)

    instance_paths = list_instance_paths(config["root"], patterns=(config["pattern"],))
    if max_instances is not None:
        instance_paths = instance_paths[: int(max_instances)]
    if not instance_paths:
        raise FileNotFoundError("No instances found.")

    episode_budget = int(config["episode_budget"])
    env = BranchingEnv(
        instance_source=random.choice(instance_paths),
        max_resources=int(config["max_resources"]),
        step_cost=0.0,                 # ignore env reward; we shape here
        terminal_makespan_coeff=0.0,   # ignore env shaping
        max_steps=episode_budget,
    )
    obs = env.reset()

    global_dim = obs["global_feats"].numel()
    candidate_dim = obs["candidate_feats"].shape[1] if obs["candidate_feats"].numel() > 0 else int(config["max_resources"]) + 5

    model = ActorCritic(
        global_dim=global_dim,
        candidate_dim=candidate_dim,
        hidden_sizes=tuple(int(x) for x in config["hidden_sizes"]),
        dropout=float(config["dropout"]),
    ).to(device)

    if bc_checkpoint:
        bc_model = load_policy_checkpoint(bc_checkpoint, device=device)
        try:
            model.policy.load_state_dict(bc_model.state_dict(), strict=True)
        except RuntimeError as e:
            if "Missing key(s)" in str(e) or "Unexpected key(s)" in str(e):
                _copy_policy_weights(bc_model, model.policy)
            else:
                raise

    optimizer = optim.AdamW(model.parameters(), lr=float(config["lr"]), eps=1e-5)

    total_env_steps = int(config["total_env_steps"])
    rollout_horizon = int(config["rollout_horizon"])
    ppo_epochs = int(config["ppo_epochs"])
    minibatches = int(config["minibatches"])
    mb_size = max(1, rollout_horizon // max(1, minibatches))

    gamma = float(config["gamma"])
    gae_lambda = float(config["gae_lambda"])
    clip_eps = float(config["clip_eps"])
    ent_coef = float(config["ent_coef"])
    vf_coef = float(config["vf_coef"])
    max_grad_norm = float(config["max_grad_norm"])

    step_eps = float(config["step_eps"])
    prune_alpha = float(config["prune_alpha"])
    lb_beta = float(config["lb_beta"])
    inc_gamma = float(config["inc_gamma"])

    eval_every_steps = int(config.get("eval_every_steps", 0))
    eval_root = str(config.get("eval_root", config["root"]))
    eval_pattern = str(config.get("eval_pattern", config["pattern"]))
    eval_limit = config.get("eval_limit", None)
    eval_max_nodes = int(config.get("eval_max_nodes", episode_budget))
    eval_time_limit = config.get("eval_time_limit", None)
    eval_output_dir = Path(config.get("eval_output_dir", "reports/eval_stats"))
    eval_compact_name = str(config.get("eval_compact_name", "wins_vs_native.json"))
    eval_checkpoint_dir = Path(config.get("eval_checkpoint_dir", "models/checkpoints"))
    eval_progress_every = int(config.get("eval_progress_every", 0))
    eval_policy_max_resources = int(config.get("eval_policy_max_resources", config["max_resources"]))

    eval_paths: Optional[List[Path]] = None
    if eval_every_steps > 0:
        eval_paths = list_eval_instances(eval_root, eval_pattern, eval_limit)
        if not eval_paths:
            raise FileNotFoundError("No eval instances found.")

    recent_returns = deque(maxlen=20)
    episodes_completed = 0
    done_reason_counts: Dict[str, int] = {}
    sps_window_start = time.time()
    sps_window_steps = 0
    global_step = 0
    episode_return_running = 0.0
    next_eval_step = eval_every_steps if eval_every_steps > 0 else None
    bc_eval_model = load_policy_checkpoint(bc_checkpoint, device=eval_device) if bc_checkpoint else None

    while global_step < total_env_steps:
        rollout_obs: List[Dict[str, torch.Tensor]] = []
        rollout_actions: List[int] = []
        rollout_logprobs: List[float] = []
        rollout_values: List[float] = []
        rollout_rewards: List[float] = []
        rollout_dones: List[int] = []

        for _ in range(rollout_horizon):
            global_step += 1
            sps_window_steps += 1

            with torch.no_grad():
                dist, v = policy_dist_and_value(model, obs, device)
            action = int(dist.sample().item())
            logprob = float(dist.log_prob(torch.tensor(action, device=device)).item())

            # --- shaping signals BEFORE step ---
            stack_before = len(env.stack)
            lb_before = float(env.node.lower_bound) if getattr(env, "node", None) is not None else 0.0
            best_before = env.best_makespan

            step_out = env.step(action)
            done = bool(step_out.done)

            # --- shaping signals AFTER step ---
            stack_after = int(step_out.info.get("stack_size", len(env.stack)))
            pruned = max(0, stack_before - stack_after)

            lb_after = float(env.node.lower_bound) if (not done and getattr(env, "node", None) is not None) else lb_before
            d_lb = max(0.0, lb_after - lb_before)

            best_after = env.best_makespan
            inc_impr = 0.0
            if best_before is not None and best_after is not None and best_after < best_before:
                inc_impr = float(best_before - best_after)

            r = (-step_eps) + (prune_alpha * pruned) + (lb_beta * d_lb) + (inc_gamma * inc_impr)

            rollout_obs.append(obs)
            rollout_actions.append(action)
            rollout_logprobs.append(logprob)
            rollout_values.append(float(v.item()))
            rollout_rewards.append(r)
            rollout_dones.append(1 if done else 0)

            episode_return_running += r

            if done:
                reason = step_out.info.get("done_reason", "unknown")
                done_reason_counts[reason] = done_reason_counts.get(reason, 0) + 1
                episodes_completed += 1
                recent_returns.append(episode_return_running)
                episode_return_running = 0.0
                obs = env.reset(random.choice(instance_paths))
            else:
                obs = step_out.observation

            if global_step >= total_env_steps:
                break

        with torch.no_grad():
            _, last_v = policy_dist_and_value(model, obs, device)
            last_value = float(last_v.item())

        rewards_t = torch.tensor(rollout_rewards, dtype=torch.float32)
        dones_t = torch.tensor(rollout_dones, dtype=torch.float32)
        values_t = torch.tensor(rollout_values, dtype=torch.float32)

        advantages_t, returns_t = compute_gae(rewards_t, dones_t, values_t, last_value, gamma, gae_lambda)
        advantages_t = (advantages_t - advantages_t.mean()) / (advantages_t.std() + 1e-8)

        b_inds = np.arange(len(rollout_obs))
        clipfracs: List[float] = []
        last_pg = last_vl = last_ent = last_kl = 0.0

        for _ in range(ppo_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, len(b_inds), mb_size):
                mb_inds = b_inds[start : start + mb_size]
                if len(mb_inds) == 0:
                    continue

                new_logps, new_ents, new_vals = [], [], []
                for i in mb_inds:
                    d_i, v_i = policy_dist_and_value(model, rollout_obs[i], device)
                    a_i = torch.tensor(rollout_actions[i], device=device)
                    new_logps.append(d_i.log_prob(a_i))
                    new_ents.append(d_i.entropy())
                    new_vals.append(v_i)

                new_logps_t = torch.stack(new_logps)
                ent_t = torch.stack(new_ents).mean()
                new_vals_t = torch.stack(new_vals).squeeze(-1)

                old_logps_t = torch.tensor([rollout_logprobs[i] for i in mb_inds], dtype=torch.float32, device=device)
                mb_adv = advantages_t[mb_inds].to(device)
                mb_ret = returns_t[mb_inds].to(device)

                logratio = new_logps_t - old_logps_t
                ratio = torch.exp(logratio)

                with torch.no_grad():
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs.append((torch.abs(ratio - 1.0) > clip_eps).float().mean().item())

                pg1 = -mb_adv * ratio
                pg2 = -mb_adv * torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps)
                pg_loss = torch.max(pg1, pg2).mean()
                v_loss = 0.5 * (new_vals_t - mb_ret).pow(2).mean()

                loss = pg_loss - ent_coef * ent_t + vf_coef * v_loss

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()

                last_pg, last_vl, last_ent, last_kl = float(pg_loss.item()), float(v_loss.item()), float(ent_t.item()), float(approx_kl.item())

            if target_kl is not None and last_kl > float(target_kl):
                break

        avg_return = float(sum(recent_returns) / len(recent_returns)) if recent_returns else 0.0
        done_summary = ", ".join(f"{k}:{v}" for k, v in sorted(done_reason_counts.items()))
        sps = int(sps_window_steps / max(time.time() - sps_window_start, 1e-6))
        sps_window_steps = 0
        sps_window_start = time.time()

        print(
            f"steps={global_step} | avg_return(last {len(recent_returns)} eps)={avg_return:.4f} | "
            f"episodes={episodes_completed} done_reasons=[{done_summary}] | "
            f"pg={last_pg:.4f} v={last_vl:.4f} ent={last_ent:.4f} kl={last_kl:.4f} "
            f"clipfrac={np.mean(clipfracs) if clipfracs else 0.0:.4f} | SPS={sps}"
        )

        save_path = Path(config["save_path"])
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state": model.policy.state_dict(),
                "value_state": model.value_head.state_dict(),
                "config": {
                    "global_dim": global_dim,
                    "candidate_dim": candidate_dim,
                    "hidden_sizes": tuple(int(x) for x in config["hidden_sizes"]),
                    "dropout": float(config["dropout"]),
                    "episode_budget": episode_budget,
                    "step_eps": step_eps,
                    "prune_alpha": prune_alpha,
                    "lb_beta": lb_beta,
                    "inc_gamma": inc_gamma,
                },
            },
            save_path,
        )

        if next_eval_step is not None and global_step >= next_eval_step:
            eval_output_dir.mkdir(parents=True, exist_ok=True)
            eval_checkpoint_dir.mkdir(parents=True, exist_ok=True)
            eval_ckpt_path = eval_checkpoint_dir / f"policy_ppo_step_{global_step}.pt"
            torch.save(
                {
                    "model_state": model.policy.state_dict(),
                    "value_state": model.value_head.state_dict(),
                    "config": {
                        "global_dim": global_dim,
                        "candidate_dim": candidate_dim,
                        "hidden_sizes": tuple(int(x) for x in config["hidden_sizes"]),
                        "dropout": float(config["dropout"]),
                        "episode_budget": episode_budget,
                        "step_eps": step_eps,
                        "prune_alpha": prune_alpha,
                        "lb_beta": lb_beta,
                        "inc_gamma": inc_gamma,
                    },
                },
                eval_ckpt_path,
            )

            was_training = model.training
            model.policy.eval()
            if bc_eval_model is not None:
                bc_eval_model.eval()
            eval_results = evaluate_bnb_suite(
                paths=eval_paths or [],
                max_nodes=eval_max_nodes,
                policy_model=model.policy,
                bc_model=bc_eval_model,
                policy_device=eval_device,
                policy_max_resources=eval_policy_max_resources,
                time_limit_s=eval_time_limit,
                progress_every=eval_progress_every,
            )
            if was_training:
                model.policy.train()
            model.policy.to(device)

            eval_payload = {
                "step": global_step,
                "timestamp": time.time(),
                "policy_checkpoint": str(eval_ckpt_path),
                "config": {
                    "eval_root": eval_root,
                    "eval_pattern": eval_pattern,
                    "eval_limit": eval_limit,
                    "eval_max_nodes": eval_max_nodes,
                    "eval_time_limit": eval_time_limit,
                    "eval_checkpoint_dir": str(eval_checkpoint_dir),
                },
                "results": eval_results,
            }
            eval_compact_path = eval_output_dir / eval_compact_name
            eval_history = load_json_list(eval_compact_path)
            wins_vs_native = eval_results.get("summary", {}).get("wins_vs_native", {})
            eval_history.append(
                {
                    "step": global_step,
                    "timestamp": eval_payload["timestamp"],
                    "policy_checkpoint": str(eval_ckpt_path),
                    "wins_vs_native": wins_vs_native,
                    "wins_vs_native_pct": eval_results.get("summary", {}).get("wins_vs_native_pct"),
                    "instances": eval_results.get("summary", {}).get("instances"),
                    "eval_limit": eval_limit,
                }
            )
            eval_compact_path.write_text(json.dumps(eval_history, indent=2, sort_keys=True))
            print(f"[eval] Saved checkpoint to {eval_ckpt_path} and stats to {eval_compact_path}")
            next_eval_step += eval_every_steps

    print(f"Training complete. Saved policy to {config['save_path']}")


if __name__ == "__main__":
    main()
