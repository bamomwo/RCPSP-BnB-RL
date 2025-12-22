import argparse
import json
import random
import sys
from collections import deque
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

# Ensure the src directory is importable when running from the repo root.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from rcpsp_bb_rl.data.dataset import list_instance_paths  # noqa: E402
from rcpsp_bb_rl.models import PolicyMLP, load_policy_checkpoint  # noqa: E402
from rcpsp_bb_rl.rl import BranchingEnv  # noqa: E402


def _copy_policy_weights(src: PolicyMLP, dst: PolicyMLP) -> None:
    """
    Copy linear layer weights/biases from a source PolicyMLP to a destination PolicyMLP,
    ignoring differences in Dropout placement. Raises if the linear layer shapes differ.
    """
    src_linears = [m for m in src.backbone if isinstance(m, nn.Linear)]
    dst_linears = [m for m in dst.backbone if isinstance(m, nn.Linear)]
    if len(src_linears) != len(dst_linears):
        raise RuntimeError(
            f"Mismatch in linear layer counts when copying BC weights (src={len(src_linears)} dst={len(dst_linears)})"
        )
    for s, d in zip(src_linears, dst_linears):
        if s.weight.shape != d.weight.shape:
            raise RuntimeError(
                f"Linear weight shape mismatch when copying BC weights (src={s.weight.shape} dst={d.weight.shape})"
            )
        d.weight.data.copy_(s.weight.data)
        d.bias.data.copy_(s.bias.data)
    dst.head.load_state_dict(src.head.state_dict())


def _instance_label(source: Path | str | object) -> str:
    """Short label for logging which instance an episode uses."""
    try:
        return Path(str(source)).name
    except Exception:
        return str(source)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PPO fine-tuning for B&B branching policy (config-driven only).")
    parser.add_argument(
        "--config",
        required=True,
        help="Path to JSON config file with all required hyperparameters.",
    )
    return parser.parse_args()


def load_config(path: Path) -> Dict:
    with path.open() as f:
        return json.load(f)


def validate_config(config: Dict[str, object]) -> Dict[str, object]:
    required_fields = {
        "root",
        "pattern",
        "max_resources",
        "step_cost",
        "terminal_makespan_coeff",
        "total_env_steps",
        "rollout_horizon",
        "ppo_epochs",
        "clip_eps",
        "gae_lambda",
        "gamma",
        "ent_coef",
        "vf_coef",
        "lr",
        "max_grad_norm",
        "hidden_sizes",
        "dropout",
        "seed",
        "device",
        "save_path",
    }
    optional_fields = {"max_instances", "max_steps_per_episode", "bc_checkpoint"}

    missing = [key for key in sorted(required_fields) if key not in config or config[key] is None]
    if missing:
        missing_fields = ", ".join(missing)
        raise ValueError(f"Config file is missing required fields: {missing_fields}")

    merged = {**{k: None for k in optional_fields}, **config}
    return merged


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class ActorCritic(nn.Module):
    """Policy head from the BC model plus a simple value head."""

    def __init__(
        self,
        global_dim: int,
        candidate_dim: int,
        hidden_sizes: Sequence[int],
        dropout: float,
    ) -> None:
        super().__init__()
        self.policy = PolicyMLP(global_dim, candidate_dim, hidden_sizes=hidden_sizes, dropout=dropout)
        self.value_head = nn.Sequential(
            nn.Linear(global_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def policy_logits(self, cand_feats: torch.Tensor, glob_feats: torch.Tensor) -> torch.Tensor:
        return self.policy(cand_feats, glob_feats)

    def value(self, glob_feats: torch.Tensor) -> torch.Tensor:
        return self.value_head(glob_feats).squeeze(-1)


def make_envs(
    instance_paths: Sequence[Path],
    max_resources: int,
    step_cost: float,
    max_steps_per_episode: int | None,
    terminal_makespan_coeff: float,
) -> List[BranchingEnv]:
    envs = []
    for p in instance_paths:
        envs.append(
            BranchingEnv(
                instance_source=p,
                max_resources=max_resources,
                step_cost=step_cost,
                terminal_makespan_coeff=terminal_makespan_coeff,
                max_steps=max_steps_per_episode,
            )
        )
    return envs


@torch.no_grad()
def select_action(
    model: ActorCritic,
    obs: Dict[str, torch.Tensor],
    device: torch.device,
) -> Tuple[int, torch.Tensor, torch.Tensor]:
    cand = obs["candidate_feats"].to(device)
    glob = obs["global_feats"].to(device)
    mask = obs["action_mask"].to(device)

    if cand.numel() == 0:
        raise RuntimeError("No available actions in the ready set.")

    glob_rep = glob.unsqueeze(0).repeat(len(cand), 1)
    logits = model.policy_logits(cand, glob_rep)
    logits = logits.masked_fill(~mask, -1e9)
    dist = Categorical(logits=logits)
    action_idx = dist.sample()
    logprob = dist.log_prob(action_idx)
    entropy = dist.entropy()
    return int(action_idx.item()), logprob, entropy


def compute_logprob_entropy(
    model: ActorCritic,
    obs: Dict[str, torch.Tensor],
    action_idx: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    cand = obs["candidate_feats"].to(device)
    glob = obs["global_feats"].to(device)
    mask = obs["action_mask"].to(device)
    glob_rep = glob.unsqueeze(0).repeat(len(cand), 1)
    logits = model.policy_logits(cand, glob_rep)
    logits = logits.masked_fill(~mask, -1e9)
    dist = Categorical(logits=logits)
    idx_tensor = torch.tensor(action_idx, device=device)
    logprob = dist.log_prob(idx_tensor)
    entropy = dist.entropy()
    return logprob, entropy


def compute_gae(
    rewards: List[float],
    values: List[float],
    dones: List[bool],
    gamma: float,
    lam: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    gae = 0.0
    returns: List[float] = []
    for t in reversed(range(len(rewards))):
        next_value = 0.0 if dones[t] else values[t + 1]
        delta = rewards[t] + gamma * next_value * (0.0 if dones[t] else 1.0) - values[t]
        gae = delta + gamma * lam * (0.0 if dones[t] else 1.0) * gae
        returns.insert(0, gae + values[t])
    adv = torch.tensor([r - v for r, v in zip(returns, values[:-1])], dtype=torch.float32)
    returns_t = torch.tensor(returns, dtype=torch.float32)
    return adv, returns_t


def main() -> None:
    args = parse_args()

    cfg_path = Path(args.config)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config file not found at {cfg_path}. Please supply a valid --config path.")
    config = validate_config(load_config(cfg_path))

    set_seed(int(config["seed"]))

    # Resolve device from config but gracefully fall back to CPU if unavailable.
    requested_device = str(config["device"])
    if requested_device == "cuda" and not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device(requested_device)

    instance_paths = list_instance_paths(config["root"], patterns=(config["pattern"],))
    if config["max_instances"] is not None:
        instance_paths = instance_paths[: int(config["max_instances"])]
    if not instance_paths:
        raise FileNotFoundError(f"No instances found under {config['root']} matching {config['pattern']}")

    # Single-env setup for now; can extend to vectorized later.
    env = BranchingEnv(
        instance_source=random.choice(instance_paths),
        max_resources=int(config["max_resources"]),
        step_cost=float(config["step_cost"]),
        terminal_makespan_coeff=float(config["terminal_makespan_coeff"]),
        max_steps=None if config["max_steps_per_episode"] is None else int(config["max_steps_per_episode"]),
    )
    obs = env.reset()
    episodes_started = 1
    print(f"[episode {episodes_started} start] instance={_instance_label(env.instance_source)}")

    # Infer feature dims from the first observation.
    global_dim = obs["global_feats"].numel()
    candidate_dim = (
        obs["candidate_feats"].shape[1]
        if obs["candidate_feats"].numel() > 0
        else int(config["max_resources"]) + 5
    )

    model = ActorCritic(
        global_dim=global_dim,
        candidate_dim=candidate_dim,
        hidden_sizes=tuple(config["hidden_sizes"]),
        dropout=float(config["dropout"]),
    ).to(device)

    # Warm start from BC if provided.
    if config.get("bc_checkpoint"):
        bc_model = load_policy_checkpoint(config["bc_checkpoint"], device=device)
        try:
            model.policy.load_state_dict(bc_model.state_dict(), strict=True)
        except RuntimeError as e:
            if "Missing key(s)" in str(e) or "Unexpected key(s)" in str(e):
                _copy_policy_weights(bc_model, model.policy)
                print(
                    "Loaded BC checkpoint by copying linear weights (dropout layouts differed between checkpoint and PPO config)."
                )
            else:
                raise

    optimizer = optim.AdamW(model.parameters(), lr=float(config["lr"]))

    total_steps = 0
    update_idx = 0
    recent_returns = deque(maxlen=10)
    episode_return = 0.0
    episodes_completed = 0
    done_reason_counts: Dict[str, int] = {}

    total_env_steps = int(config["total_env_steps"])
    rollout_horizon = int(config["rollout_horizon"])
    while total_steps < total_env_steps:
        rollout = []
        for _ in range(rollout_horizon):
            with torch.no_grad():
                value = model.value(obs["global_feats"].to(device)).item()
                action_idx, logprob, entropy = select_action(model, obs, device)
            step_out = env.step(action_idx)
            episode_return += step_out.reward
            if step_out.info.get("best_makespan") is not None:
                print(
                    f"[incumbent] step={total_steps + 1} "
                    f"makespan={step_out.info['best_makespan']} "
                    f"stack={step_out.info.get('stack_size', 'na')} "
                    f"improvement={step_out.info.get('makespan_improvement', 'na')} "
                    f"reward={step_out.reward:.4f} "
                    f"episode_return={episode_return:.2f}"
                )
            transition = {
                "obs": obs,
                "action_idx": action_idx,
                "logprob": logprob.detach(),
                "entropy": entropy.detach(),
                "reward": step_out.reward,
                "done": step_out.done,
                "value": value,
            }
            rollout.append(transition)
            total_steps += 1
            if step_out.done:
                recent_returns.append(episode_return)
                episodes_completed += 1
                reason = step_out.info.get("done_reason", "unknown")
                done_reason_counts[reason] = done_reason_counts.get(reason, 0) + 1
                episode_return = 0.0
                next_instance = random.choice(instance_paths)
                episodes_started += 1
                print(f"[episode {episodes_started} start] instance={_instance_label(next_instance)}")
                obs = env.reset(next_instance)
                # Reset rollout collection for the next episode within the same horizon
                # while still respecting the rollout size.
            else:
                obs = step_out.observation

            if total_steps >= total_env_steps:
                break

        # Bootstrap value for GAE
        with torch.no_grad():
            next_value = model.value(obs["global_feats"].to(device)).item()
        rewards = [t["reward"] for t in rollout]
        dones = [t["done"] for t in rollout]
        values = [t["value"] for t in rollout] + [next_value]
        advantages, returns = compute_gae(
            rewards,
            values,
            dones,
            float(config["gamma"]),
            float(config["gae_lambda"]),
        )
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # PPO update
        for _ in range(int(config["ppo_epochs"])):
            policy_losses = []
            value_losses = []
            entropies = []
            for t, adv, ret in zip(rollout, advantages, returns):
                logprob_new, entropy = compute_logprob_entropy(model, t["obs"], t["action_idx"], device)
                ratio = torch.exp(logprob_new - t["logprob"])
                clipped = torch.clamp(ratio, 1.0 - float(config["clip_eps"]), 1.0 + float(config["clip_eps"]))
                policy_loss = -torch.min(ratio * adv, clipped * adv)

                value_pred = model.value(t["obs"]["global_feats"].to(device))
                value_loss = (value_pred - ret) ** 2

                policy_losses.append(policy_loss)
                value_losses.append(value_loss)
                entropies.append(entropy)

            optimizer.zero_grad()
            loss_pi = torch.stack(policy_losses).mean()
            loss_v = torch.stack(value_losses).mean()
            loss_ent = torch.stack(entropies).mean()
            loss = loss_pi + float(config["vf_coef"]) * loss_v - float(config["ent_coef"]) * loss_ent
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), float(config["max_grad_norm"]))
        optimizer.step()

        update_idx += 1
        avg_return = sum(recent_returns) / len(recent_returns) if recent_returns else 0.0
        done_summary = ", ".join(f"{k}:{v}" for k, v in sorted(done_reason_counts.items()))
        print(
            f"Update {update_idx} | steps={total_steps} | "
            f"loss_pi={loss_pi.item():.4f} loss_v={loss_v.item():.4f} ent={loss_ent.item():.4f} | "
            f"avg_return(last {len(recent_returns)} eps)={avg_return:.2f} | "
            f"episodes={episodes_completed} done_reasons=[{done_summary}]"
        )

        # Save checkpoint each update
        save_path = Path(config["save_path"])
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state": model.policy.state_dict(),
                "value_state": model.value_head.state_dict(),
                "config": {
                    "global_dim": global_dim,
                    "candidate_dim": candidate_dim,
                    "hidden_sizes": tuple(config["hidden_sizes"]),
                    "dropout": float(config["dropout"]),
                },
            },
            save_path,
        )

        # TODO: add optional evaluation hook that calls scripts/report_bnb.py on a held-out set.

    print(f"Training complete. Saved policy to {config['save_path']}")


if __name__ == "__main__":
    main()
