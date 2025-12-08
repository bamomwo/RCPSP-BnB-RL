import argparse
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PPO fine-tuning for B&B branching policy.")
    parser.add_argument("--root", default="data/j30rcp", help="Directory of RCPSP instances for training.")
    parser.add_argument("--pattern", default="*.RCP", help="Glob pattern for instances under root.")
    parser.add_argument("--max-instances", type=int, default=None, help="Optional limit on instances to sample from.")
    parser.add_argument("--max-steps-per-episode", type=int, default=None, help="Hard cap on env steps per episode.")
    parser.add_argument("--max-resources", type=int, default=4, help="Pad/truncate resource dims for features.")
    parser.add_argument("--step-cost", type=float, default=1.0, help="Per-step penalty (kept at -1 by default).")
    parser.add_argument("--total-env-steps", type=int, default=5000, help="Total env steps to collect.")
    parser.add_argument("--rollout-horizon", type=int, default=256, help="Steps per PPO rollout before updating.")
    parser.add_argument("--ppo-epochs", type=int, default=4, help="SGD passes over each rollout batch.")
    parser.add_argument("--clip-eps", type=float, default=0.2, help="PPO clip range.")
    parser.add_argument("--gae-lambda", type=float, default=0.95, help="GAE lambda.")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor.")
    parser.add_argument("--ent-coef", type=float, default=0.01, help="Entropy bonus coefficient.")
    parser.add_argument("--vf-coef", type=float, default=0.5, help="Value loss coefficient.")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate.")
    parser.add_argument("--max-grad-norm", type=float, default=0.5, help="Gradient clipping.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", help="Torch device.")
    parser.add_argument("--bc-checkpoint", default=None, help="Optional path to a supervised BC checkpoint to warm start.")
    parser.add_argument("--save-path", default=PROJECT_ROOT / "models" / "policy_ppo.pt", help="Where to save checkpoints.")
    return parser.parse_args()


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
) -> List[BranchingEnv]:
    envs = []
    for p in instance_paths:
        envs.append(
            BranchingEnv(
                instance_source=p,
                max_resources=max_resources,
                step_cost=step_cost,
                terminal_makespan_coeff=0.0,  # TODO: add terminal makespan shaping when optimizing final quality.
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
    set_seed(args.seed)

    device = torch.device(args.device)
    instance_paths = list_instance_paths(args.root, patterns=(args.pattern,))
    if args.max_instances:
        instance_paths = instance_paths[: args.max_instances]
    if not instance_paths:
        raise FileNotFoundError(f"No instances found under {args.root} matching {args.pattern}")

    # Single-env setup for now; can extend to vectorized later.
    env = BranchingEnv(
        instance_source=random.choice(instance_paths),
        max_resources=args.max_resources,
        step_cost=args.step_cost,
        terminal_makespan_coeff=0.0,
        max_steps=args.max_steps_per_episode,
    )
    obs = env.reset()

    # Infer feature dims from the first observation.
    global_dim = obs["global_feats"].numel()
    candidate_dim = obs["candidate_feats"].shape[1] if obs["candidate_feats"].numel() > 0 else args.max_resources + 5

    model = ActorCritic(
        global_dim=global_dim,
        candidate_dim=candidate_dim,
        hidden_sizes=(128, 128),
        dropout=0.1,
    ).to(device)

    # Warm start from BC if provided.
    if args.bc_checkpoint:
        bc_model = load_policy_checkpoint(args.bc_checkpoint, device=device)
        model.policy.load_state_dict(bc_model.state_dict())

    optimizer = optim.AdamW(model.parameters(), lr=args.lr)

    total_steps = 0
    update_idx = 0
    recent_returns = deque(maxlen=10)

    while total_steps < args.total_env_steps:
        rollout = []
        for _ in range(args.rollout_horizon):
            with torch.no_grad():
                value = model.value(obs["global_feats"].to(device)).item()
                action_idx, logprob, entropy = select_action(model, obs, device)
            step_out = env.step(action_idx)
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
            obs = step_out.observation if not step_out.done else env.reset(instance_source=random.choice(instance_paths))

            if step_out.done:
                episode_return = sum(t["reward"] for t in rollout)
                recent_returns.append(episode_return)
                # Reset rollout collection for the next episode within the same horizon
                # while still respecting the rollout size.
            if total_steps >= args.total_env_steps:
                break

        # Bootstrap value for GAE
        with torch.no_grad():
            next_value = model.value(obs["global_feats"].to(device)).item()
        rewards = [t["reward"] for t in rollout]
        dones = [t["done"] for t in rollout]
        values = [t["value"] for t in rollout] + [next_value]
        advantages, returns = compute_gae(rewards, values, dones, args.gamma, args.gae_lambda)
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # PPO update
        for _ in range(args.ppo_epochs):
            policy_losses = []
            value_losses = []
            entropies = []
            for t, adv, ret in zip(rollout, advantages, returns):
                logprob_new, entropy = compute_logprob_entropy(model, t["obs"], t["action_idx"], device)
                ratio = torch.exp(logprob_new - t["logprob"])
                clipped = torch.clamp(ratio, 1.0 - args.clip_eps, 1.0 + args.clip_eps)
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
            loss = loss_pi + args.vf_coef * loss_v - args.ent_coef * loss_ent
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()

        update_idx += 1
        avg_return = sum(recent_returns) / len(recent_returns) if recent_returns else 0.0
        print(
            f"Update {update_idx} | steps={total_steps} | "
            f"loss_pi={loss_pi.item():.4f} loss_v={loss_v.item():.4f} ent={loss_ent.item():.4f} | "
            f"avg_return(last {len(recent_returns)} eps)={avg_return:.2f}"
        )

        # Save checkpoint each update
        save_path = Path(args.save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state": model.policy.state_dict(),
                "value_state": model.value_head.state_dict(),
                "config": {
                    "global_dim": global_dim,
                    "candidate_dim": candidate_dim,
                    "hidden_sizes": (128, 128),
                    "dropout": 0.1,
                },
            },
            save_path,
        )

        # TODO: add optional evaluation hook that calls scripts/report_bnb.py on a held-out set.

    print(f"Training complete. Saved policy to {args.save_path}")


if __name__ == "__main__":
    main()
