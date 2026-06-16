"""
PPO fine-tuning for the RCPSP branching policy.

Initialises from a BC checkpoint (optional), then trains with PPO using
the reward function defined in BranchingEnv / RewardConfig.

Key design decisions:
  - One episode = one RCPSP instance solved until time_limit_s or search exhausted
  - Design A: one episode == one rollout == one PPO update. Each episode is
    collected to its `done` boundary, the finished search tree is captured, and
    credit is assigned by a closure-based subtree-return backup (Machine 2,
    src/rcpsp_bb_rl/ml/rl/tree_return.py) — NOT linear GAE. A decision's return
    is the sum of per-node rewards over its own subtree; only transitions whose
    subtree closed are used in the update (open/timeout branches are dropped).
  - The transformer forward pass is unbatched (one node at a time) because
    the ready set size varies per node — we collect (logprob, value, reward)
    tuples and batch only the PPO update
  - BC checkpoint weights are loaded into both actor and critic backbone
  - Periodic evaluation on a held-out set with known optima tracks solve rate
    and gap-to-optimal; instances without a known optimum are skipped
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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
from rcpsp_bb_rl.data.parsing import load_instance  # noqa: E402
from rcpsp_bb_rl.ml.models import BranchingTransformer, load_policy_checkpoint, save_policy_checkpoint  # noqa: E402
from rcpsp_bb_rl.ml.il.featurize import global_feature_dim, candidate_feature_dim, critic_feature_dim  # noqa: E402
from rcpsp_bb_rl.ml.rl import BranchingEnv, RewardConfig  # noqa: E402
from rcpsp_bb_rl.ml.rl.tree_return import compute_episode_advantages, make_node_reward_fn  # noqa: E402
from rcpsp_bb_rl.bnb.branching_order import make_order_fn  # noqa: E402
from rcpsp_bb_rl.bnb.solver import BnBSolver  # noqa: E402


# ---------------------------------------------------------------------------
# Actor-Critic wrapper
# ---------------------------------------------------------------------------

class ActorCritic(nn.Module):
    """
    Wraps BranchingTransformer for PPO.

    The transformer already has a value_head on the CLS token — we expose
    it here as the critic. The policy head produces logits over the ready set.
    """

    def __init__(self, model: BranchingTransformer) -> None:
        super().__init__()
        self.model = model

    def forward(
        self,
        candidate_feats: torch.Tensor,
        global_feats: torch.Tensor,
        action_mask: Optional[torch.Tensor] = None,
        critic_feats: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns (logits [R], value scalar)."""
        return self.model(candidate_feats, global_feats, action_mask, critic_feats)

    def get_action_and_value(
        self,
        obs: Dict[str, torch.Tensor],
        device: torch.device,
        action: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample or evaluate an action.

        Returns (action, log_prob, entropy, value).
        If action is provided, evaluates that action (used in PPO update).
        """
        cand = obs["candidate_feats"].to(device)
        glob = obs["global_feats"].to(device)
        mask = obs["action_mask"].to(device)
        critic = obs.get("critic_feats")
        if critic is not None:
            critic = critic.to(device)

        logits, value = self.forward(cand, glob, mask, critic)
        dist = Categorical(logits=logits)

        if action is None:
            action = dist.sample()

        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        return action, log_prob, entropy, value


# ---------------------------------------------------------------------------
# Rollout buffer
# ---------------------------------------------------------------------------

class RolloutBuffer:
    """Stores transitions from one rollout horizon."""

    def __init__(self) -> None:
        self.obs: List[Dict[str, torch.Tensor]] = []
        self.actions: List[int] = []
        self.log_probs: List[float] = []
        self.values: List[float] = []
        self.rewards: List[float] = []
        self.dones: List[bool] = []
        self.terminateds: List[bool] = []
        # Tree identity of the node each decision was made at. Needed for the
        # closure-based (subtree) return backup, which reattaches transitions
        # to the search tree instead of treating them as a flat sequence.
        self.node_ids: List[Optional[int]] = []
        self.parent_ids: List[Optional[int]] = []

    def add(
        self,
        obs: Dict[str, torch.Tensor],
        action: int,
        log_prob: float,
        value: float,
        reward: float,
        done: bool,
        terminated: bool,
        node_id: Optional[int] = None,
        parent_id: Optional[int] = None,
    ) -> None:
        self.obs.append(obs)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.rewards.append(reward)
        self.dones.append(done)
        self.terminateds.append(terminated)
        self.node_ids.append(node_id)
        self.parent_ids.append(parent_id)

    def __len__(self) -> int:
        return len(self.rewards)

    def clear(self) -> None:
        self.__init__()


# ---------------------------------------------------------------------------
# GAE computation
# LEGACY: replaced by compute_episode_advantages (tree_return.py) under Design A.
# Kept for reference / possible A-B comparison; no longer called.
# ---------------------------------------------------------------------------

def compute_gae(
    rewards: List[float],
    values: List[float],
    dones: List[bool],
    terminateds: List[bool],
    last_value: float,
    gamma: float,
    gae_lambda: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generalised Advantage Estimation with separate termination/truncation
    handling.

    An episode can end two ways:
      - terminated  : the MDP genuinely ends (search_exhausted = optimality
                      proven, or invalid_action). There is no continuation,
                      so the future value is 0.
      - truncated   : the episode is cut off without the MDP ending
                      (time_limit). The search has real remaining cost, so we
                      must bootstrap V(s') instead of zeroing it. The
                      post-truncation frontier state is never materialised as
                      an observation (the solver dies mid-advance), so we
                      approximate V(s') with V(s_t), the last value we did
                      observe (buffer.values[t]).

    Two masks are needed because the single `done` flag cannot express
    truncation:
      - value_mask : 0 only on a true terminal; 1 on truncation and
                     mid-episode. Controls whether the future value is
                     bootstrapped in the TD delta.
      - trace_mask : 0 on ANY episode boundary (terminated or truncated); 1
                     only mid-episode. Stops the GAE lambda-trace from bleeding
                     advantage across an episode boundary into the next
                     instance's transitions.

    Returns (advantages [T], returns [T]).
    """
    T = len(rewards)
    advantages = torch.zeros(T)
    last_gae = 0.0

    for t in reversed(range(T)):
        done_t = bool(dones[t])
        terminated_t = bool(terminateds[t])
        # Truncation = episode boundary that is not a true terminal.
        truncated_t = done_t and not terminated_t

        # value_mask: zero the future only on a true terminal.
        value_mask = 0.0 if terminated_t else 1.0
        # trace_mask: cut the lambda-trace on any boundary.
        trace_mask = 0.0 if done_t else 1.0

        if t == T - 1:
            next_value = last_value
        else:
            next_value = values[t + 1]

        # On truncation the post-state observation is unavailable; approximate
        # V(s') with V(s_t) (Option A — see docstring).
        if truncated_t:
            next_value = values[t]

        delta = rewards[t] + gamma * next_value * value_mask - values[t]
        last_gae = delta + gamma * gae_lambda * trace_mask * last_gae
        advantages[t] = last_gae

    returns = advantages + torch.tensor(values)
    return advantages, returns


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(
    model: BranchingTransformer,
    instance_paths: List[Path],
    max_resources: int,
    time_limit_s: float,
    dominance: str,
    device: torch.device,
    optimal_makespans: Dict[str, int],
) -> Dict[str, float]:
    """
    Run the policy on a set of instances and report solve metrics.

    Requires optimal_makespans — instances without a known optimum are skipped.
    Returns dict with keys: solved_frac, mean_gap, mean_nodes.
    """
    model.eval()

    solved = 0
    gaps = []
    node_counts = []

    for path in instance_paths:
        key = path.stem.lower()
        opt = optimal_makespans.get(key)
        if opt is None:
            continue

        instance = load_instance(path)
        solver = BnBSolver(instance)
        order_fn = make_order_fn(
            "policy",
            instance=instance,
            model=model,
            max_resources=max_resources,
            device=device,
            predecessors=solver.predecessors,
        )
        result = solver.solve(
            order_ready_fn=order_fn,
            time_limit_s=time_limit_s,
            dominance=dominance,
        )
        node_counts.append(result.nodes_expanded)

        if result.best_makespan is not None:
            gap = (result.best_makespan - opt) / opt * 100.0
            gaps.append(gap)
            if result.best_makespan == opt:
                solved += 1

    n = len(gaps)
    return {
        "solved_frac": solved / n if n > 0 else 0.0,
        "mean_gap": float(np.mean(gaps)) if gaps else 0.0,
        "mean_nodes": float(np.mean(node_counts)) if node_counts else 0.0,
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="PPO fine-tuning for the RCPSP branching policy.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--config", required=True, help="Path to JSON config file.")
    p.add_argument(
        "--log",
        action="store_true",
        help="Tee training output to a .txt file in the same directory as the saved model.",
    )
    return p.parse_args()


class _Tee:
    """Duplicate writes to several streams (console + log file).

    Flushes after every write so the log file always reflects the latest
    output even if training is interrupted.
    """

    def __init__(self, *streams) -> None:
        self._streams = streams

    def write(self, data: str) -> int:
        for s in self._streams:
            s.write(data)
            s.flush()
        return len(data)

    def flush(self) -> None:
        for s in self._streams:
            s.flush()


def load_json(path: Path) -> Dict[str, Any]:
    with path.open() as f:
        return json.load(f)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


DEFAULT_CONFIG: Dict[str, Any] = {
    # Data
    "root": "data/train",
    "pattern": "*.RCP",
    "max_instances": None,
    "max_resources": 4,
    "dominance": "set_based",
    # Model
    "d_model": 64,
    "n_heads": 4,
    "n_layers": 2,
    "ffn_dim": 128,
    "dropout": 0.0,
    "bc_checkpoint": None,
    # PPO
    "total_env_steps": 1_000_000,
    "rollout_horizon": 256,
    "ppo_epochs": 4,
    "minibatches": 4,
    "clip_eps": 0.2,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    # Tree-return backup (Machine 2). Replaces GAE under Design A (one episode
    # per rollout). gamma/gae_lambda above are legacy and unused on this path.
    "tree_gamma": 1.0,        # subtree discount (1.0 = undiscounted); sweep later
    "tree_keep_open": False,  # truncation: False drops open (unclosed) subtrees
    "ent_coef": 0.01,
    "vf_coef": 0.5,
    "lr": 3e-4,
    "max_grad_norm": 0.5,
    "target_kl": 0.02,
    "time_limit_s": 60.0,
    # Reward
    "alpha": 0.01,
    "beta1": 1.0,
    "beta2": 1.0,
    # Eval
    "eval_every_steps": 20_000,
    "eval_root": None,
    "eval_pattern": "*.RCP",
    "eval_time_limit_s": 60.0,
    "eval_optimal_json": None,
    # Output
    "save_path": "models/policy_ppo.pt",
    "checkpoint_dir": "models/checkpoints",
    "seed": 42,
    "device": "cpu",
}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    config = DEFAULT_CONFIG.copy()
    config.update(load_json(Path(args.config)))

    # --- Optional log file (tee stdout to a .txt next to the saved model) ---
    if args.log:
        save_path = Path(config["save_path"])
        save_path.parent.mkdir(parents=True, exist_ok=True)
        log_file_path = save_path.parent / (save_path.stem + "_train_log.txt")
        log_fh = open(log_file_path, "w")
        sys.stdout = _Tee(sys.__stdout__, log_fh)
        sys.stderr = _Tee(sys.__stderr__, log_fh)
        print(f"Logging training output to: {log_file_path}")

    set_seed(int(config["seed"]))
    device = torch.device(
        "cpu" if (config["device"] == "cuda" and not torch.cuda.is_available())
        else config["device"]
    )
    print(f"Device: {device}")

    # --- Instance paths ---
    instance_paths = list_instance_paths(config["root"], patterns=(config["pattern"],))
    if config["max_instances"] is not None:
        instance_paths = instance_paths[: int(config["max_instances"])]
    if not instance_paths:
        raise FileNotFoundError(f"No instances found under {config['root']}")
    print(f"Training instances: {len(instance_paths)}")

    # --- Eval instances ---
    eval_paths: List[Path] = []
    if config["eval_root"] is not None:
        eval_paths = list_instance_paths(
            config["eval_root"], patterns=(config["eval_pattern"],)
        )
    print(f"Eval instances: {len(eval_paths)}")

    # --- Optimal makespans for eval ---
    optimal_makespans: Optional[Dict[str, int]] = None
    if config.get("eval_optimal_json"):
        raw = json.loads(Path(config["eval_optimal_json"]).read_text())
        instances = raw.get("instances", {})
        optimal_makespans = {
            Path(k).stem.lower(): int(v["makespan"])
            for k, v in instances.items()
            if isinstance(v, dict) and "makespan" in v
        }

    # --- Model ---
    max_resources = int(config["max_resources"])
    global_dim = global_feature_dim(max_resources)
    candidate_dim = candidate_feature_dim(max_resources)
    critic_dim = critic_feature_dim()

    if config["bc_checkpoint"] is not None:
        print(f"Loading BC checkpoint: {config['bc_checkpoint']}")
        # The BC checkpoint has a CLS-only value head; override the value-head
        # width with the critic runtime-feature dim. Every other weight loads
        # verbatim; the resized value head is reinitialised (BC never trains it).
        base_model = load_policy_checkpoint(
            config["bc_checkpoint"], device=device, dropout=float(config["dropout"]),
            critic_feature_dim=critic_dim,
        )
    else:
        print("No BC checkpoint — initialising from scratch.")
        base_model = BranchingTransformer(
            global_dim=global_dim,
            candidate_dim=candidate_dim,
            d_model=int(config["d_model"]),
            n_heads=int(config["n_heads"]),
            n_layers=int(config["n_layers"]),
            ffn_dim=int(config["ffn_dim"]),
            dropout=float(config["dropout"]),
            critic_feature_dim=critic_dim,
        )

    ac = ActorCritic(base_model).to(device)
    optimizer = optim.AdamW(ac.parameters(), lr=float(config["lr"]))
    print(f"Model params: {sum(p.numel() for p in ac.parameters()):,}")

    # --- Reward config ---
    reward_cfg = RewardConfig(
        alpha=float(config["alpha"]),
        beta1=float(config["beta1"]),
        beta2=float(config["beta2"]),
    )

    # --- Output paths ---
    save_path = Path(config["save_path"])
    save_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = Path(config["checkpoint_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_model_path = save_path.parent / (save_path.stem + "_best.pt")
    eval_log_path = save_path.parent / (save_path.stem + "_eval_log.json")

    # --- Training loop ---
    total_env_steps = int(config["total_env_steps"])
    rollout_horizon = int(config["rollout_horizon"])
    ppo_epochs = int(config["ppo_epochs"])
    minibatches = int(config["minibatches"])
    clip_eps = float(config["clip_eps"])
    gamma = float(config["gamma"])
    gae_lambda = float(config["gae_lambda"])
    tree_gamma = float(config["tree_gamma"])
    tree_keep_open = bool(config["tree_keep_open"])
    alpha = float(config["alpha"])  # flat per-node cost for the tree backup
    beta1 = float(config["beta1"])  # first-incumbent strength bonus weight
    beta2 = float(config["beta2"])  # incumbent-improvement bonus weight
    ent_coef = float(config["ent_coef"])
    vf_coef = float(config["vf_coef"])
    max_grad_norm = float(config["max_grad_norm"])
    target_kl = config.get("target_kl")
    eval_every = int(config["eval_every_steps"])
    time_limit_s = float(config["time_limit_s"])
    dominance = str(config["dominance"])

    env = BranchingEnv(
        instance_source=instance_paths[0],
        max_resources=max_resources,
        time_limit_s=time_limit_s,
        reward_cfg=reward_cfg,
        dominance=dominance,
    )

    buffer = RolloutBuffer()
    global_step = 0
    update_count = 0
    episode_count = 0
    next_eval_step = eval_every
    best_mean_gap = float("inf")
    eval_log: List[Dict] = []
    last_eval_step = -1  # global_step of the most recent eval; guards a duplicate final eval

    # Shuffle instance order each pass
    inst_order = list(range(len(instance_paths)))
    random.shuffle(inst_order)
    inst_idx = 0

    def next_instance() -> Path:
        nonlocal inst_idx, inst_order
        if inst_idx >= len(inst_order):
            inst_order = list(range(len(instance_paths)))
            random.shuffle(inst_order)
            inst_idx = 0
        path = instance_paths[inst_order[inst_idx]]
        inst_idx += 1
        return path

    # Start first episode
    current_instance_path = next_instance()
    obs = env.reset(instance=load_instance(current_instance_path))
    episode_reward = 0.0
    episode_steps = 0
    # Per-step rewards this episode (item 2: variance diagnostic).
    episode_rewards: List[float] = []

    t_start = time.perf_counter()
    print(f"\n{'='*80}")
    print(f"  PPO Training")
    print(f"  total_steps={total_env_steps:,}  rollout={rollout_horizon}  train_instances={len(instance_paths)}  eval_instances={len(eval_paths)}")
    print(f"{'='*80}")
    print(f"[Episode 1] start  {current_instance_path.name}\n")

    while global_step < total_env_steps:

        # ---- Collect ONE complete episode (Design A: rollout == episode) ----
        # The subtree return needs the finished tree, which only exists at
        # `done`. So we step until the episode ends, then capture its tree.
        ac.eval()
        buffer.clear()
        episode_tree = None

        while True:
            with torch.no_grad():
                action_t, log_prob_t, _, value_t = ac.get_action_and_value(obs, device)

            action = int(action_t.item())
            step_out = env.step(action)

            buffer.add(
                obs=obs,
                action=action,
                log_prob=log_prob_t.item(),
                value=value_t.item(),
                reward=step_out.reward,
                done=step_out.done,
                terminated=bool(step_out.info.get("terminated", False)),
                node_id=step_out.info.get("node_id"),
                parent_id=step_out.info.get("parent_id"),
            )

            global_step += 1
            episode_reward += step_out.reward
            episode_steps += 1
            episode_rewards.append(step_out.reward)

            if step_out.done:
                # Capture the finished search tree for the subtree backup.
                episode_tree = env.search_tree()
                stats = env.episode_stats
                episode_count += 1
                elapsed = time.perf_counter() - t_start
                bd = stats.reward_breakdown
                # Item 2: variance of per-step rewards this episode.
                rwd_std = float(np.std(episode_rewards)) if episode_rewards else 0.0
                print(
                    f"[Episode {episode_count}] Done  "
                    f"Instance={current_instance_path.name}  "
                    f"Reason={stats.done_reason}  "
                    f"Steps={episode_steps}  "
                    f"Nodes={stats.nodes_expanded}  "
                    f"Best_Ms={stats.best_makespan}  "
                    f"Inc_Improves={stats.incumbent_improvements}  "
                    f"Reward={episode_reward:+.2f}  "
                    f"rwd_std={rwd_std:.5f}  "
                    f"elapsed={elapsed:.0f}s"
                )
                print()
                break
            else:
                obs = step_out.observation

        # ---- Subtree-return backup (Machine 2) with Machine 1 rewards ----
        # One tree per rollout. Per-node reward r(n) = flat cost (-alpha) plus
        # first-incumbent strength (beta1) and incumbent-improvement (beta2)
        # bonuses placed on the incumbent nodes; the backup sums r(n) over each
        # decision's subtree.
        node_reward_fn = make_node_reward_fn(
            episode_tree,
            alpha=alpha,
            beta1=beta1,
            beta2=beta2,
            root_lb=episode_tree.get("root_lb") if episode_tree else None,
        )
        ta = compute_episode_advantages(
            tree=episode_tree,
            node_ids=buffer.node_ids,
            values=buffer.values,
            reward_fn=node_reward_fn,
            gamma=tree_gamma,
            keep_open=tree_keep_open,
        )
        advantages = torch.tensor(ta.advantages, dtype=torch.float32)
        returns = torch.tensor(ta.returns, dtype=torch.float32)
        valid_mask = torch.tensor(ta.valid, dtype=torch.bool)

        # Valid = transitions whose decision-node subtree closed (complete G).
        # Open subtrees (time-limit frontier path) are dropped under the default
        # truncation policy. Index the full-length tensors by these positions.
        valid_idx = valid_mask.nonzero().squeeze(-1).tolist()
        n_valid = len(valid_idx)
        n_total = len(buffer)

        if n_valid == 0:
            # Whole episode open (e.g. nothing closed before timeout, or an
            # invalid-action episode with no tree). No usable signal — start the
            # next episode without an update.
            print(f"[Update skipped] no closed subtrees this episode "
                  f"(transitions={n_total})\n")
            if global_step < total_env_steps:
                current_instance_path = next_instance()
                obs = env.reset(instance=load_instance(current_instance_path))
                print(f"[Episode {episode_count+1}] start  {current_instance_path.name}\n")
            episode_reward = 0.0
            episode_steps = 0
            episode_rewards = []
            continue

        indices = np.array(valid_idx)

        # Explained variance + ret_std on the VALID subset only.
        values_valid = torch.tensor([buffer.values[i] for i in valid_idx])
        returns_valid = returns[indices]
        var_returns = returns_valid.var()
        explained_var = (
            float("nan") if var_returns.item() == 0.0
            else (1.0 - (returns_valid - values_valid).var() / var_returns).item()
        )
        ret_std = returns_valid.std().item()

        # Normalise advantages over the valid subset (kept full-length; only
        # valid entries are ever read in the update).
        adv_valid = advantages[indices]
        advantages = (advantages - adv_valid.mean()) / (adv_valid.std() + 1e-8)

        # ---- PPO update ----
        # Operates on the VALID transitions only. `indices` holds their buffer
        # positions (set above); advantages/returns are full-length and indexed
        # by those positions, so mb_idx values are real buffer indices.
        ac.train()
        T = n_valid
        mb_size = max(T // minibatches, 1)

        total_pg_loss = total_vf_loss = total_ent = total_kl = 0.0
        n_kl_samples = 0
        update_count += 1
        early_stop = False

        for _ in range(ppo_epochs):
            if early_stop:
                break
            np.random.shuffle(indices)

            for start in range(0, T, mb_size):
                mb_idx = indices[start: start + mb_size]

                mb_log_probs_old = torch.tensor(
                    [buffer.log_probs[i] for i in mb_idx], dtype=torch.float32, device=device
                )
                mb_actions = torch.tensor(
                    [buffer.actions[i] for i in mb_idx], dtype=torch.long, device=device
                )
                mb_advantages = advantages[mb_idx].to(device)
                mb_returns = returns[mb_idx].to(device)

                # Re-evaluate actions under current policy
                mb_log_probs_new_list = []
                mb_entropies = []
                mb_values_new = []

                for i, idx in enumerate(mb_idx):
                    _, lp, ent, val = ac.get_action_and_value(
                        buffer.obs[idx], device, action=mb_actions[i]
                    )
                    mb_log_probs_new_list.append(lp)
                    mb_entropies.append(ent)
                    mb_values_new.append(val)

                mb_log_probs_new = torch.stack(mb_log_probs_new_list)
                mb_entropies_t = torch.stack(mb_entropies)
                mb_values_new_t = torch.stack(mb_values_new)

                # Policy loss (clipped surrogate)
                ratio = torch.exp(mb_log_probs_new - mb_log_probs_old)
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                vf_loss = nn.functional.mse_loss(mb_values_new_t, mb_returns)

                # Entropy bonus
                entropy_loss = -mb_entropies_t.mean()

                loss = pg_loss + vf_coef * vf_loss + ent_coef * entropy_loss

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(ac.parameters(), max_grad_norm)
                optimizer.step()

                total_pg_loss += pg_loss.item()
                total_vf_loss += vf_loss.item()
                total_ent += (-entropy_loss.item())

                # Always compute KL for logging; also use for early stopping
                with torch.no_grad():
                    approx_kl = ((ratio - 1) - torch.log(ratio)).mean().item()
                total_kl += approx_kl
                n_kl_samples += 1

                if target_kl is not None and approx_kl > float(target_kl):
                    early_stop = True
                    break

        n_updates = ppo_epochs * max(T // mb_size, 1)
        mean_kl = total_kl / n_kl_samples if n_kl_samples > 0 else 0.0
        elapsed = time.perf_counter() - t_start
        print(
            f"[Update {update_count}] "
            f"steps={global_step}  "
            f"episodes={episode_count}  "
            f"valid={n_valid}/{n_total}  "
            f"pg={total_pg_loss/n_updates:+.4f}  "
            f"vf={total_vf_loss/n_updates:.4f}  "
            f"ev={explained_var:+.3f}  "
            f"ret_std={ret_std:.4f}  "
            f"ent={total_ent/n_updates:.4f}  "
            f"kl={mean_kl:.4f}  "
            f"elapsed={elapsed:.0f}s"
            f"{'  [KL stop]' if early_stop else ''}"
        )

        # ---- Periodic evaluation ----
        # NOTE: eval runs BEFORE resetting the next episode. The env.reset()
        # starts the solver thread and its wall-clock time-limit clock; eval can
        # take >time_limit_s of wall-clock, so resetting first would make the
        # next episode time out after a single step ("born expired"). Eval uses
        # ac.model directly and does not touch env episode state, so it is safe
        # to run here first.
        if eval_paths and optimal_makespans and global_step >= next_eval_step:
            next_eval_step += eval_every
            last_eval_step = global_step
            ac.eval()
            metrics = evaluate(
                model=ac.model,
                instance_paths=eval_paths,
                max_resources=max_resources,
                time_limit_s=float(config["eval_time_limit_s"]),
                dominance=dominance,
                device=device,
                optimal_makespans=optimal_makespans,
            )

            is_best = metrics["mean_gap"] < best_mean_gap
            best_tag = "  [best]" if is_best else ""
            sep = "-" * 60
            print(f"\n{sep}")
            print(
                f"[Checkpoint] "
                f"steps={global_step}  "
                f"solved={metrics['solved_frac']*100:.1f}%  "
                f"gap={metrics['mean_gap']:.2f}%  "
                f"nodes={metrics['mean_nodes']:.0f}"
                f"{best_tag}"
            )

            # Log eval entry
            log_entry = {"step": global_step, **metrics}
            eval_log.append(log_entry)
            eval_log_path.write_text(json.dumps(eval_log, indent=2))

            # Save periodic checkpoint
            ckpt_path = checkpoint_dir / f"policy_ppo_step{global_step}.pt"
            save_policy_checkpoint(ac.model, str(ckpt_path), extra={"train_config": config, "eval_metrics": metrics})
            print(f"[Checkpoint] saved  → {ckpt_path}")

            # Save best model separately
            if is_best:
                best_mean_gap = metrics["mean_gap"]
                save_policy_checkpoint(ac.model, str(best_model_path), extra={"train_config": config, "eval_metrics": metrics, "step": global_step})
                print(f"[Checkpoint] best   → {best_model_path}  (gap={best_mean_gap:.2f}%)")
            print(f"{sep}\n")

        # ---- Reset for the next episode (Design A: one episode per update) ----
        # Done LAST so the solver's time-limit clock starts immediately before
        # the next collection pass (see eval note above).
        if global_step < total_env_steps:
            current_instance_path = next_instance()
            obs = env.reset(instance=load_instance(current_instance_path))
            print(f"[Episode {episode_count+1}] start  {current_instance_path.name}\n")
        episode_reward = 0.0
        episode_steps = 0
        episode_rewards = []

    # ---- Final model save skipped; best model is saved during evaluation ----
    elapsed = time.perf_counter() - t_start

    # ---- Final evaluation ----
    # Skip if a periodic eval already ran at this exact step on the same model
    # (happens when the step budget is crossed right at an eval boundary) — it
    # would be an identical, redundant eval pass.
    if eval_paths and optimal_makespans and global_step != last_eval_step:
        ac.eval()
        metrics = evaluate(
            model=ac.model,
            instance_paths=eval_paths,
            max_resources=max_resources,
            time_limit_s=float(config["eval_time_limit_s"]),
            dominance=dominance,
            device=device,
            optimal_makespans=optimal_makespans,
        )
        is_best = metrics["mean_gap"] < best_mean_gap
        best_tag = "  [best]" if is_best else ""
        sep = "-" * 60
        print(f"\n{sep}")
        print(
            f"[Final Eval] "
            f"steps={global_step}  "
            f"solved={metrics['solved_frac']*100:.1f}%  "
            f"gap={metrics['mean_gap']:.2f}%  "
            f"nodes={metrics['mean_nodes']:.0f}"
            f"{best_tag}"
        )
        log_entry = {"step": global_step, "final": True, **metrics}
        eval_log.append(log_entry)
        eval_log_path.write_text(json.dumps(eval_log, indent=2))

        if is_best:
            best_mean_gap = metrics["mean_gap"]
            save_policy_checkpoint(ac.model, str(best_model_path), extra={"train_config": config, "eval_metrics": metrics, "step": global_step})
            print(f"[Final Eval] best   → {best_model_path}  (gap={best_mean_gap:.2f}%)")
        print(f"{sep}")

    # ---- Training summary ----
    print(f"\n{'='*80}")
    print(f"  Training complete")
    print(f"  steps={global_step:,}  episodes={episode_count:,}  updates={update_count:,}  elapsed={elapsed:.0f}s")
    if eval_log:
        best_entry = min(eval_log, key=lambda e: e["mean_gap"])
        print(f"  best gap     : {best_entry['mean_gap']:.2f}% at step {best_entry['step']:,}")
        print(f"  best solved  : {best_entry['solved_frac']*100:.1f}% at step {best_entry['step']:,}")
        print(f"  best model   → {best_model_path}")
    print(f"  eval log     → {eval_log_path}")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
