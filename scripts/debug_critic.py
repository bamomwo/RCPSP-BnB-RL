"""
Critic debug rollout — FROZEN, READ-ONLY (no training).

Runs a single policy rollout on one RCPSP instance through the SAME machinery
training uses (BranchingEnv + tree-return backup), then reports how the critic's
value estimates compare to the realized subtree returns at each node.

There is NO weight update here: the model is loaded from a checkpoint, put in
eval() mode, and every forward pass is wrapped in torch.no_grad(). This is the
data-collection + return-computation half of a training iteration with the
gradient step removed.

Normalization note
------------------
The critic is trained to predict NORMALIZED returns:  (raw_return - mean) / std,
where mean/std come from the running value-normaliser (RunningMeanStd) saved in
the checkpoint under extra["value_norm"]. So the critic's raw output already
lives in normalized-return space. To compare against the tree-backup returns
(which are in RAW units) we must either normalize the returns or denormalize the
value. This script shows BOTH spaces so nothing is ambiguous.

Usage
-----
    python3 scripts/debug_critic.py \
        --checkpoint models/train_1/policy_ppo_best.pt \
        --instance data/eval/J30/j30_480/<name>.RCP \
        --time-limit-s 60

Optional:
    --alpha / --beta1 / --beta2         reward weights (default to training values)
    --gamma-cost / --gamma-bonus        subtree discount factors
    --dominance                         dominance spec (default set_based)
    --sample                            sample actions instead of greedy
    --seed                              RNG seed when --sample is used
    --max-print                         nodes to show per depth bucket (top/mid/bottom)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from torch.distributions import Categorical

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from rcpsp_bb_rl.data.parsing import load_instance  # noqa: E402
from rcpsp_bb_rl.ml.models import load_policy_checkpoint  # noqa: E402
from rcpsp_bb_rl.ml.rl import BranchingEnv  # noqa: E402
from rcpsp_bb_rl.ml.rl.tree_return import (  # noqa: E402
    compute_episode_advantages_decoupled,
    compute_subtree_returns,
    make_cost_reward_fn,
    make_bonus_reward_fn,
)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Frozen, read-only critic debug rollout for one RCPSP instance.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--checkpoint", required=True, help="Path to a policy checkpoint (.pt).")
    p.add_argument("--instance", required=True, help="Path to one RCPSP instance file.")
    p.add_argument("--time-limit-s", type=float, default=60.0, help="Per-solve time limit.")
    p.add_argument("--dominance", default="set_based", help="Dominance spec.")
    p.add_argument("--max-resources", type=int, default=4, help="Resource feature dim.")
    p.add_argument("--device", default="cpu", help="Torch device (cpu|cuda).")
    # Reward / backup params (default to the training config values)
    p.add_argument("--alpha", type=float, default=0.001, help="Cost per node.")
    p.add_argument("--beta1", type=float, default=5.0, help="First-incumbent bonus weight.")
    p.add_argument("--beta2", type=float, default=2.0, help="Improvement bonus weight.")
    p.add_argument("--gamma-cost", type=float, default=0.85, help="Cost-channel discount.")
    p.add_argument("--gamma-bonus", type=float, default=1.0, help="Bonus-channel discount.")
    p.add_argument("--keep-open", action="store_true", help="Treat open subtrees as valid.")
    # Action selection
    p.add_argument("--sample", action="store_true", help="Sample actions (default: greedy argmax).")
    p.add_argument("--seed", type=int, default=0, help="RNG seed when --sample is set.")
    # Reporting
    p.add_argument("--max-print", type=int, default=15, help="Nodes to show per depth bucket.")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Checkpoint value-norm recovery
# ---------------------------------------------------------------------------

def load_value_norm(path: str, device: str) -> tuple[float, float]:
    """Recover the (mean, std) the critic was trained against. Falls back to
    (0, 1) — the identity — if the checkpoint predates value-norm saving, in
    which case normalized == raw and the two columns will match."""
    ckpt = torch.load(path, map_location=device)
    vn = ckpt.get("value_norm") if isinstance(ckpt, dict) else None
    if not vn:
        print("[warn] checkpoint has no 'value_norm' — assuming identity (mean=0, std=1). "
              "Normalized and raw columns will be identical.")
        return 0.0, 1.0
    mean = float(vn.get("mean", 0.0))
    var = vn.get("var")
    std = float(vn["std"]) if "std" in vn else float(var) ** 0.5 if var is not None else 1.0
    return mean, max(std, 1e-8)


# ---------------------------------------------------------------------------
# Rollout (frozen, read-only)
# ---------------------------------------------------------------------------

def rollout(
    model,
    env: BranchingEnv,
    instance,
    device: torch.device,
    sample: bool,
) -> tuple[List[int], List[float], List[int], List[int]]:
    """
    Run one episode. Returns per-transition lists:
      node_ids, values(normalized space), depths, cand_counts(R)
    No gradients, no updates.
    """
    node_ids: List[int] = []
    values: List[float] = []
    depths: List[int] = []
    cand_counts: List[int] = []

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
            dist = Categorical(logits=logits)
            action = dist.sample() if sample else torch.argmax(logits)

        cand_counts.append(int(obs["candidate_feats"].shape[0]))
        values.append(float(value.item()))

        step_out = env.step(int(action.item()))
        node_ids.append(step_out.info.get("node_id"))
        depths.append(int(step_out.info.get("depth", -1)))

        if step_out.done:
            break
        obs = step_out.observation

    return node_ids, values, depths, cand_counts


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def bucket_indices(depths: List[int], valid: List[bool], k: int) -> Dict[str, List[int]]:
    """Pick up to k valid transition indices from the top, middle, and bottom of
    the tree (by depth). Only valid (closed / trainable) nodes are considered."""
    valid_idx = [i for i, v in enumerate(valid) if v]
    if not valid_idx:
        return {"top": [], "mid": [], "bottom": []}
    # Sort valid transitions by depth (shallow -> deep).
    ordered = sorted(valid_idx, key=lambda i: depths[i])
    n = len(ordered)
    top = ordered[:k]
    bottom = ordered[-k:]
    mid_start = max(0, n // 2 - k // 2)
    mid = ordered[mid_start: mid_start + k]
    return {"top": top, "mid": mid, "bottom": bottom}


def main() -> None:
    args = parse_args()
    device = torch.device(
        "cpu" if (args.device == "cuda" and not torch.cuda.is_available()) else args.device
    )
    if args.sample:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

    inst_path = Path(args.instance)
    instance = load_instance(inst_path)

    # --- Load frozen model + value normalization stats ---
    model = load_policy_checkpoint(args.checkpoint, device=device)
    model.eval()
    ret_mean, ret_std = load_value_norm(args.checkpoint, args.device)

    env = BranchingEnv(
        instance_source=instance,
        max_resources=args.max_resources,
        time_limit_s=args.time_limit_s,
        dominance=args.dominance,
    )

    # --- Rollout (frozen) ---
    node_ids, values, depths, cand_counts = rollout(
        model, env, instance, device, args.sample
    )
    stats = env.episode_stats
    tree = env.search_tree()

    # --- Returns via the SAME functions training uses ---
    cost_fn = make_cost_reward_fn(alpha=args.alpha)
    root_lb = tree.get("root_lb") if tree else None
    bonus_fn = make_bonus_reward_fn(tree, beta1=args.beta1, beta2=args.beta2, root_lb=root_lb)

    ta = compute_episode_advantages_decoupled(
        tree=tree,
        node_ids=node_ids,
        values=values,
        cost_reward_fn=cost_fn,
        bonus_reward_fn=bonus_fn,
        gamma_cost=args.gamma_cost,
        gamma_bonus=args.gamma_bonus,
        keep_open=args.keep_open,
    )
    raw_returns = ta.returns          # RAW units (cost + bonus)
    valid = ta.valid

    # Channel split (per node_id) — for interpretability.
    cost_res = compute_subtree_returns(tree, cost_fn, args.gamma_cost) if tree else None
    bonus_res = compute_subtree_returns(tree, bonus_fn, args.gamma_bonus) if tree else None

    # --- Per-instance summary ---
    n_total = len(node_ids)
    n_valid = sum(valid)
    valid_frac = n_valid / n_total if n_total else 0.0

    # Work in normalized space for EV (matches the loss the critic minimizes).
    ret_norm = [(r - ret_mean) / ret_std for r in raw_returns]
    val_arr = np.array([values[i] for i in range(n_total) if valid[i]], dtype=np.float64)
    ret_norm_arr = np.array([ret_norm[i] for i in range(n_total) if valid[i]], dtype=np.float64)
    raw_ret_arr = np.array([raw_returns[i] for i in range(n_total) if valid[i]], dtype=np.float64)

    if val_arr.size > 0 and ret_norm_arr.var() > 0:
        ev = 1.0 - ((ret_norm_arr - val_arr).var() / ret_norm_arr.var())
    else:
        ev = float("nan")
    adv_norm = ret_norm_arr - val_arr if val_arr.size else np.array([])
    mean_abs_adv = float(np.mean(np.abs(adv_norm))) if adv_norm.size else float("nan")

    sep = "=" * 78
    print(f"\n{sep}")
    print(f"  CRITIC DEBUG (frozen, read-only)  —  {inst_path.name}")
    print(f"{sep}")
    print(f"  done_reason      : {stats.done_reason}")
    print(f"  nodes_expanded   : {stats.nodes_expanded}")
    print(f"  best_makespan    : {stats.best_makespan}")
    print(f"  inc_improvements : {stats.incumbent_improvements}")
    print(f"  transitions      : {n_total}")
    print(f"  valid (closed)   : {n_valid}  ({valid_frac*100:.1f}% of transitions)")
    print(f"  action mode      : {'sampled (seed=%d)' % args.seed if args.sample else 'greedy'}")
    print(f"  value_norm        : mean={ret_mean:.4f}  std={ret_std:.4f}")
    print(f"  --- critic fit on VALID nodes (normalized space) ---")
    print(f"  explained_var    : {ev:+.4f}")
    print(f"  mean|advantage|  : {mean_abs_adv:.4f}")
    if raw_ret_arr.size:
        print(f"  raw return range : [{raw_ret_arr.min():+.3f}, {raw_ret_arr.max():+.3f}]  "
              f"mean={raw_ret_arr.mean():+.3f}")
    print(f"{sep}\n")

    # --- Per-node dump: top / mid / bottom of the tree ---
    buckets = bucket_indices(depths, valid, args.max_print)
    header = (
        f"  {'node':>7} {'depth':>5} {'R':>4} {'closed':>6} "
        f"{'V(norm)':>9} {'G(norm)':>9} {'adv':>8} | "
        f"{'V(raw)':>9} {'G(raw)':>9} {'G_cost':>9} {'G_bonus':>9}"
    )
    for label in ("top", "mid", "bottom"):
        idxs = buckets[label]
        print(f"  [{label.upper()} of tree]  ({len(idxs)} nodes)")
        print(header)
        print("  " + "-" * (len(header) - 2))
        for i in idxs:
            nid = node_ids[i]
            v_norm = values[i]
            g_raw = raw_returns[i]
            g_norm = (g_raw - ret_mean) / ret_std
            adv = g_norm - v_norm
            v_raw = v_norm * ret_std + ret_mean
            g_cost = cost_res.G.get(nid) if (cost_res and nid is not None) else None
            g_bonus = bonus_res.G.get(nid) if (bonus_res and nid is not None) else None
            closed_flag = "yes" if valid[i] else "no"
            gc = f"{g_cost:+.3f}" if g_cost is not None else "  -  "
            gb = f"{g_bonus:+.3f}" if g_bonus is not None else "  -  "
            print(
                f"  {nid:>7} {depths[i]:>5} {cand_counts[i]:>4} {closed_flag:>6} "
                f"{v_norm:>+9.3f} {g_norm:>+9.3f} {adv:>+8.3f} | "
                f"{v_raw:>+9.3f} {g_raw:>+9.3f} {gc:>9} {gb:>9}"
            )
        print()


if __name__ == "__main__":
    main()
