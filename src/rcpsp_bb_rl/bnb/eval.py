from __future__ import annotations

import statistics
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import torch

from rcpsp_bb_rl.bnb.core import BnBSolver
from rcpsp_bb_rl.bnb.policy_guidance import make_policy_order_fn
from rcpsp_bb_rl.data.dataset import list_instance_paths
from rcpsp_bb_rl.data.parsing import load_instance


def run_instance(
    path: Path,
    max_nodes: int,
    policy_model=None,
    policy_device: Optional[torch.device] = None,
    policy_max_resources: int = 4,
    time_limit_s: Optional[float] = None,
    return_bounds: bool = False,
    return_debug: bool = False,
) -> Tuple[int | None, float] | Tuple[int | None, int | None, float] | Tuple[int | None, int | None, float, int | None, int | None]:
    instance = load_instance(path)
    solver = BnBSolver(instance)
    order_fn = None
    if policy_model is not None:
        order_fn = make_policy_order_fn(
            instance=instance,
            model=policy_model,
            max_resources=policy_max_resources,
            device=policy_device or "cpu",
            predecessors=solver.predecessors,
        )

    start = time.perf_counter()
    result = solver.solve(max_nodes=max_nodes, order_ready_fn=order_fn, time_limit_s=time_limit_s)
    elapsed = time.perf_counter() - start
    if not return_bounds:
        return result.best_makespan, elapsed

    pending_lbs = [n.lower_bound for n in result.nodes if n.status == "pending"]
    if pending_lbs:
        lb_global_final = min(pending_lbs)
    elif result.best_makespan is not None:
        lb_global_final = result.best_makespan
    else:
        lb_global_final = None

    lb_root = result.nodes[0].lower_bound if result.nodes else None
    lower_bound = lb_global_final
    if return_debug:
        return result.best_makespan, lower_bound, elapsed, lb_root, lb_global_final
    return result.best_makespan, lower_bound, elapsed


def _safe_mean(values: Iterable[float]) -> Optional[float]:
    vals = list(values)
    if not vals:
        return None
    return float(statistics.mean(vals))


def _safe_median(values: Iterable[float]) -> Optional[float]:
    vals = list(values)
    if not vals:
        return None
    return float(statistics.median(vals))


def evaluate_bnb_suite(
    paths: List[Path],
    max_nodes: int,
    policy_model=None,
    bc_model=None,
    policy_device: torch.device | str = "cpu",
    policy_max_resources: int = 4,
    time_limit_s: Optional[float] = None,
    progress_every: int = 0,
) -> Dict[str, object]:
    policy_device = torch.device(policy_device)
    if policy_model is not None:
        policy_model = policy_model.to(policy_device).eval()
    if bc_model is not None:
        bc_model = bc_model.to(policy_device).eval()

    rows: List[Dict[str, object]] = []
    start_time = time.perf_counter()
    for idx, path in enumerate(paths, start=1):
        entry: Dict[str, object] = {"instance": path.name}

        native_mk, native_t = run_instance(path, max_nodes, time_limit_s=time_limit_s)
        entry["native_makespan"] = native_mk
        entry["native_time"] = native_t

        if policy_model is not None:
            policy_mk, policy_t = run_instance(
                path,
                max_nodes,
                policy_model=policy_model,
                policy_device=policy_device,
                policy_max_resources=policy_max_resources,
                time_limit_s=time_limit_s,
            )
            entry["policy_makespan"] = policy_mk
            entry["policy_time"] = policy_t

        if bc_model is not None:
            bc_mk, bc_t = run_instance(
                path,
                max_nodes,
                policy_model=bc_model,
                policy_device=policy_device,
                policy_max_resources=policy_max_resources,
                time_limit_s=time_limit_s,
            )
            entry["bc_makespan"] = bc_mk
            entry["bc_time"] = bc_t

        rows.append(entry)

        if progress_every > 0 and (idx % progress_every == 0 or idx == len(paths)):
            elapsed = time.perf_counter() - start_time
            rate = idx / elapsed if elapsed > 0 else 0.0
            remaining = len(paths) - idx
            eta = remaining / rate if rate > 0 else float("inf")
            eta_str = f"{eta/60:.1f}m" if eta != float("inf") else "unknown"
            print(f"[eval progress] {idx}/{len(paths)} | elapsed {elapsed/60:.1f}m | eta {eta_str}")

    def _wins_losses_ties(a_key: str, b_key: str) -> Dict[str, int]:
        wins = losses = ties = 0
        for r in rows:
            a = r.get(a_key)
            b = r.get(b_key)
            if a is None or b is None:
                continue
            if a < b:
                wins += 1
            elif a > b:
                losses += 1
            else:
                ties += 1
        return {"wins": wins, "losses": losses, "ties": ties, "total": wins + losses + ties}

    def _pct_wins(wlt: Dict[str, int]) -> Optional[float]:
        total = wlt.get("total", 0)
        if total == 0:
            return None
        return wlt["wins"] / total * 100.0

    runtimes = {
        "native": [r["native_time"] for r in rows if r.get("native_time") is not None],
        "policy": [r["policy_time"] for r in rows if r.get("policy_time") is not None],
        "bc": [r["bc_time"] for r in rows if r.get("bc_time") is not None],
    }

    wins_vs_native = _wins_losses_ties("policy_makespan", "native_makespan")
    wins_vs_bc = _wins_losses_ties("policy_makespan", "bc_makespan")

    summary = {
        "instances": len(rows),
        "wins_vs_native": wins_vs_native,
        "wins_vs_native_pct": _pct_wins(wins_vs_native),
        "wins_vs_bc": wins_vs_bc,
        "wins_vs_bc_pct": _pct_wins(wins_vs_bc),
        "runtime_mean": {k: _safe_mean(v) for k, v in runtimes.items()},
        "runtime_median": {k: _safe_median(v) for k, v in runtimes.items()},
    }

    return {
        "summary": summary,
        "instances": rows,
    }


def list_eval_instances(root: str, pattern: str, limit: Optional[int] = None) -> List[Path]:
    paths = list_instance_paths(root, patterns=(pattern,))
    if limit is not None:
        paths = paths[: int(limit)]
    return paths
