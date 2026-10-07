"""
Validation metrics for the search-effort estimator, expressed in the terms the
estimator is actually judged on.

The objective is a multiplicative band on the prediction:

    band_lo <= N_hat / N <= band_hi

so the headline metric is BAND MEMBERSHIP, not regression error. A model can have
an excellent MAE and still be useless (systematically 3x high on hard instances),
and can have a mediocre MAE while sitting inside the band everywhere that matters.

Three reporting rules follow from that, and each one exists because its absence
hid a real failure in the previous setup:

  1. Report under- and over-violation SEPARATELY. They are different failures with
     different downstream costs (over -> alpha too small -> efficiency signal dies;
     under -> alpha too large -> node cost drowns the other reward channels), so a
     single miss-rate cannot tell you which way to correct.

  2. STRATIFY by regime and by target decile. The dataset is bimodal — instances
     that exhaust inside the horizon have a median node count ~690x smaller than
     those cut off by it — so a pooled hit rate of 85% is entirely consistent with
     95% on the easy mode and 45% on the hard tail. PPO spends its time on the
     hard tail.

  3. Report the REALIZED COST SPREAD. alpha * N = c_target / ratio is the quantity
     the episode return actually inherits, so its spread across the validation set
     is directly comparable to the observed spread in PPO's own G_root logging.

Everything here is numpy-only so it can be used from data tooling without pulling
in torch.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np


def node_ratio(y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
    """
    Node-space ratio N_hat / N from log-space predictions and targets.

    The band is defined on node counts, so it is measured on node counts here even
    though the loss operates on the log residual d = y_hat - y. The two agree
    closely except at very small N, where the +1 in log(1 + N) matters; N is
    floored at 1 so trivially-small instances cannot produce a divide-by-zero.
    """
    n_pred = np.maximum(np.expm1(np.asarray(y_pred, dtype=np.float64)), 0.0)
    n_true = np.maximum(np.expm1(np.asarray(y_true, dtype=np.float64)), 1.0)
    return n_pred / n_true


def _rates(ratio: np.ndarray, band_lo: float, band_hi: float) -> Dict[str, float]:
    """Band hit rate plus the two directional violation rates. Sums to 1.0."""
    if ratio.size == 0:
        return {"band_hit_rate": float("nan"), "under_rate": float("nan"),
                "over_rate": float("nan")}
    under = ratio < band_lo
    over = ratio > band_hi
    return {
        "band_hit_rate": float(np.mean(~(under | over))),
        "under_rate": float(np.mean(under)),
        "over_rate": float(np.mean(over)),
    }


def _spread(ratio: np.ndarray, c_target: float) -> Dict[str, float]:
    """
    Ratio dispersion, and the realized node-cost magnitude it implies.

    median_ratio alone is a weak signal: a model can centre at 1.0 with a p10/p90
    spanning two orders of magnitude. cost_* inverts the ratio into the units the
    return sees, so cost_ratio_p90_p10 is the multiplicative spread of the
    node-cost channel across instances — the number dynamic scaling exists to
    compress.
    """
    if ratio.size == 0:
        return {}
    p10, p50, p90 = (float(v) for v in np.percentile(ratio, [10, 50, 90]))
    safe = np.maximum(ratio, 1e-12)
    cost = c_target / safe
    c10, c50, c90 = (float(v) for v in np.percentile(cost, [10, 50, 90]))
    return {
        "ratio_p10": p10,
        "ratio_p50": p50,
        "ratio_p90": p90,
        "cost_p10": c10,
        "cost_p50": c50,
        "cost_p90": c90,
        "cost_ratio_p90_p10": float(c90 / max(c10, 1e-12)),
    }


def band_metrics(
    y_pred: np.ndarray,
    y_true: np.ndarray,
    solved: Optional[np.ndarray] = None,
    band_lo: float = 0.5,
    band_hi: float = 2.0,
    c_target: float = 1.0,
    n_bins: int = 10,
) -> Dict[str, object]:
    """
    Full metric set for one split.

    Returns a flat set of pooled scalars plus two nested breakdowns:
      by_regime: {"exhausted": {...}, "censored": {...}} when `solved` is given
      by_decile: list of per-decile dicts, ordered easiest -> hardest by target

    The pooled scalars are for the training log and early stopping; the breakdowns
    are what you read to decide whether the model is good enough, and specifically
    whether the hard tail is being served.
    """
    y_pred = np.asarray(y_pred, dtype=np.float64)
    y_true = np.asarray(y_true, dtype=np.float64)
    ratio = node_ratio(y_pred, y_true)

    out: Dict[str, object] = {}
    out.update(_rates(ratio, band_lo, band_hi))
    out.update(_spread(ratio, c_target))
    out["mae_log"] = float(np.mean(np.abs(y_pred - y_true))) if y_pred.size else float("nan")

    if solved is not None:
        solved = np.asarray(solved, dtype=np.float64)
        by_regime: Dict[str, Dict[str, float]] = {}
        for label, flag in (("exhausted", 1.0), ("censored", 0.0)):
            mask = solved == flag
            if not np.any(mask):
                continue
            cell = {"n": int(mask.sum())}
            cell.update(_rates(ratio[mask], band_lo, band_hi))
            cell.update(_spread(ratio[mask], c_target))
            by_regime[label] = cell
        out["by_regime"] = by_regime

    # Deciles by target difficulty. Rank-based so the heavy skew of y cannot dump
    # most rows into one bin the way fixed-width bins on log-nodes would.
    if y_true.size:
        order = np.argsort(y_true, kind="stable")
        ranks = np.empty(y_true.size, dtype=np.int64)
        ranks[order] = np.arange(y_true.size)
        bins = (ranks * n_bins) // y_true.size
        by_decile: List[Dict[str, float]] = []
        for b in range(n_bins):
            mask = bins == b
            if not np.any(mask):
                continue
            cell = {
                "decile": b,
                "n": int(mask.sum()),
                "nodes_median": float(np.median(np.expm1(y_true[mask]))),
            }
            cell.update(_rates(ratio[mask], band_lo, band_hi))
            cell["ratio_p50"] = float(np.median(ratio[mask]))
            by_decile.append(cell)
        out["by_decile"] = by_decile

    return out


def format_band_report(
    metrics: Dict[str, object],
    band_lo: float,
    band_hi: float,
    indent: str = "  ",
) -> str:
    """Render band_metrics as an aligned block for the end-of-training summary."""
    lines: List[str] = []
    lines.append(f"{indent}band = [{band_lo:g}, {band_hi:g}] on N_hat/N")
    lines.append(
        f"{indent}pooled: hit={metrics['band_hit_rate']*100:5.1f}%  "
        f"under={metrics['under_rate']*100:5.1f}%  "
        f"over={metrics['over_rate']*100:5.1f}%  "
        f"ratio p10/p50/p90={metrics.get('ratio_p10', float('nan')):.2f}/"
        f"{metrics.get('ratio_p50', float('nan')):.2f}/"
        f"{metrics.get('ratio_p90', float('nan')):.2f}"
    )
    cost_spread = metrics.get("cost_ratio_p90_p10")
    if cost_spread is not None:
        lines.append(f"{indent}realized node-cost spread (p90/p10) = {cost_spread:.1f}x")

    by_regime = metrics.get("by_regime") or {}
    for label, cell in by_regime.items():
        lines.append(
            f"{indent}{label:>9}: n={cell['n']:5d}  hit={cell['band_hit_rate']*100:5.1f}%  "
            f"under={cell['under_rate']*100:5.1f}%  over={cell['over_rate']*100:5.1f}%  "
            f"ratio_p50={cell.get('ratio_p50', float('nan')):.2f}"
        )

    by_decile = metrics.get("by_decile") or []
    if by_decile:
        lines.append(f"{indent}by target decile (easiest -> hardest):")
        lines.append(f"{indent}  dec     n  nodes_med    hit   under    over  ratio_p50")
        for cell in by_decile:
            lines.append(
                f"{indent}  {cell['decile']:3d} {cell['n']:5d} {cell['nodes_median']:10.0f} "
                f"{cell['band_hit_rate']*100:6.1f}% {cell['under_rate']*100:6.1f}% "
                f"{cell['over_rate']*100:6.1f}% {cell['ratio_p50']:10.2f}"
            )
    return "\n".join(lines)
