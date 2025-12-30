from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt


STEP_RE = re.compile(r"eval_step_(\d+)\.json$")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot PPO eval stats over checkpoints.")
    p.add_argument(
        "--eval-dir",
        default="reports/eval_stats",
        help="Directory containing eval stats JSON files.",
    )
    p.add_argument(
        "--compact-file",
        default="wins_vs_native.json",
        help="Compact eval file name within eval-dir (JSON list).",
    )
    p.add_argument(
        "--output",
        default="reports/eval_plots/eval_trends.png",
        help="Output PNG path.",
    )
    return p.parse_args()


def _extract_step(path: Path) -> Optional[int]:
    match = STEP_RE.search(path.name)
    if not match:
        return None
    return int(match.group(1))


def _load_eval_files(eval_dir: Path) -> List[Tuple[int, Dict[str, object]]]:
    entries: List[Tuple[int, Dict[str, object]]] = []
    for path in sorted(eval_dir.glob("eval_step_*.json")):
        step = _extract_step(path)
        if step is None:
            continue
        data = json.loads(path.read_text())
        entries.append((step, data))
    entries.sort(key=lambda x: x[0])
    return entries


def _load_compact_file(path: Path) -> List[Tuple[int, Dict[str, object]]]:
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError:
        return []
    if not isinstance(data, list):
        return []
    entries: List[Tuple[int, Dict[str, object]]] = []
    for row in data:
        if not isinstance(row, dict):
            continue
        step = row.get("step")
        if step is None:
            continue
        entries.append((int(step), row))
    entries.sort(key=lambda x: x[0])
    return entries


def _rel_gain(base: Optional[float], policy: Optional[float]) -> Optional[float]:
    if base is None or policy is None or base == 0:
        return None
    return (float(base) - float(policy)) / float(base)


def _wlt(policy: Optional[float], base: Optional[float]) -> str:
    if policy is None or base is None:
        return "skip"
    if policy < base:
        return "win"
    if policy > base:
        return "loss"
    return "tie"


def _mean(values: List[float]) -> Optional[float]:
    if not values:
        return None
    return sum(values) / len(values)


def _median(values: List[float]) -> Optional[float]:
    if not values:
        return None
    vals = sorted(values)
    mid = len(vals) // 2
    if len(vals) % 2:
        return vals[mid]
    return 0.5 * (vals[mid - 1] + vals[mid])


def _ratio(num: Optional[float], den: Optional[float]) -> Optional[float]:
    if num is None or den is None or den == 0:
        return None
    return float(num) / float(den)


def summarize_eval(data: Dict[str, object]) -> Dict[str, object]:
    instances = data.get("results", {}).get("instances", [])
    gains_native: List[float] = []
    gains_bc: List[float] = []
    ratio_native: List[float] = []
    ratio_bc: List[float] = []
    wlt_native = {"win": 0, "loss": 0, "tie": 0}
    wlt_bc = {"win": 0, "loss": 0, "tie": 0}

    for row in instances:
        native = row.get("native_makespan")
        bc = row.get("bc_makespan")
        policy = row.get("policy_makespan")
        policy_time = row.get("policy_time")
        native_time = row.get("native_time")
        bc_time = row.get("bc_time")

        gain_n = _rel_gain(native, policy)
        if gain_n is not None:
            gains_native.append(gain_n)
        gain_bc = _rel_gain(bc, policy)
        if gain_bc is not None:
            gains_bc.append(gain_bc)

        r_n = _ratio(policy_time, native_time)
        if r_n is not None:
            ratio_native.append(r_n)
        r_bc = _ratio(policy_time, bc_time)
        if r_bc is not None:
            ratio_bc.append(r_bc)

        w_n = _wlt(policy, native)
        if w_n in wlt_native:
            wlt_native[w_n] += 1
        w_bc = _wlt(policy, bc)
        if w_bc in wlt_bc:
            wlt_bc[w_bc] += 1

    total_native = sum(wlt_native.values()) or 1
    total_bc = sum(wlt_bc.values()) or 1

    return {
        "gain_native_mean": _mean(gains_native),
        "gain_native_median": _median(gains_native),
        "gain_bc_mean": _mean(gains_bc),
        "gain_bc_median": _median(gains_bc),
        "win_rate_native": wlt_native["win"] / total_native,
        "tie_rate_native": wlt_native["tie"] / total_native,
        "win_rate_bc": wlt_bc["win"] / total_bc,
        "tie_rate_bc": wlt_bc["tie"] / total_bc,
        "runtime_ratio_native_mean": _mean(ratio_native),
        "runtime_ratio_bc_mean": _mean(ratio_bc),
    }


def plot_series(steps: List[int], values: List[Optional[float]], label: str) -> None:
    xs = [s for s, v in zip(steps, values) if v is not None]
    ys = [v for v in values if v is not None]
    if xs and ys:
        plt.plot(xs, ys, marker="o", linewidth=1.5, label=label)


def main() -> None:
    args = parse_args()
    eval_dir = Path(args.eval_dir)
    output_path = Path(args.output)

    compact_path = eval_dir / args.compact_file
    entries = _load_compact_file(compact_path)
    compact_mode = bool(entries)
    if not entries:
        raise FileNotFoundError(f"No eval stats found in {compact_path}.")

    steps: List[int] = []
    stats: List[Dict[str, object]] = []
    for step, data in entries:
        steps.append(step)
        if compact_mode:
            stats.append(data)
        else:
            stats.append(summarize_eval(data))

    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(11, 7))

    win_pct = [s.get("wins_vs_native_pct") for s in stats]
    win_rate = [v / 100.0 if isinstance(v, (int, float)) else None for v in win_pct]
    plt.subplot(1, 1, 1)
    plot_series(steps, win_rate, "Win rate vs native")
    plt.ylabel("Win rate")
    plt.xlabel("Env steps")
    plt.title("PPO Policy Trend (Wins vs Native)")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"Saved plot to {output_path}")


if __name__ == "__main__":
    main()
