import argparse
import statistics
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch

# Ensure the src directory is importable when running from the repo root.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from rcpsp_bb_rl.bnb.eval import run_instance  # noqa: E402
from rcpsp_bb_rl.bnb.teacher import solve_optimal_schedule  # noqa: E402
from rcpsp_bb_rl.data.parsing import load_instance  # noqa: E402
from rcpsp_bb_rl.data.dataset import list_instance_paths  # noqa: E402
from rcpsp_bb_rl.models import load_policy_checkpoint  # noqa: E402


def run_ortools(
    path: Path,
    time_limit_s: Optional[float] = None,
) -> Tuple[int | None, float | None]:
    """Solve with OR-Tools CP-SAT and return makespan and wall time (or None on failure)."""
    instance = load_instance(path)
    start = time.perf_counter()
    try:
        starts = solve_optimal_schedule(instance, time_limit_s=time_limit_s)
        makespan = max(starts[aid] + instance.activities[aid].duration for aid in starts)
    except Exception:
        return None, None
    elapsed = time.perf_counter() - start
    return makespan, elapsed






def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run B&B on a set of RCPSP instances and print a summary table."
    )
    parser.add_argument(
        "--root",
        default="data/j30rcp",
        help="Directory containing RCPSP instance files.",
    )
    parser.add_argument(
        "--pattern",
        default="*.RCP",
        help="Glob pattern to select instance files (applied recursively under root).",
    )
    parser.add_argument(
        "--max-nodes",
        type=int,
        default=2000,
        help="Maximum number of nodes to expand per instance.",
    )
    parser.add_argument(
        "--policy",
        default=None,
        help="Optional path to a policy checkpoint (.pt) to also evaluate policy-guided branching.",
    )
    parser.add_argument(
        "--policy-device",
        default="cpu",
        help="Torch device for running the policy (e.g., cpu or cuda:0).",
    )
    parser.add_argument(
        "--policy-max-resources",
        type=int,
        default=4,
        help="Pad/truncate resource dimensions to this size for policy features.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit on number of instances to process.",
    )
    parser.add_argument(
        "--output-dir",
        default=PROJECT_ROOT / "reports" / "results",
        help="Directory to write summary text files (default: reports/results).",
    )
    parser.add_argument(
        "--output-name",
        default="summary.txt",
        help="Filename for the summary text output.",
    )
    parser.add_argument(
        "--with-ortools",
        action="store_true",
        help="Also evaluate an OR-Tools CP-SAT solve for comparison.",
    )
    parser.add_argument(
        "--only-ortools",
        action="store_true",
        help="Skip native/policy runs and only evaluate OR-Tools.",
    )
    parser.add_argument(
        "--time-limit",
        type=float,
        default=None,
        help="Optional wall-clock time limit (seconds) applied to all solvers (native, policy, OR-Tools).",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=50,
        help="Print progress every N instances (default: 50). Set <=0 to disable.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    policy_model = None
    policy_device = torch.device(args.policy_device)
    if args.policy:
        policy_model = load_policy_checkpoint(args.policy, device=policy_device)
    time_limit_all = args.time_limit

    include_native = not args.only_ortools
    include_policy = policy_model is not None and include_native
    include_ortools = args.with_ortools or args.only_ortools

    paths: List[Path] = list_instance_paths(args.root, patterns=(args.pattern,))
    if args.limit is not None:
        paths = paths[: args.limit]

    rows = []
    summary_rows: List[Dict[str, object]] = []
    start_time = time.perf_counter()
    for idx, path in enumerate(paths, start=1):
        row_vals = [idx, path.name]
        entry: Dict[str, object] = {"instance": path.name}

        if include_native:
            best_naive, time_naive = run_instance(path, args.max_nodes, time_limit_s=time_limit_all)
            row_vals.extend([best_naive, time_naive])
            entry["native_makespan"] = best_naive
            entry["native_time"] = time_naive

        if include_policy:
            best_policy, time_policy = run_instance(
                path,
                args.max_nodes,
                policy_model=policy_model,
                policy_device=policy_device,
                policy_max_resources=args.policy_max_resources,
                time_limit_s=time_limit_all,
            )
            row_vals.extend([best_policy, time_policy])
            entry["policy_makespan"] = best_policy
            entry["policy_time"] = time_policy

        if include_ortools:
            best_ortools, time_ortools = run_ortools(
                path,
                time_limit_s=time_limit_all,
            )
            row_vals.extend([best_ortools, time_ortools])
            entry["ortools_makespan"] = best_ortools
            entry["ortools_time"] = time_ortools

        rows.append(row_vals)
        summary_rows.append(entry)
        if args.progress_every > 0 and (idx % args.progress_every == 0 or idx == len(paths)):
            elapsed = time.perf_counter() - start_time
            rate = idx / elapsed if elapsed > 0 else 0.0
            remaining = len(paths) - idx
            eta = remaining / rate if rate > 0 else float("inf")
            eta_str = f"{eta/60:.1f}m" if eta != float("inf") else "unknown"
            print(f"[progress] {idx}/{len(paths)} | elapsed {elapsed/60:.1f}m | eta {eta_str}")

    # Print a simple table similar to benchmark reports.
    lines = []

    # Dynamically configure columns based on requested comparisons.
    header = ["#", "Instance"]
    time_cols = set()

    if include_native:
        header.extend(["Makespan(native)", "CPU-Time[native]"])
        time_cols.add(len(header) - 1)
    if include_policy:
        header.extend(["Makespan(policy)", "CPU-Time[policy]"])
        time_cols.add(len(header) - 1)
    if include_ortools:
        header.extend(["Makespan(ortools)", "CPU-Time[ortools]"])
        time_cols.add(len(header) - 1)

    # Compute column widths.
    col_widths = [len(h) for h in header]
    for row in rows:
        for i, val in enumerate(row[: len(header)]):
            if i in time_cols:
                if val is None:
                    col_widths[i] = max(col_widths[i], 1)
                else:
                    col_widths[i] = max(col_widths[i], len(f"{val:.2f}"))
            else:
                col_widths[i] = max(col_widths[i], len(str(val)) if val is not None else 1)

    def fmt_row(cols, is_header: bool = False) -> str:
        parts = []
        for i, val in enumerate(cols):
            width = col_widths[i]
            if i == 1:  # Instance column left-aligned
                parts.append(str(val).ljust(width))
                continue
            if i in time_cols:
                if is_header or isinstance(val, str):
                    parts.append(str(val).ljust(width))
                elif val is None:
                    parts.append("-".ljust(width))
                else:
                    parts.append(f"{val:>{width}.2f}")
            else:
                parts.append(str(val if val is not None else "-").rjust(width))
        return "  ".join(parts)

    comparisons = []
    if include_native:
        comparisons.append("native")
    if include_policy:
        comparisons.append("policy")
    if include_ortools:
        comparisons.append("ortools")
    lines.append("Benchmark-style summary (" + " vs ".join(comparisons) + ")")

    lines.append("-" * (sum(col_widths) + 2 * (len(col_widths) - 1)))
    lines.append(fmt_row(header, is_header=True))
    lines.append("-" * (sum(col_widths) + 2 * (len(col_widths) - 1)))
    for row in rows:
        lines.append(fmt_row(row[: len(header)]))

    # Aggregate comparisons.
    def compute_gaps(
        key: str,
        baseline_key: str = "ortools_makespan",
        rows: List[Dict[str, object]] = summary_rows,
    ) -> List[float]:
        gaps: List[float] = []
        for r in rows:
            base = r.get(baseline_key)
            val = r.get(key)
            if base is None or val is None:
                continue
            if base == 0:
                continue
            gaps.append((float(val) - float(base)) / float(base))
        return gaps

    gap_native = compute_gaps("native_makespan") if include_native and include_ortools else []
    gap_policy = compute_gaps("policy_makespan") if include_policy and include_ortools else []

    wins = ties = losses = 0
    if include_policy and include_native:
        for r in summary_rows:
            n = r.get("native_makespan")
            p = r.get("policy_makespan")
            if n is None or p is None:
                continue
            if p < n:
                wins += 1
            elif p > n:
                losses += 1
            else:
                ties += 1

    def success_rate(key: str) -> float:
        solved = sum(1 for r in summary_rows if r.get(key) is not None)
        return solved / len(summary_rows) if summary_rows else 0.0

    if include_policy or include_native:
        lines.append("")
        lines.append("Aggregates:")

        show_wins = include_policy and include_native
        agg_header = ["Solver", "Median Gap", "Mean Gap"]
        if show_wins:
            agg_header.append("% Wins vs Native")
        agg_header.append("% <= 5% Gap")

        comparisons_total = wins + losses + ties
        wins_pct = (wins / comparisons_total * 100) if comparisons_total else None

        def fmt_pct(value: Optional[float], digits: int = 1) -> str:
            if value is None:
                return "-"
            if digits <= 0:
                return f"{value:.0f}%"
            return f"{value:.{digits}f}%"

        def gap_stat(gaps: List[float], stat_fn) -> Optional[float]:
            if not gaps:
                return None
            return stat_fn(gaps) * 100

        def pct_within(gaps: List[float], threshold: float) -> Optional[float]:
            if not gaps:
                return None
            return sum(g <= threshold for g in gaps) / len(gaps) * 100

        def build_row(label: str, gaps: List[float], wins_value: Optional[float] = None) -> List[str]:
            row = [
                label,
                fmt_pct(gap_stat(gaps, statistics.median)),
                fmt_pct(gap_stat(gaps, statistics.mean)),
            ]
            if show_wins:
                row.append(fmt_pct(wins_value, digits=0))
            row.append(fmt_pct(pct_within(gaps, 0.05), digits=0))
            return row

        agg_rows: List[List[str]] = []
        if include_native:
            agg_rows.append(build_row("Native B&B", gap_native))
        if include_policy:
            agg_rows.append(build_row("Policy B&B", gap_policy, wins_pct))

        agg_col_widths = [len(h) for h in agg_header]
        for row in agg_rows:
            for i, val in enumerate(row):
                agg_col_widths[i] = max(agg_col_widths[i], len(val))

        def fmt_agg_row(row: List[str]) -> str:
            parts: List[str] = []
            for i, val in enumerate(row):
                width = agg_col_widths[i]
                if i == 0:
                    parts.append(val.ljust(width))
                else:
                    parts.append(val.rjust(width))
            return "  ".join(parts)

        lines.append("-" * (sum(agg_col_widths) + 2 * (len(agg_col_widths) - 1)))
        lines.append(fmt_agg_row(agg_header))
        lines.append("-" * (sum(agg_col_widths) + 2 * (len(agg_col_widths) - 1)))
        for row in agg_rows:
            lines.append(fmt_agg_row(row))

        if include_ortools:
            lines.append("Gaps measured relative to OR-Tools solutions when available.")
        else:
            lines.append("Gaps shown as '-' when no OR-Tools baseline is available.")

        if include_policy and include_native:
            no_tie_total = wins + losses
            wins_pct_no_ties = (wins / no_tie_total * 100) if no_tie_total else None
            ties_pct = (ties / comparisons_total * 100) if comparisons_total else None
            lines.append(
                "Policy vs Native record: "
                f"W={wins}, L={losses}, T={ties} "
                f"(wins/all={fmt_pct(wins_pct)}, wins/no-ties={fmt_pct(wins_pct_no_ties)}, ties={fmt_pct(ties_pct)})"
            )

        success_bits = []
        if include_native:
            success_bits.append(f"Native success: {success_rate('native_makespan')*100:.1f}%")
        if include_policy:
            success_bits.append(f"Policy success: {success_rate('policy_makespan')*100:.1f}%")
        if include_ortools:
            success_bits.append(f"OR-Tools success: {success_rate('ortools_makespan')*100:.1f}%")
        if success_bits:
            lines.append("Success rates: " + ", ".join(success_bits))

    if include_policy:
        lines.append("")
        lines.append("Note: columns labelled '(policy)' were generated using the trained policy;")
        lines.append("columns without the suffix come from the native B&B branching.")
    if include_ortools:
        lines.append("Note: columns labelled '(ortools)' were solved via OR-Tools CP-SAT (may be optimal or best-found).")

    # Print to stdout.
    for line in lines:
        print(line)

    # Persist to a text file under reports.
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / args.output_name
    out_path.write_text("\n".join(lines))
    print(f"\nSaved summary to {out_path}")



if __name__ == "__main__":
    main()
