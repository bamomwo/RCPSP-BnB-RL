import argparse
import json
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
) -> Tuple[int | None, int | None, float | None]:
    """Solve with OR-Tools CP-SAT and return makespan, lower bound, and wall time (or None on failure)."""
    instance = load_instance(path)
    start = time.perf_counter()
    try:
        starts = solve_optimal_schedule(instance, time_limit_s=time_limit_s)
        makespan = max(starts[aid] + instance.activities[aid].duration for aid in starts)
    except Exception:
        return None, None, None
    elapsed = time.perf_counter() - start
    lower_bound = makespan if time_limit_s is None else None
    return makespan, lower_bound, elapsed



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
        help="(Deprecated) Path to a PPO policy checkpoint (.pt). Use --ppo-policy instead.",
    )
    parser.add_argument(
        "--ppo-policy",
        default=None,
        help="Optional path to a PPO policy checkpoint (.pt).",
    )
    parser.add_argument(
        "--bc-policy",
        default=None,
        help="Optional path to a BC policy checkpoint (.pt).",
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
        "--optimal-json",
        default=None,
        help="Optional JSON with optimal makespans keyed by instance filename.",
    )
    parser.add_argument(
        "--with-ortools",
        action="store_true",
        help="Also evaluate an OR-Tools CP-SAT solve for comparison.",
    )
    parser.add_argument(
        "--only-ortools",
        action="store_true",
        help="Skip native/PPO/BC runs and only evaluate OR-Tools.",
    )
    parser.add_argument(
        "--time-limit",
        type=float,
        default=None,
        help="Optional wall-clock time limit (seconds) applied to all solvers (native, PPO, BC, OR-Tools).",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=50,
        help="Print progress every N instances (default: 50). Set <=0 to disable.",
    )
    return parser.parse_args()


def load_optimal_makespans(path: Optional[str]) -> Optional[Dict[str, int]]:
    if path is None:
        return None
    data = json.loads(Path(path).read_text())
    instances = data.get("instances", {})
    if not isinstance(instances, dict):
        raise ValueError("Optimal JSON must contain an 'instances' mapping.")
    mapping = {}
    for name, payload in instances.items():
        if not isinstance(payload, dict) or "makespan" not in payload:
            continue
        mapping[str(name)] = int(payload["makespan"])
    return mapping


def main() -> None:
    args = parse_args()
    policy_device = torch.device(args.policy_device)
    ppo_path = args.ppo_policy or args.policy
    ppo_model = load_policy_checkpoint(ppo_path, device=policy_device) if ppo_path else None
    bc_model = load_policy_checkpoint(args.bc_policy, device=policy_device) if args.bc_policy else None
    time_limit_all = args.time_limit

    include_native = not args.only_ortools
    include_ppo = ppo_model is not None and include_native
    include_bc = bc_model is not None and include_native
    include_ortools = args.with_ortools or args.only_ortools
    include_bc_display = include_bc or (include_ortools and not include_bc)

    optimal_makespans = load_optimal_makespans(args.optimal_json)
    include_opt_gap = optimal_makespans is not None

    paths: List[Path] = list_instance_paths(args.root, patterns=(args.pattern,))
    if args.limit is not None:
        paths = paths[: args.limit]

    summary_rows: List[Dict[str, object]] = []
    start_time = time.perf_counter()
    for idx, path in enumerate(paths, start=1):
        entry: Dict[str, object] = {"instance": path.name}
        if optimal_makespans is not None:
            entry["optimal_makespan"] = optimal_makespans.get(path.name)

        if include_native:
            best_naive, lb_naive, time_naive, stats_naive = run_instance(
                path,
                args.max_nodes,
                time_limit_s=time_limit_all,
                return_bounds=True,
                return_stats=True,
            )
            entry["native_makespan"] = best_naive
            entry["native_lower_bound"] = lb_naive
            entry["native_time"] = time_naive
            if stats_naive is not None:
                entry["native_nodes_to_inc"] = stats_naive.get("nodes_to_first_incumbent")
                entry["native_pruned_after_inc"] = stats_naive.get("nodes_pruned_after_incumbent")
                entry["native_expanded_after_inc"] = stats_naive.get("nodes_expanded_after_incumbent")

        if include_ppo:
            best_policy, lb_policy, time_policy, stats_ppo = run_instance(
                path,
                args.max_nodes,
                policy_model=ppo_model,
                policy_device=policy_device,
                policy_max_resources=args.policy_max_resources,
                time_limit_s=time_limit_all,
                return_bounds=True,
                return_stats=True,
            )
            entry["ppo_makespan"] = best_policy
            entry["ppo_lower_bound"] = lb_policy
            entry["ppo_time"] = time_policy
            if stats_ppo is not None:
                entry["ppo_nodes_to_inc"] = stats_ppo.get("nodes_to_first_incumbent")
                entry["ppo_pruned_after_inc"] = stats_ppo.get("nodes_pruned_after_incumbent")
                entry["ppo_expanded_after_inc"] = stats_ppo.get("nodes_expanded_after_incumbent")

        if include_bc:
            best_bc, lb_bc, time_bc, stats_bc = run_instance(
                path,
                args.max_nodes,
                policy_model=bc_model,
                policy_device=policy_device,
                policy_max_resources=args.policy_max_resources,
                time_limit_s=time_limit_all,
                return_bounds=True,
                return_stats=True,
            )
            entry["bc_makespan"] = best_bc
            entry["bc_lower_bound"] = lb_bc
            entry["bc_time"] = time_bc
            if stats_bc is not None:
                entry["bc_nodes_to_inc"] = stats_bc.get("nodes_to_first_incumbent")
                entry["bc_pruned_after_inc"] = stats_bc.get("nodes_pruned_after_incumbent")
                entry["bc_expanded_after_inc"] = stats_bc.get("nodes_expanded_after_incumbent")

        if include_ortools:
            best_ortools, lb_ortools, time_ortools = run_ortools(
                path,
                time_limit_s=time_limit_all,
            )
            entry["ortools_makespan"] = best_ortools
            entry["ortools_lower_bound"] = lb_ortools
            entry["ortools_time"] = time_ortools
            if not include_bc:
                entry["bc_makespan"] = best_ortools
                entry["bc_lower_bound"] = lb_ortools
                entry["bc_time"] = time_ortools

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
    def compute_gap_pct(upper: Optional[int], lower: Optional[int]) -> Optional[float]:
        if upper is None or lower is None:
            return None
        denom = max(1, int(lower))
        return (float(upper) - float(lower)) / float(denom) * 100.0

    def compute_bound_gap_pct(incumbent: Optional[int], best_bound: Optional[int]) -> Optional[float]:
        if incumbent is None or best_bound is None or incumbent <= 0:
            return None
        return (float(incumbent) - float(best_bound)) / float(incumbent) * 100.0

    def compute_prune_ratio(pruned: Optional[int], expanded: Optional[int]) -> Optional[float]:
        if pruned is None or expanded is None:
            return None
        denom = pruned + expanded
        if denom <= 0:
            return None
        return float(pruned) / float(denom) * 100.0

    for entry in summary_rows:
        entry["native_gap"] = compute_gap_pct(
            entry.get("native_makespan"), entry.get("native_lower_bound")
        )
        entry["bc_gap"] = compute_gap_pct(
            entry.get("bc_makespan"), entry.get("bc_lower_bound")
        )
        entry["ppo_gap"] = compute_gap_pct(
            entry.get("ppo_makespan"), entry.get("ppo_lower_bound")
        )
        entry["native_prune_ratio"] = compute_prune_ratio(
            entry.get("native_pruned_after_inc"), entry.get("native_expanded_after_inc")
        )
        entry["bc_prune_ratio"] = compute_prune_ratio(
            entry.get("bc_pruned_after_inc"), entry.get("bc_expanded_after_inc")
        )
        entry["ppo_prune_ratio"] = compute_prune_ratio(
            entry.get("ppo_pruned_after_inc"), entry.get("ppo_expanded_after_inc")
        )
        entry["native_bound_gap"] = compute_bound_gap_pct(
            entry.get("native_makespan"), entry.get("native_lower_bound")
        )
        entry["bc_bound_gap"] = compute_bound_gap_pct(
            entry.get("bc_makespan"), entry.get("bc_lower_bound")
        )
        entry["ppo_bound_gap"] = compute_bound_gap_pct(
            entry.get("ppo_makespan"), entry.get("ppo_lower_bound")
        )
        if include_opt_gap:
            entry["native_gap_opt"] = compute_gap_pct(
                entry.get("native_makespan"), entry.get("optimal_makespan")
            )
            entry["bc_gap_opt"] = compute_gap_pct(
                entry.get("bc_makespan"), entry.get("optimal_makespan")
            )
            entry["ppo_gap_opt"] = compute_gap_pct(
                entry.get("ppo_makespan"), entry.get("optimal_makespan")
            )

    solver_labels = [("native", "Native"), ("bc", "BC"), ("ppo", "PPO")]
    groups = [
        ("Makespan/Upperbound", "makespan"),
        ("Lowerbound", "lower_bound"),
        ("Optimality Gap", "gap"),
    ]
    if include_opt_gap:
        groups.append(("Gap to Optimal", "gap_opt"))
    groups.extend(
        [
            ("Nodes to Incumbent", "nodes_to_inc"),
            ("Prune Ratio After Inc", "prune_ratio"),
            ("Bound Gap Closure", "bound_gap"),
        ]
    )
    header2 = [""] + [label for _, label in solver_labels] * len(groups)

    def fmt_int(val: Optional[int]) -> str:
        return "-" if val is None else str(int(val))

    def fmt_gap(val: Optional[float]) -> str:
        return "-" if val is None else f"{val:.1f}%"

    def fmt_ratio(val: Optional[float]) -> str:
        return "-" if val is None else f"{val:.1f}%"

    def fmt_float(val: Optional[float]) -> str:
        return "-" if val is None else f"{val:.1f}"

    data_rows: List[List[str]] = []
    for entry in summary_rows:
        row: List[str] = [str(entry["instance"])]
        for group_name, group_key in groups:
            for key, _ in solver_labels:
                if group_key in ("makespan", "lower_bound", "nodes_to_inc"):
                    row.append(fmt_int(entry.get(f"{key}_{group_key}")))
                elif group_key in ("gap", "gap_opt", "bound_gap"):
                    row.append(fmt_gap(entry.get(f"{key}_{group_key}")))
                elif group_key == "prune_ratio":
                    row.append(fmt_ratio(entry.get(f"{key}_{group_key}")))
                else:
                    row.append("-")
        data_rows.append(row)

    col_widths = []
    for i, label in enumerate(header2):
        if i == 0:
            col_widths.append(len("Instance"))
        else:
            col_widths.append(len(label))
    for row in data_rows:
        for i, val in enumerate(row):
            col_widths[i] = max(col_widths[i], len(val))

    group_size = len(solver_labels)
    separators: List[str] = []
    for i in range(len(col_widths) - 1):
        solver_col_idx = i - 1
        end_of_group = solver_col_idx >= 0 and (solver_col_idx % group_size) == (group_size - 1)
        separators.append("    " if end_of_group else "  ")

    def fmt_row(cols: List[str]) -> str:
        parts: List[str] = []
        for i, val in enumerate(cols):
            width = col_widths[i]
            if i == 0:
                parts.append(val.ljust(width))
            else:
                parts.append(val.rjust(width))
        out = parts[0] if parts else ""
        for i in range(1, len(parts)):
            out += separators[i - 1] + parts[i]
        return out

    def span_width(start: int, end: int) -> int:
        sep_width = sum(len(separators[i]) for i in range(start, end))
        return sum(col_widths[start : end + 1]) + sep_width

    header1 = "Instance".ljust(col_widths[0])
    start_idx = 1
    for group_label, _group_key in groups:
        end_idx = start_idx + len(solver_labels) - 1
        header1 += separators[start_idx - 1] + group_label.center(span_width(start_idx, end_idx))
        start_idx = end_idx + 1
    header2_line = fmt_row(header2)

    lines.append("Benchmark-style summary (Native vs BC vs PPO)")
    total_width = sum(col_widths) + sum(len(sep) for sep in separators)
    lines.append("-" * total_width)
    lines.append(header1)
    lines.append(header2_line)
    lines.append("-" * total_width)
    for row in data_rows:
        lines.append(fmt_row(row))

    # Aggregate comparisons.
    def collect_gaps(
        key: str,
        rows: List[Dict[str, object]] = summary_rows,
    ) -> List[float]:
        return [float(r[key]) for r in rows if r.get(key) is not None]

    def collect_vals(
        key: str,
        rows: List[Dict[str, object]] = summary_rows,
    ) -> List[float]:
        return [float(r[key]) for r in rows if r.get(key) is not None]

    gap_native = collect_gaps("native_gap") if include_native else []
    gap_bc = collect_gaps("bc_gap") if include_bc_display else []
    gap_ppo = collect_gaps("ppo_gap") if include_ppo else []

    wins = ties = losses = 0
    if include_ppo and include_native:
        for r in summary_rows:
            n = r.get("native_makespan")
            p = r.get("ppo_makespan")
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

    if include_ppo or include_bc_display or include_native or include_ortools:
        lines.append("")
        lines.append("Aggregates:")

        show_wins = include_ppo and include_native
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
            return stat_fn(gaps)

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
            row.append(fmt_pct(pct_within(gaps, 5.0), digits=0))
            return row

        def build_row_no_wins(label: str, gaps: List[float]) -> List[str]:
            return [
                label,
                fmt_pct(gap_stat(gaps, statistics.median)),
                fmt_pct(gap_stat(gaps, statistics.mean)),
                fmt_pct(pct_within(gaps, 5.0), digits=0),
            ]

        agg_rows: List[List[str]] = []
        if include_native:
            agg_rows.append(build_row("Native", gap_native))
        if include_bc_display:
            agg_rows.append(build_row("BC", gap_bc))
        if include_ppo:
            agg_rows.append(build_row("PPO", gap_ppo, wins_pct))

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

        lines.append("Gaps computed as (upper-lower)/max(1, lower) in percentage.")

        if include_opt_gap:
            gap_opt_native = collect_gaps("native_gap_opt") if include_native else []
            gap_opt_bc = collect_gaps("bc_gap_opt") if include_bc_display else []
            gap_opt_ppo = collect_gaps("ppo_gap_opt") if include_ppo else []

            lines.append("")
            lines.append("Aggregates (Gap to Optimal):")
            opt_header = ["Solver", "Median Gap", "Mean Gap", "% <= 5% Gap"]
            opt_rows: List[List[str]] = []
            if include_native:
                opt_rows.append(build_row_no_wins("Native", gap_opt_native))
            if include_bc_display:
                opt_rows.append(build_row_no_wins("BC", gap_opt_bc))
            if include_ppo:
                opt_rows.append(build_row_no_wins("PPO", gap_opt_ppo))

            opt_col_widths = [len(h) for h in opt_header]
            for row in opt_rows:
                for i, val in enumerate(row):
                    opt_col_widths[i] = max(opt_col_widths[i], len(val))

            def fmt_opt_row(row: List[str]) -> str:
                parts: List[str] = []
                for i, val in enumerate(row):
                    width = opt_col_widths[i]
                    if i == 0:
                        parts.append(val.ljust(width))
                    else:
                        parts.append(val.rjust(width))
                return "  ".join(parts)

            lines.append("-" * (sum(opt_col_widths) + 2 * (len(opt_col_widths) - 1)))
            lines.append(fmt_opt_row(opt_header))
            lines.append("-" * (sum(opt_col_widths) + 2 * (len(opt_col_widths) - 1)))
            for row in opt_rows:
                lines.append(fmt_opt_row(row))

            lines.append("Gap to optimal computed as (makespan-optimal)/optimal in percentage.")

            non_trivial_rows = [
                r for r in summary_rows if r.get("native_gap_opt") not in (None, 0.0)
            ]
            gap_opt_native_nt = collect_gaps("native_gap_opt", non_trivial_rows) if include_native else []
            gap_opt_bc_nt = collect_gaps("bc_gap_opt", non_trivial_rows) if include_bc_display else []
            gap_opt_ppo_nt = collect_gaps("ppo_gap_opt", non_trivial_rows) if include_ppo else []

            lines.append("")
            lines.append("Aggregates (Gap to Optimal, Non-trivial vs Native):")
            lines.append(f"Instances included: {len(non_trivial_rows)}/{len(summary_rows)}")
            lines.append("-" * (sum(opt_col_widths) + 2 * (len(opt_col_widths) - 1)))
            lines.append(fmt_opt_row(opt_header))
            lines.append("-" * (sum(opt_col_widths) + 2 * (len(opt_col_widths) - 1)))
            if include_native:
                lines.append(fmt_opt_row(build_row_no_wins("Native", gap_opt_native_nt)))
            if include_bc_display:
                lines.append(fmt_opt_row(build_row_no_wins("BC", gap_opt_bc_nt)))
            if include_ppo:
                lines.append(fmt_opt_row(build_row_no_wins("PPO", gap_opt_ppo_nt)))

        nodes_native = collect_vals("native_nodes_to_inc") if include_native else []
        nodes_bc = collect_vals("bc_nodes_to_inc") if include_bc_display else []
        nodes_ppo = collect_vals("ppo_nodes_to_inc") if include_ppo else []
        prune_native = collect_vals("native_prune_ratio") if include_native else []
        prune_bc = collect_vals("bc_prune_ratio") if include_bc_display else []
        prune_ppo = collect_vals("ppo_prune_ratio") if include_ppo else []
        bound_native = collect_vals("native_bound_gap") if include_native else []
        bound_bc = collect_vals("bc_bound_gap") if include_bc_display else []
        bound_ppo = collect_vals("ppo_bound_gap") if include_ppo else []

        lines.append("")
        lines.append("Aggregates (Search Metrics):")
        search_header = [
            "Solver",
            "Median Nodes to Inc",
            "Mean Nodes to Inc",
            "Mean Prune Ratio",
            "Median Bound Gap",
            "Mean Bound Gap",
        ]

        def build_search_row(label: str, nodes: List[float], prunes: List[float], bounds: List[float]) -> List[str]:
            return [
                label,
                fmt_int(gap_stat(nodes, statistics.median)),
                fmt_float(gap_stat(nodes, statistics.mean)),
                fmt_pct(gap_stat(prunes, statistics.mean)),
                fmt_pct(gap_stat(bounds, statistics.median)),
                fmt_pct(gap_stat(bounds, statistics.mean)),
            ]

        search_rows: List[List[str]] = []
        if include_native:
            search_rows.append(build_search_row("Native", nodes_native, prune_native, bound_native))
        if include_bc_display:
            search_rows.append(build_search_row("BC", nodes_bc, prune_bc, bound_bc))
        if include_ppo:
            search_rows.append(build_search_row("PPO", nodes_ppo, prune_ppo, bound_ppo))

        search_col_widths = [len(h) for h in search_header]
        for row in search_rows:
            for i, val in enumerate(row):
                search_col_widths[i] = max(search_col_widths[i], len(val))

        def fmt_search_row(row: List[str]) -> str:
            parts: List[str] = []
            for i, val in enumerate(row):
                width = search_col_widths[i]
                if i == 0:
                    parts.append(val.ljust(width))
                else:
                    parts.append(val.rjust(width))
            return "  ".join(parts)

        lines.append("-" * (sum(search_col_widths) + 2 * (len(search_col_widths) - 1)))
        lines.append(fmt_search_row(search_header))
        lines.append("-" * (sum(search_col_widths) + 2 * (len(search_col_widths) - 1)))
        for row in search_rows:
            lines.append(fmt_search_row(row))

        lines.append(
            "Search metrics: nodes to first incumbent (expanded), prune ratio after incumbent, "
            "bound gap closure as (incumbent-best_bound)/incumbent."
        )

        if include_ppo and include_native:
            no_tie_total = wins + losses
            wins_pct_no_ties = (wins / no_tie_total * 100) if no_tie_total else None
            ties_pct = (ties / comparisons_total * 100) if comparisons_total else None
            lines.append(
                "PPO vs Native record: "
                f"W={wins}, L={losses}, T={ties} "
                f"(wins/all={fmt_pct(wins_pct)}, wins/no-ties={fmt_pct(wins_pct_no_ties)}, ties={fmt_pct(ties_pct)})"
            )

        success_bits = []
        if include_native:
            success_bits.append(f"Native success: {success_rate('native_makespan')*100:.1f}%")
        if include_ppo:
            success_bits.append(f"PPO success: {success_rate('ppo_makespan')*100:.1f}%")
        if include_bc_display:
            success_bits.append(f"BC success: {success_rate('bc_makespan')*100:.1f}%")
        if include_ortools:
            success_bits.append(f"OR-Tools success: {success_rate('ortools_makespan')*100:.1f}%")
        if success_bits:
            lines.append("Success rates: " + ", ".join(success_bits))

    if include_ppo:
        lines.append("")
        lines.append("Note: PPO columns were generated using the trained policy;")
        lines.append("Native columns come from the native B&B branching.")
    if include_bc_display:
        if include_bc:
            lines.append("Note: BC columns were generated using the BC policy checkpoint.")
        else:
            lines.append("Note: BC columns are populated from OR-Tools because no BC policy was provided.")
    if include_ortools:
        lines.append("Note: OR-Tools columns are produced via CP-SAT (may be optimal or best-found).")
        if time_limit_all is not None:
            lines.append("Note: OR-Tools lower bounds are omitted when a time limit is set.")

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
