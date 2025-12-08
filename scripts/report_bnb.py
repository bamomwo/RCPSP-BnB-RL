import argparse
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

import torch

# Ensure the src directory is importable when running from the repo root.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from rcpsp_bb_rl.bnb.core import BnBSolver  # noqa: E402
from rcpsp_bb_rl.bnb.teacher import solve_optimal_schedule  # noqa: E402
from rcpsp_bb_rl.bnb.policy_guidance import make_policy_order_fn  # noqa: E402
from rcpsp_bb_rl.data.dataset import list_instance_paths  # noqa: E402
from rcpsp_bb_rl.data.parsing import load_instance  # noqa: E402
from rcpsp_bb_rl.models import load_policy_checkpoint  # noqa: E402


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
        "--ortools-time-limit",
        type=float,
        default=None,
        help="Optional time limit (seconds) for the OR-Tools solve.",
    )
    return parser.parse_args()


def run_instance(
    path: Path,
    max_nodes: int,
    policy_model=None,
    policy_device: Optional[torch.device] = None,
    policy_max_resources: int = 4,
) -> Tuple[int | None, float]:
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
    result = solver.solve(max_nodes=max_nodes, order_ready_fn=order_fn)
    elapsed = time.perf_counter() - start
    return result.best_makespan, elapsed


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


def main() -> None:
    args = parse_args()
    policy_model = None
    policy_device = torch.device(args.policy_device)
    if args.policy:
        policy_model = load_policy_checkpoint(args.policy, device=policy_device)

    include_native = not args.only_ortools
    include_policy = policy_model is not None and include_native
    include_ortools = args.with_ortools or args.only_ortools

    paths: List[Path] = list_instance_paths(args.root, patterns=(args.pattern,))
    if args.limit is not None:
        paths = paths[: args.limit]

    rows = []
    for idx, path in enumerate(paths, start=1):
        row_vals = [idx, path.name]

        if include_native:
            best_naive, time_naive = run_instance(path, args.max_nodes)
            row_vals.extend([best_naive, time_naive])

        if include_policy:
            best_policy, time_policy = run_instance(
                path,
                args.max_nodes,
                policy_model=policy_model,
                policy_device=policy_device,
                policy_max_resources=args.policy_max_resources,
            )
            row_vals.extend([best_policy, time_policy])

        if include_ortools:
            best_ortools, time_ortools = run_ortools(
                path,
                time_limit_s=args.ortools_time_limit,
            )
            row_vals.extend([best_ortools, time_ortools])

        rows.append(row_vals)

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
