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


def main() -> None:
    args = parse_args()
    policy_model = None
    policy_device = torch.device(args.policy_device)
    if args.policy:
        policy_model = load_policy_checkpoint(args.policy, device=policy_device)

    paths: List[Path] = list_instance_paths(args.root, patterns=(args.pattern,))
    if args.limit is not None:
        paths = paths[: args.limit]

    rows = []
    for idx, path in enumerate(paths, start=1):
        best_naive, time_naive = run_instance(path, args.max_nodes)
        best_policy, time_policy = (None, None)

        if policy_model is not None:
            best_policy, time_policy = run_instance(
                path,
                args.max_nodes,
                policy_model=policy_model,
                policy_device=policy_device,
                policy_max_resources=args.policy_max_resources,
            )

        rows.append((idx, path.name, best_naive, time_naive, best_policy, time_policy))

    # Print a simple table similar to benchmark reports.
    lines = []

    if policy_model is None:
        header = ("#", "Instance", "Makespan", "CPU-Time[s]")
        col_widths = [len(h) for h in header]
        for row in rows:
            col_widths[0] = max(col_widths[0], len(str(row[0])))
            col_widths[1] = max(col_widths[1], len(row[1]))
            col_widths[2] = max(col_widths[2], len(str(row[2])) if row[2] is not None else 1)
            col_widths[3] = max(col_widths[3], len(f"{row[3]:.2f}"))

        def fmt_row(cols, is_header: bool = False) -> str:
            time_part = (
                str(cols[3]).ljust(col_widths[3])
                if is_header or isinstance(cols[3], str)
                else f"{cols[3]:>{col_widths[3]}.2f}"
            )
            return (
                f"{str(cols[0]).rjust(col_widths[0])}  "
                f"{str(cols[1]).ljust(col_widths[1])}  "
                f"{str(cols[2]).rjust(col_widths[2])}  "
                f"{time_part}"
            )

        lines.append("Benchmark-style summary (native B&B)")
        lines.append("-" * sum(col_widths) + "-" * 6)
        lines.append(fmt_row(header, is_header=True))
        lines.append("-" * sum(col_widths) + "-" * 6)
        for row in rows:
            makespan = row[2] if row[2] is not None else "-"
            lines.append(fmt_row((row[0], row[1], makespan, row[3])))
    else:
        header = (
            "#",
            "Instance",
            "Makespan(native)",
            "CPU-Time[native]",
            "Makespan(policy)",
            "CPU-Time[policy]",
        )
        col_widths = [len(h) for h in header]
        for row in rows:
            col_widths[0] = max(col_widths[0], len(str(row[0])))
            col_widths[1] = max(col_widths[1], len(row[1]))
            col_widths[2] = max(col_widths[2], len(str(row[2])) if row[2] is not None else 1)
            col_widths[3] = max(col_widths[3], len(f"{row[3]:.2f}"))
            col_widths[4] = max(col_widths[4], len(str(row[4])) if row[4] is not None else 1)
            col_widths[5] = max(col_widths[5], len(f"{row[5]:.2f}")) if row[5] is not None else col_widths[5]

        def fmt_row(cols, is_header: bool = False) -> str:
            def fmt_time(val, width):
                if is_header or isinstance(val, str):
                    return str(val).ljust(width)
                if val is None:
                    return "-".ljust(width)
                return f"{val:>{width}.2f}"

            return (
                f"{str(cols[0]).rjust(col_widths[0])}  "
                f"{str(cols[1]).ljust(col_widths[1])}  "
                f"{str(cols[2] if cols[2] is not None else '-').rjust(col_widths[2])}  "
                f"{fmt_time(cols[3], col_widths[3])}  "
                f"{str(cols[4] if cols[4] is not None else '-').rjust(col_widths[4])}  "
                f"{fmt_time(cols[5], col_widths[5])}"
            )

        lines.append("Benchmark-style summary (native vs policy-guided)")
        lines.append("-" * sum(col_widths) + "-" * 10)
        lines.append(fmt_row(header, is_header=True))
        lines.append("-" * sum(col_widths) + "-" * 10)
        for row in rows:
            lines.append(fmt_row(row))

        lines.append("")
        lines.append("Note: columns labelled '(policy)' were generated using the trained policy;")
        lines.append("columns without the suffix come from the native B&B branching.")

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
