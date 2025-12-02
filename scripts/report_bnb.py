import argparse
import sys
import time
from pathlib import Path
from typing import List, Tuple

# Ensure the src directory is importable when running from the repo root.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from rcpsp_bb_rl.bnb.core import BnBSolver  # noqa: E402
from rcpsp_bb_rl.data.dataset import list_instance_paths  # noqa: E402
from rcpsp_bb_rl.data.parsing import load_instance  # noqa: E402


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
        "--limit",
        type=int,
        default=None,
        help="Optional limit on number of instances to process.",
    )
    return parser.parse_args()


def run_instance(path: Path, max_nodes: int) -> Tuple[int | None, float]:
    instance = load_instance(path)
    solver = BnBSolver(instance)
    start = time.perf_counter()
    result = solver.solve(max_nodes=max_nodes)
    elapsed = time.perf_counter() - start
    return result.best_makespan, elapsed


def main() -> None:
    args = parse_args()
    paths: List[Path] = list_instance_paths(args.root, patterns=(args.pattern,))
    if args.limit is not None:
        paths = paths[: args.limit]

    rows = []
    for idx, path in enumerate(paths, start=1):
        best, elapsed = run_instance(path, args.max_nodes)
        rows.append((idx, path.name, best, elapsed))

    # Print a simple table similar to benchmark reports.
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

    print("Benchmark-style summary")
    print("-" * sum(col_widths) + "-" * 6)
    print(fmt_row(header, is_header=True))
    print("-" * sum(col_widths) + "-" * 6)
    for row in rows:
        makespan = row[2] if row[2] is not None else "-"
        print(fmt_row((row[0], row[1], makespan, row[3])))


if __name__ == "__main__":
    main()
