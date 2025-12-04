import argparse
import sys
from pathlib import Path

# Ensure the src directory is importable when running from the repo root.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from rcpsp_bb_rl.bnb.teacher import generate_trace, solve_optimal_schedule, write_trace  # noqa: E402
from rcpsp_bb_rl.data.dataset import list_instance_paths  # noqa: E402
from rcpsp_bb_rl.data.parsing import load_instance  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Log teacher trajectories using OR-Tools optimal schedules.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--instance",
        help="Path to a single RCPSP instance file (custom .RCP format).",
    )
    group.add_argument(
        "--root",
        help="Directory containing RCPSP instances; will recurse with --pattern.",
    )
    parser.add_argument(
        "--pattern",
        default="*.RCP",
        help="Glob pattern when using --root (default: *.RCP).",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory to write JSONL trajectories. Default: reports/trajectories/",
    )
    parser.add_argument(
        "--time-limit",
        type=float,
        default=None,
        help="Optional CP-SAT time limit in seconds.",
    )
    return parser.parse_args()


def process_instance(inst_path: Path, output_dir: Path, time_limit: float | None) -> Path:
    instance = load_instance(inst_path)
    start_times = solve_optimal_schedule(instance, time_limit_s=time_limit)
    records = generate_trace(instance, start_times, instance_name=inst_path.name)
    out_path = output_dir / f"{inst_path.stem}.jsonl"
    write_trace(records, out_path)
    return out_path


def main() -> None:
    args = parse_args()

    out_dir = (
        Path(args.output_dir)
        if args.output_dir
        else PROJECT_ROOT / "data" / "trajectories"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.instance:
        inst_paths = [Path(args.instance)]
    else:
        inst_paths = list_instance_paths(args.root, patterns=(args.pattern,))
        if not inst_paths:
            raise FileNotFoundError(f"No instances found under {args.root} with pattern {args.pattern}")

    for idx, inst_path in enumerate(inst_paths, 1):
        out_path = process_instance(inst_path, out_dir, args.time_limit)
        print(f"[{idx}/{len(inst_paths)}] Wrote {inst_path.name} -> {out_path}")


if __name__ == "__main__":
    main()
