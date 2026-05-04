"""
Generate optimal teacher trajectories for RCPSP instances.

Each instance is solved to optimality with OR-Tools CP-SAT, then replayed
to produce a JSONL file of (state, action) records for imitation learning.

Features:
  - Parallel workers (--workers N)
  - Skip instances whose output file already exists (--skip-existing)
  - Optional per-instance CP-SAT time limit (--time-limit)
  - Progress reporting with success / failure / skipped counts
"""
from __future__ import annotations

import argparse
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from rcpsp_bb_rl.data.dataset import list_instance_paths  # noqa: E402
from rcpsp_bb_rl.data.parsing import load_instance  # noqa: E402
from rcpsp_bb_rl.ml.il.teacher import generate_trace, solve_optimal_schedule, write_trace  # noqa: E402


# ---------------------------------------------------------------------------
# Worker function (runs in a subprocess)
# ---------------------------------------------------------------------------

def _process_one(
    inst_path: Path,
    output_dir: Path,
    time_limit: Optional[float],
) -> Tuple[str, str]:
    """
    Solve and trace one instance.

    Returns (status, message) where status is "ok", "skip", or "fail".
    """
    out_path = output_dir / f"{inst_path.stem}.jsonl"
    if out_path.exists():
        return "skip", str(inst_path.name)

    try:
        instance = load_instance(inst_path)
        start_times = solve_optimal_schedule(instance, time_limit_s=time_limit)
        records = generate_trace(instance, start_times, instance_name=inst_path.name)
        write_trace(records, out_path)
        return "ok", str(inst_path.name)
    except Exception as exc:
        return "fail", f"{inst_path.name}: {exc}\n{traceback.format_exc()}"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate optimal teacher trajectories for RCPSP instances.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--instance",
        help="Path to a single RCPSP instance file.",
    )
    source.add_argument(
        "--root",
        help="Directory containing RCPSP instances (searched recursively).",
    )

    parser.add_argument(
        "--pattern",
        default="*.RCP",
        help="Glob pattern when using --root.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory to write JSONL trajectories. Default: data/trajectories/",
    )
    parser.add_argument(
        "--time-limit",
        type=float,
        default=None,
        help="CP-SAT time limit per instance in seconds (None = no limit).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of parallel worker processes.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        default=True,
        help="Skip instances whose output JSONL already exists (default: on).",
    )
    parser.add_argument(
        "--no-skip-existing",
        dest="skip_existing",
        action="store_false",
        help="Re-generate trajectories even if output already exists.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process at most this many instances (useful for dry runs).",
    )
    return parser.parse_args()


def _collect_paths(args: argparse.Namespace) -> List[Path]:
    if args.instance:
        return [Path(args.instance)]
    paths = list_instance_paths(args.root, patterns=(args.pattern,))
    if not paths:
        raise FileNotFoundError(
            f"No instances found under {args.root} with pattern {args.pattern}"
        )
    return paths


def main() -> None:
    args = parse_args()

    out_dir = (
        Path(args.output_dir)
        if args.output_dir
        else PROJECT_ROOT / "data" / "trajectories"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    inst_paths = _collect_paths(args)
    if args.limit is not None:
        inst_paths = inst_paths[: args.limit]

    # Pre-filter skips in the main process to avoid spawning workers needlessly.
    if args.skip_existing:
        pending = [p for p in inst_paths if not (out_dir / f"{p.stem}.jsonl").exists()]
        n_skipped_upfront = len(inst_paths) - len(pending)
    else:
        pending = inst_paths
        n_skipped_upfront = 0

    total = len(inst_paths)
    n_ok = 0
    n_fail = 0
    n_skip = n_skipped_upfront

    print(
        f"Instances: {total} total | {len(pending)} to process | "
        f"{n_skipped_upfront} already done | workers={args.workers}"
    )

    if not pending:
        print("Nothing to do.")
        return

    if args.workers <= 1:
        # Single-process path — easier to debug.
        for idx, inst_path in enumerate(pending, 1):
            status, msg = _process_one(inst_path, out_dir, args.time_limit)
            if status == "ok":
                n_ok += 1
                print(f"[{idx}/{len(pending)}] ok  {msg}")
            elif status == "skip":
                n_skip += 1
            else:
                n_fail += 1
                print(f"[{idx}/{len(pending)}] FAIL  {msg}")
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = {
                pool.submit(_process_one, p, out_dir, args.time_limit): p
                for p in pending
            }
            completed = 0
            for future in as_completed(futures):
                completed += 1
                status, msg = future.result()
                if status == "ok":
                    n_ok += 1
                elif status == "skip":
                    n_skip += 1
                else:
                    n_fail += 1
                    print(f"FAIL  {msg}")

                if completed % 100 == 0 or completed == len(pending):
                    print(
                        f"  progress: {completed}/{len(pending)} | "
                        f"ok={n_ok} skip={n_skip} fail={n_fail}"
                    )

    print(
        f"\nDone. ok={n_ok} | skipped={n_skip} | failed={n_fail} | "
        f"output_dir={out_dir}"
    )


if __name__ == "__main__":
    main()
