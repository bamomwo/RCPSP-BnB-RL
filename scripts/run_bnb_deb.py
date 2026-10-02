"""Evaluate upper-bound B&B with a certified external lower-bound floor.

This script isolates branching/search guidance from lower-bound development.
For each instance it loads a published lower bound from a merged PSPLIB-style
baseline JSON and runs one ordinary upper-bound search with

    LB(node) = max(LB_external(instance), LB_internal(node)).

The published upper bound is used only for comparison; it is never installed
as an incumbent.  If the search finds a feasible schedule whose makespan equals
the certified external lower bound, that equality proves the schedule optimal.

Example
-------
python3 scripts/run_bnb_2.py \
    --config config/run_bnb.json \
    --root data/eval/J60 \
    --baseline-json data/baseline/j60_sm_merged.json \
    --only-unproven \
    --skip-missing-bounds \
    --output-path results/policy_external_lb_j60.txt
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from rcpsp_bb_rl.bnb.branching_order import make_order_fn  # noqa: E402
from rcpsp_bb_rl.bnb.dominance import (  # noqa: E402
    format_dominance_spec,
    normalize_dominance_spec,
)
from rcpsp_bb_rl.bnb.lower_bounds import (  # noqa: E402
    DEFAULT_LOWER_BOUND_ID,
    format_lower_bound_spec,
    list_lower_bound_ids,
    lower_bound,
    normalize_lower_bound_spec,
)
from rcpsp_bb_rl.bnb.solver import BnBSolver, SolverResult  # noqa: E402
from rcpsp_bb_rl.data.dataset import list_instance_paths  # noqa: E402
from rcpsp_bb_rl.data.parsing import RCPSPInstance, load_instance  # noqa: E402


SUPPORTED_BRANCHING_ORDERS = {"activity_id", "lower_bound", "mtw", "random", "policy"}
DEFAULT_INSTANCE_PATTERNS = ("*.rcp",)


@dataclass(frozen=True)
class BaselineRecord:
    parameter: int
    instance: int
    external_lower_bound: Optional[int]
    lower_bound_source: Optional[str]
    lower_bound_author: Optional[str]
    published_upper_bound: Optional[int]
    optimal_makespan: Optional[int]
    was_proven_optimal: bool


class BaselineIndex:
    """Validated per-instance lookup for a merged benchmark baseline."""

    def __init__(self, path: Path) -> None:
        raw = _load_json(path)
        family = str(raw.get("instance_set", "")).strip().lower()
        match = re.fullmatch(r"j(30|60|90|120)", family)
        if match is None:
            raise ValueError(
                f"{path}: 'instance_set' must be one of j30, j60, j90, j120."
            )
        self.path = path
        self.family = family
        self.job_count = int(match.group(1))

        rows = raw.get("solutions")
        if not isinstance(rows, list) or not rows:
            raise ValueError(f"{path}: expected a non-empty 'solutions' array.")

        self._by_alias: Dict[str, BaselineRecord] = {}
        self.records: List[BaselineRecord] = []
        seen_keys: set[Tuple[int, int]] = set()
        for row_number, row in enumerate(rows, start=1):
            if not isinstance(row, dict):
                raise ValueError(f"{path}: solutions[{row_number - 1}] must be an object.")
            try:
                parameter = int(row["parameter"])
                instance_number = int(row["instance"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(
                    f"{path}: solutions[{row_number - 1}] has an invalid benchmark key."
                ) from exc
            key = (parameter, instance_number)
            if key in seen_keys:
                raise ValueError(f"{path}: duplicate benchmark record {key}.")
            seen_keys.add(key)

            published_ub = _optional_int(row.get("upper_bound"))
            explicit_lb = _optional_int(row.get("lower_bound"))
            optimal = _optional_int(row.get("optimal_makespan"))
            proven = bool(row.get("is_proven_optimal", False))

            if proven:
                if optimal is None:
                    raise ValueError(
                        f"{path}: proven record {key} has no optimal_makespan."
                    )
                external_lb = optimal
                source = "proven_optimal"
            else:
                external_lb = explicit_lb
                source = "published_lower_bound" if explicit_lb is not None else None

            if external_lb is not None and external_lb < 0:
                raise ValueError(f"{path}: record {key} has a negative lower bound.")
            if (
                external_lb is not None
                and published_ub is not None
                and external_lb > published_ub
            ):
                raise ValueError(
                    f"{path}: record {key} has LB={external_lb} > UB={published_ub}."
                )
            if proven and published_ub is not None and optimal != published_ub:
                raise ValueError(
                    f"{path}: proven record {key} has optimal={optimal} but UB={published_ub}."
                )

            record = BaselineRecord(
                parameter=parameter,
                instance=instance_number,
                external_lower_bound=external_lb,
                lower_bound_source=source,
                lower_bound_author=(
                    None
                    if row.get("lower_bound_author") is None
                    else str(row["lower_bound_author"])
                ),
                published_upper_bound=published_ub,
                optimal_makespan=optimal,
                was_proven_optimal=proven,
            )
            self.records.append(record)
            for alias in self._aliases(record):
                previous = self._by_alias.get(alias)
                if previous is not None and previous != record:
                    raise ValueError(
                        f"{path}: filename alias '{alias}' maps to multiple records."
                    )
                self._by_alias[alias] = record

    def _aliases(self, record: BaselineRecord) -> Iterable[str]:
        p = record.parameter
        i = record.instance
        flat = (p - 1) * 10 + i
        # Canonical forms found in this repository, plus explicit alternatives
        # that preserve the unambiguous (parameter, instance) pair.
        yield f"{self.family}{p}_{i}"
        yield f"{self.family}_{p}_{i}"
        if self.family == "j90":
            yield f"j90_{flat}"
        elif self.family == "j120":
            yield f"x{p}_{i}"

    def resolve(self, instance_path: Path) -> BaselineRecord:
        alias = instance_path.stem.strip().lower()
        record = self._by_alias.get(alias)
        if record is None:
            raise KeyError(
                f"No {self.family} baseline record matches '{instance_path.name}'. "
                "Check that the instance folder and --baseline-json describe the same set."
            )
        return record


def _optional_int(value: object) -> Optional[int]:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Expected an integer or null, got {value!r}.") from exc


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open() as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"JSON at {path} must be an object.")
    return data


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run upper-bound B&B with a per-instance external LB floor.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", required=True, help="Path to run configuration JSON.")
    parser.add_argument(
        "--baseline-json",
        required=True,
        help="Merged benchmark JSON containing solutions[].lower_bound records.",
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--instance", help="Path to one RCPSP instance.")
    source.add_argument("--root", help="Directory of RCPSP instances.")

    parser.add_argument("--max-nodes", type=int, default=None)
    parser.add_argument("--time-limit-s", type=float, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--lower-bound",
        default=None,
        help=f"Internal lower-bound id(s). Available: {', '.join(list_lower_bound_ids())}.",
    )
    parser.add_argument("--dominance", default=None)
    parser.add_argument("--output-path", default=None)
    parser.add_argument("--emit-csv", default=None)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--show-schedule", action="store_true")
    parser.add_argument(
        "--only-unproven",
        action="store_true",
        help="Run only records not already marked proven optimal in the baseline.",
    )
    parser.add_argument(
        "--skip-missing-bounds",
        action="store_true",
        help="Skip matched records with no certified external LB instead of failing.",
    )
    return parser.parse_args()


def _resolve_patterns(config: Dict[str, Any]) -> Sequence[str]:
    raw = config.get("patterns", DEFAULT_INSTANCE_PATTERNS)
    if isinstance(raw, str):
        raw = [raw]
    if not isinstance(raw, list) and not isinstance(raw, tuple):
        raise ValueError("config 'patterns' must be a string or non-empty list.")
    patterns = [str(item).strip() for item in raw if str(item).strip()]
    if not patterns:
        raise ValueError("config 'patterns' cannot be empty.")
    return patterns


def _resolve_paths(args: argparse.Namespace, config: Dict[str, Any]) -> List[Path]:
    if args.instance:
        path = Path(args.instance)
        if not path.is_file():
            raise FileNotFoundError(f"Instance not found: {path}")
        return [path]
    paths = list_instance_paths(args.root, patterns=_resolve_patterns(config))
    if not paths:
        raise FileNotFoundError(f"No instances found under {args.root}.")
    return paths


def _resolve_positive_optional(
    cli_value: Optional[float],
    config: Dict[str, Any],
    key: str,
) -> Optional[float]:
    value = cli_value if cli_value is not None else config.get(key)
    if value is None:
        return None
    result = float(value)
    if result <= 0:
        raise ValueError(f"{key} must be > 0 when provided.")
    return result


def _resolve_device(requested: str):
    import torch

    name = str(requested).strip().lower()
    if name == "cuda" and not torch.cuda.is_available():
        print("[warn] CUDA is unavailable; using CPU for policy inference.")
        name = "cpu"
    return torch.device(name)


def _effective_termination(result: SolverResult, max_nodes: Optional[int]) -> str:
    if result.done_reason == "external_bound_matched":
        return "external_bound_matched"
    pending = any(node.status == "pending" for node in result.nodes)
    if not pending:
        return "search_exhausted"
    if result.done_reason == "time_limit":
        return "time_limit"
    if max_nodes is not None and result.nodes_expanded >= max_nodes:
        return "node_limit"
    return "truncated"


def _certified_lower_bound(
    result: SolverResult,
    effective_root_lb: int,
) -> int:
    pending_lbs = [node.lower_bound for node in result.nodes if node.status == "pending"]
    if not pending_lbs:
        return (
            int(result.best_makespan)
            if result.best_makespan is not None
            else int(effective_root_lb)
        )
    frontier_lb = int(min(pending_lbs))
    if result.best_makespan is not None:
        return min(frontier_lb, int(result.best_makespan))
    return frontier_lb


def _fmt_int(value: object) -> str:
    return "-" if value is None else str(int(value))


def _fmt_gap(value: object) -> str:
    return "-" if value is None else f"{float(value):.2f}%"


def _render_table(headers: List[str], rows: List[List[str]]) -> List[str]:
    widths = [len(header) for header in headers]
    for row in rows:
        for index, cell in enumerate(row):
            widths[index] = max(widths[index], len(cell))

    def render(values: List[str]) -> str:
        return "  ".join(
            value.ljust(widths[index]) if index == 0 else value.rjust(widths[index])
            for index, value in enumerate(values)
        )

    return [render(headers), "-" * (sum(widths) + 2 * (len(widths) - 1))] + [
        render(row) for row in rows
    ]


def main() -> None:
    args = parse_args()
    config = _load_json(Path(args.config))
    strategy = str(config.get("search_strategy", "ubs")).strip().lower()
    if strategy != "ubs":
        raise ValueError(
            "run_bnb_2.py implements upper-bound search only; "
            "config search_strategy must be 'ubs'."
        )

    paths = _resolve_paths(args, config)
    baseline = BaselineIndex(Path(args.baseline_json))

    raw_limit = args.limit if args.limit is not None else config.get("limit")
    limit: Optional[int] = None
    if raw_limit is not None:
        limit = int(raw_limit)
        if limit <= 0:
            raise ValueError("limit must be > 0 when provided.")

    selected: List[Tuple[Path, BaselineRecord]] = []
    skipped_proven = 0
    skipped_missing = 0
    for path in paths:
        record = baseline.resolve(path)
        if args.only_unproven and record.was_proven_optimal:
            skipped_proven += 1
            continue
        if record.external_lower_bound is None:
            if args.skip_missing_bounds:
                skipped_missing += 1
                continue
            raise ValueError(
                f"{path.name}: matched baseline record "
                f"({record.parameter}, {record.instance}) has no certified lower bound. "
                "Use --skip-missing-bounds to omit such records."
            )
        selected.append((path, record))

    if limit is not None:
        selected = selected[:limit]
    if not selected:
        raise RuntimeError(
            "No instances were selected after applying the baseline filters "
            f"(already_proven={skipped_proven}, missing_bound={skipped_missing})."
        )

    max_nodes_raw = _resolve_positive_optional(args.max_nodes, config, "max_nodes")
    max_nodes = None if max_nodes_raw is None else int(max_nodes_raw)
    time_limit_s = _resolve_positive_optional(args.time_limit_s, config, "time_limit_s")

    raw_lb = (
        args.lower_bound
        if args.lower_bound is not None
        else config.get("lower_bound", DEFAULT_LOWER_BOUND_ID)
    )
    lb_spec = normalize_lower_bound_spec(raw_lb)
    dominance = normalize_dominance_spec(
        args.dominance if args.dominance is not None else config.get("dominance", False)
    )

    branch_order = str(config.get("branching_order", "activity_id")).strip().lower()
    if branch_order == "classical":
        branch_order = "activity_id"
    if branch_order not in SUPPORTED_BRANCHING_ORDERS:
        raise ValueError(
            f"branching_order must be one of {sorted(SUPPORTED_BRANCHING_ORDERS)}."
        )

    policy_model = None
    policy_device = None
    policy_max_resources = int(config.get("policy_max_resources", 4))
    if branch_order == "policy":
        policy_path = config.get("policy_path")
        if not policy_path:
            raise ValueError("branching_order=policy requires config 'policy_path'.")
        import torch
        from rcpsp_bb_rl.ml.models import load_policy_checkpoint

        requested_device = str(config.get("policy_device", "cpu"))
        if requested_device.strip().lower() == "cpu":
            torch.set_num_threads(1)
        policy_device = _resolve_device(requested_device)
        policy_model = load_policy_checkpoint(str(policy_path), device=policy_device)

    rows: List[Dict[str, object]] = []
    progress_every = max(0, int(config.get("progress_every", 1)))
    random_seed = int(config.get("random_seed", 0))

    for position, (path, record) in enumerate(selected, start=1):
        external_lb = record.external_lower_bound
        assert external_lb is not None  # selected records are checked above

        instance: RCPSPInstance = load_instance(path)
        expected_activities = baseline.job_count + 2
        if instance.num_activities != expected_activities:
            raise ValueError(
                f"{path.name}: contains {instance.num_activities} activities, but "
                f"{baseline.family} expects {expected_activities} including dummies."
            )

        internal_root_lb = int(
            lower_bound(instance, set(instance.activities), {}, lb_id=lb_spec)
        )
        effective_root_lb = max(internal_root_lb, external_lb)

        solver = BnBSolver(instance)
        order_fn = None
        if branch_order == "policy":
            order_fn = make_order_fn(
                "policy",
                instance=instance,
                model=policy_model,
                max_resources=policy_max_resources,
                device=policy_device,
                predecessors=solver.predecessors,
            )
        elif branch_order == "lower_bound":
            order_fn = make_order_fn(
                "lower_bound",
                instance=instance,
                predecessors=solver.predecessors,
                lb_id=lb_spec,
            )
        elif branch_order == "mtw":
            order_fn = make_order_fn(
                "mtw",
                instance=instance,
            )
        elif branch_order == "random":
            order_fn = make_order_fn("random", seed=random_seed)

        started = time.perf_counter()
        result = solver.solve(
            max_nodes=max_nodes,
            order_ready_fn=order_fn,
            time_limit_s=time_limit_s,
            lb_spec=lb_spec,
            dominance=dominance,
            debug=args.debug,
            external_lower_bound=external_lb,
        )
        elapsed = time.perf_counter() - started

        best = result.best_makespan
        if best is not None and best < external_lb:
            raise RuntimeError(
                f"{path.name}: feasible makespan {best} contradicts certified LB {external_lb}."
            )
        certified_lb = _certified_lower_bound(result, effective_root_lb)
        matched_external = best is not None and best == external_lb
        proven_optimal = best is not None and certified_lb == best
        newly_proven = proven_optimal and not record.was_proven_optimal
        improved_ub = (
            best is not None
            and record.published_upper_bound is not None
            and best < record.published_upper_bound
        )
        gap_pct = (
            None
            if best is None or best <= 0
            else max(0.0, (best - certified_lb) / best * 100.0)
        )
        termination = _effective_termination(result, max_nodes)
        if newly_proven:
            outcome = "NEW_OPTIMUM"
        elif proven_optimal:
            outcome = "OPTIMAL"
        elif improved_ub:
            outcome = "IMPROVED_UB"
        elif best is not None:
            outcome = "FEASIBLE"
        else:
            outcome = "NO_SOLUTION"

        rows.append(
            {
                "instance": path.name,
                "parameter": record.parameter,
                "instance_number": record.instance,
                "external_lb": external_lb,
                "external_lb_source": record.lower_bound_source,
                "external_lb_author": record.lower_bound_author,
                "internal_root_lb": internal_root_lb,
                "effective_root_lb": effective_root_lb,
                "published_ub": record.published_upper_bound,
                "previously_proven": record.was_proven_optimal,
                "best_makespan": best,
                "certified_lb": certified_lb,
                "gap_pct": gap_pct,
                "matched_external_lb": matched_external,
                "improved_published_ub": improved_ub,
                "proven_optimal": proven_optimal,
                "newly_proven": newly_proven,
                "first_incumbent_node": result.first_incumbent_expanded,
                "nodes": result.nodes_expanded,
                "cpu_time_s": elapsed,
                "termination": termination,
                "outcome": outcome,
                "schedule": result.best_schedule,
            }
        )

        if args.debug:
            from run_bnb import print_debug_report

            print_debug_report(path.name, result)
        if progress_every and (position % progress_every == 0 or position == len(selected)):
            print(f"[{position}/{len(selected)}] processed {path.name}", file=sys.stderr)

    headers = [
        "Instance", "ExtLB", "IntLB", "EffLB", "PriorUB", "Best", "CertLB",
        "Gap", "Nodes", "CPU[s]", "Termination", "Outcome",
    ]
    table_rows = [
        [
            str(row["instance"]),
            _fmt_int(row["external_lb"]),
            _fmt_int(row["internal_root_lb"]),
            _fmt_int(row["effective_root_lb"]),
            _fmt_int(row["published_ub"]),
            _fmt_int(row["best_makespan"]),
            _fmt_int(row["certified_lb"]),
            _fmt_gap(row["gap_pct"]),
            str(int(row["nodes"])),
            f"{float(row['cpu_time_s']):.3f}",
            str(row["termination"]),
            str(row["outcome"]),
        ]
        for row in rows
    ]

    rendered: List[str] = []

    def emit(line: str = "") -> None:
        rendered.append(line)
        print(line)

    emit()
    for line in _render_table(headers, table_rows):
        emit(line)

    emit()
    emit("External-bound evaluation summary")
    emit(f"Instances evaluated       : {len(rows)}")
    emit(f"Skipped already proven    : {skipped_proven}")
    emit(f"Skipped missing bounds     : {skipped_missing}")
    emit(f"Matched external LB        : {sum(bool(r['matched_external_lb']) for r in rows)}")
    emit(f"Improved published UB      : {sum(bool(r['improved_published_ub']) for r in rows)}")
    emit(f"Newly proven optimal       : {sum(bool(r['newly_proven']) for r in rows)}")
    emit(f"Any optimality proof       : {sum(bool(r['proven_optimal']) for r in rows)}")
    gaps = [float(r["gap_pct"]) for r in rows if r["gap_pct"] is not None]
    if gaps:
        emit(f"Mean final certified gap   : {statistics.mean(gaps):.2f}%")
    emit(f"Total expanded nodes       : {sum(int(r['nodes']) for r in rows)}")
    emit(f"Total CPU time [s]         : {sum(float(r['cpu_time_s']) for r in rows):.3f}")

    emit()
    emit("Configuration")
    emit(f"  baseline JSON       : {baseline.path}")
    emit(f"  benchmark set       : {baseline.family}")
    emit(f"  branching order     : {branch_order}")
    if branch_order == "random":
        emit(f"  random seed         : {random_seed}")
    emit("  search strategy     : upper-bound search")
    emit(f"  internal lower bound: {format_lower_bound_spec(lb_spec)}")
    emit("  effective node bound: max(external LB, internal node LB)")
    emit(f"  dominance           : {format_dominance_spec(dominance)}")

    if args.show_schedule:
        emit()
        emit("Schedules")
        for row in rows:
            schedule = row["schedule"]
            if schedule is None:
                emit(f"  {row['instance']}: no feasible schedule")
                continue
            ordered = sorted(
                schedule.items(), key=lambda item: (item[1].start, int(item[0]))
            )
            emit(f"  {row['instance']}: {[int(activity) for activity, _ in ordered]}")

    output_path = args.output_path or config.get("output_path")
    if output_path:
        destination = Path(str(output_path))
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text("\n".join(rendered) + "\n")
        print(f"Saved results to {destination}")

    csv_path_raw = args.emit_csv or config.get("emit_csv")
    if csv_path_raw:
        csv_path = Path(str(csv_path_raw))
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = [key for key in rows[0] if key != "schedule"]
        with csv_path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow({key: row[key] for key in fieldnames})
        print(f"Saved CSV results to {csv_path}")


if __name__ == "__main__":
    main()
