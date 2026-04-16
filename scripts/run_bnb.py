import argparse
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

# Ensure the src directory is importable when running from the repo root.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from rcpsp_bb_rl.bnb.branching_order import make_order_fn  # noqa: E402
from rcpsp_bb_rl.bnb.lower_bounds import (  # noqa: E402
    DEFAULT_LOWER_BOUND_ID,
    format_lower_bound_spec,
    list_lower_bound_ids,
    normalize_lower_bound_spec,
)
from rcpsp_bb_rl.bnb.dominance import format_dominance_spec, normalize_dominance_spec  # noqa: E402
from rcpsp_bb_rl.bnb.solver import BnBSolver, SolverResult  # noqa: E402
from rcpsp_bb_rl.bnb.search_strategy import build_search_strategy  # noqa: E402
from rcpsp_bb_rl.data.dataset import list_instance_paths  # noqa: E402
from rcpsp_bb_rl.data.parsing import load_instance  # noqa: E402

REQUIRED_CONFIG_KEYS = {
    "instance_patterns_config",
}

SUPPORTED_BRANCHING_ORDERS = {"activity_id", "lower_bound", "policy"}

OPTIONAL_CONFIG_KEYS = {
    "max_nodes",
    "time_limit_s",
    "lower_bound",
    "branching_order",
    "policy_path",
    "policy_device",
    "policy_max_resources",
    "dominance",
    "progress_every",
    "limit",
    "search_strategy",
    "lbs_initial_lower_bound",
    "output_path",
    "optimal_json",
}

ALLOWED_CONFIG_KEYS = REQUIRED_CONFIG_KEYS | OPTIONAL_CONFIG_KEYS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run branch-and-bound (single instance or directory batch)."
    )
    parser.add_argument("--config", required=True, help="Path to run config JSON.")
    parser.add_argument(
        "--policy",
        default=None,
        help="Policy checkpoint override. Providing this forces branching_order=policy for this run.",
    )

    src_group = parser.add_mutually_exclusive_group(required=True)
    src_group.add_argument("--instance", help="Path to one RCPSP instance file.")
    src_group.add_argument("--root", help="Directory containing RCPSP instances.")

    # Optional explicit overrides.
    parser.add_argument(
        "--max-nodes",
        type=int,
        default=None,
        help="Max nodes override. Omit to run until full search completion.",
    )
    parser.add_argument("--time-limit-s", type=float, default=None, help="Time limit override.")
    parser.add_argument("--limit", type=int, default=None, help="Instance limit override for --root runs.")
    parser.add_argument(
        "--lower-bound",
        default=None,
        help=f"Lower bound id override. Available: {', '.join(list_lower_bound_ids())}.",
    )
    parser.add_argument(
        "--dominance",
        default=None,
        help=(
            "Dominance override. Accepted: off|on|all|"
            "set_based,contradiction,extended_global_shift"
        ),
    )
    parser.add_argument(
        "--search-strategy",
        default=None,
        help="Search strategy override. Supported: ubs|lbs.",
    )
    parser.add_argument(
        "--lbs-initial-lower-bound",
        type=int,
        default=None,
        help="Optional fixed initial lower bound for lbs strategy.",
    )
    parser.add_argument(
        "--output-path",
        default=None,
        help="Optional file path to save the displayed results table and notes.",
    )
    parser.add_argument(
        "--optimal-json",
        default=None,
        help=(
            "Optional baseline JSON path with optimal makespans "
            "(e.g. data/baseline/j30_opt.json) for gap-to-optimal reporting."
        ),
    )
    return parser.parse_args()


def load_json(path: Path) -> Dict[str, Any]:
    with path.open() as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"JSON at {path} must be an object.")
    return data


def load_run_config(path: Path) -> Dict[str, Any]:
    cfg = load_json(path)
    unknown = set(cfg) - ALLOWED_CONFIG_KEYS
    if unknown:
        names = ", ".join(sorted(unknown))
        raise ValueError(f"Unknown config key(s) in {path}: {names}")

    missing = [key for key in sorted(REQUIRED_CONFIG_KEYS) if key not in cfg]
    if missing:
        names = ", ".join(missing)
        raise ValueError(f"Missing required config key(s) in {path}: {names}")

    return cfg


def load_patterns(path: Path) -> List[str]:
    payload = load_json(path)
    unknown = set(payload) - {"patterns"}
    if unknown:
        names = ", ".join(sorted(unknown))
        raise ValueError(f"Unknown key(s) in patterns config {path}: {names}")
    patterns = payload.get("patterns")
    if not isinstance(patterns, list) or not patterns:
        raise ValueError(f"'patterns' in {path} must be a non-empty list of glob strings.")
    clean = [str(p).strip() for p in patterns if str(p).strip()]
    if not clean:
        raise ValueError(f"'patterns' in {path} cannot be empty after stripping.")
    return clean


def resolve_paths(args: argparse.Namespace, cfg: Dict[str, Any]) -> List[Path]:
    if args.instance:
        return [Path(args.instance)]

    patterns_cfg = Path(str(cfg["instance_patterns_config"]))
    patterns = load_patterns(patterns_cfg)
    paths = list_instance_paths(args.root, patterns=tuple(patterns))
    limit = args.limit if args.limit is not None else cfg.get("limit")
    if limit is not None and int(limit) <= 0:
        raise ValueError("limit must be > 0 when provided.")
    if limit is not None:
        paths = paths[: int(limit)]
    if not paths:
        raise FileNotFoundError(
            f"No instances found under {args.root} with patterns from {patterns_cfg}: {patterns}"
        )
    return paths


def resolve_policy_device(requested: str):
    import torch

    req = str(requested).strip().lower()
    if req == "cuda" and not torch.cuda.is_available():
        print("[warn] Requested policy_device=cuda but CUDA is unavailable; falling back to cpu.")
        return torch.device("cpu")
    return torch.device(req)


def normalize_instance_key(name: str) -> str:
    return str(name).strip().lower()


def load_optimal_makespans(path: Optional[str]) -> Optional[Dict[str, int]]:
    if path is None:
        return None
    data = json.loads(Path(path).read_text())
    instances = data.get("instances", {})
    if not isinstance(instances, dict):
        raise ValueError("Optimal JSON must contain an 'instances' mapping.")
    mapping: Dict[str, int] = {}
    for name, payload in instances.items():
        if not isinstance(payload, dict) or "makespan" not in payload:
            continue
        mapping[normalize_instance_key(str(name))] = int(payload["makespan"])
    return mapping


def validate_and_resolve(
    args: argparse.Namespace,
    cfg: Dict[str, Any],
) -> tuple[
    Optional[int],
    Optional[float],
    str,
    Optional[str],
    str,
    int,
    List[str],
    object,
    str,
    Optional[int],
    Optional[str],
    Optional[str],
]:
    raw_max_nodes = args.max_nodes if args.max_nodes is not None else cfg.get("max_nodes")
    max_nodes: Optional[int]
    if raw_max_nodes is None:
        max_nodes = None
    else:
        max_nodes = int(raw_max_nodes)
        if max_nodes <= 0:
            raise ValueError("max_nodes must be > 0 when provided.")

    raw_time_limit = args.time_limit_s if args.time_limit_s is not None else cfg.get("time_limit_s")
    time_limit_s = None if raw_time_limit is None else float(raw_time_limit)
    if time_limit_s is not None and time_limit_s <= 0:
        raise ValueError("time_limit_s must be > 0 when provided.")

    branch_order_raw = str(cfg.get("branching_order", "activity_id")).strip().lower()
    branch_order = "activity_id" if branch_order_raw == "classical" else branch_order_raw
    if branch_order not in SUPPORTED_BRANCHING_ORDERS:
        raise ValueError(
            "branching_order must be one of: "
            + ", ".join(sorted(SUPPORTED_BRANCHING_ORDERS))
            + " (or 'classical' as alias for activity_id)."
        )

    policy_path: Optional[str]
    if args.policy is not None:
        branch_order = "policy"
        policy_path = str(args.policy)
    else:
        policy_path = cfg.get("policy_path")
        policy_path = None if policy_path is None else str(policy_path)

    policy_device = str(cfg.get("policy_device", "cpu"))
    policy_max_resources = int(cfg.get("policy_max_resources", 4))
    if policy_max_resources <= 0:
        raise ValueError("policy_max_resources must be > 0.")
    if branch_order == "policy" and not policy_path:
        raise ValueError(
            "branching_order=policy requires a policy checkpoint via config 'policy_path' or --policy."
        )

    raw_lb_spec: object
    if args.lower_bound is not None:
        raw_lb_spec = args.lower_bound
    elif "lower_bound" in cfg:
        raw_lb_spec = cfg.get("lower_bound")
    else:
        raw_lb_spec = DEFAULT_LOWER_BOUND_ID

    lb_spec = normalize_lower_bound_spec(raw_lb_spec)
    raw_dom_spec: object
    if args.dominance is not None:
        raw_dom_spec = args.dominance
    elif "dominance" in cfg:
        raw_dom_spec = cfg.get("dominance")
    else:
        raw_dom_spec = False
    dominance_spec = normalize_dominance_spec(raw_dom_spec)

    raw_search_strategy = (
        args.search_strategy
        if args.search_strategy is not None
        else cfg.get("search_strategy", "ubs")
    )
    search_strategy = str(raw_search_strategy).strip().lower()

    raw_lbs_initial_lower_bound = (
        args.lbs_initial_lower_bound
        if args.lbs_initial_lower_bound is not None
        else cfg.get("lbs_initial_lower_bound")
    )
    lbs_initial_lower_bound = (
        None
        if raw_lbs_initial_lower_bound is None
        else int(raw_lbs_initial_lower_bound)
    )
    if lbs_initial_lower_bound is not None and lbs_initial_lower_bound < 0:
        raise ValueError("lbs_initial_lower_bound must be >= 0 when provided.")
    # Validate strategy early so config/CLI errors fail before running instances.
    build_search_strategy(
        search_strategy,
        initial_lower_bound=lbs_initial_lower_bound,
    )

    output_path = (
        str(args.output_path)
        if args.output_path is not None
        else (
            None
            if cfg.get("output_path") is None
            else str(cfg.get("output_path"))
        )
    )
    optimal_json = (
        str(args.optimal_json)
        if args.optimal_json is not None
        else (
            None
            if cfg.get("optimal_json") is None
            else str(cfg.get("optimal_json"))
        )
    )

    return (
        max_nodes,
        time_limit_s,
        branch_order,
        (None if policy_path is None else str(policy_path)),
        policy_device,
        policy_max_resources,
        lb_spec,
        dominance_spec,
        search_strategy,
        lbs_initial_lower_bound,
        output_path,
        optimal_json,
    )


def compute_global_lower_bound(result: SolverResult) -> Optional[int]:
    pending_lbs = [node.lower_bound for node in result.nodes if node.status == "pending"]
    if pending_lbs:
        return int(min(pending_lbs))
    if result.best_makespan is not None:
        return int(result.best_makespan)
    return None


def main() -> None:
    args = parse_args()
    cfg = load_run_config(Path(args.config))
    (
        max_nodes,
        time_limit_s,
        branch_order,
        policy_path,
        policy_device,
        policy_max_resources,
        lb_spec,
        dominance_spec,
        search_strategy,
        lbs_initial_lower_bound,
        output_path,
        optimal_json,
    ) = validate_and_resolve(args, cfg)
    paths = resolve_paths(args, cfg)
    optimal_by_name = load_optimal_makespans(optimal_json)

    progress_every = int(cfg.get("progress_every", 1))
    if progress_every <= 0:
        progress_every = 0

    policy_model = None
    device = None
    use_policy = branch_order == "policy"
    if use_policy:
        from rcpsp_bb_rl.models import load_policy_checkpoint

        device = resolve_policy_device(policy_device)
        policy_model = load_policy_checkpoint(policy_path, device=device)

    rows: List[Dict[str, object]] = []
    for idx, path in enumerate(paths, start=1):
        instance = load_instance(path)
        order_context_solver = BnBSolver(instance)
        order_fn = None
        if use_policy:
            order_fn = make_order_fn(
                "policy",
                instance=instance,
                model=policy_model,
                max_resources=policy_max_resources,
                device=device,
                predecessors=order_context_solver.predecessors,
            )
        elif branch_order == "lower_bound":
            order_fn = make_order_fn(
                "lower_bound",
                instance=instance,
                predecessors=order_context_solver.predecessors,
                lb_id=lb_spec,
            )

        def solver_pass(
            *,
            instance,
            max_nodes: Optional[int] = None,
            order_ready_fn=None,
            time_limit_s: Optional[float] = None,
            lb_spec: object = DEFAULT_LOWER_BOUND_ID,
            dominance: object = False,
            target_makespan: Optional[int] = None,
            stop_on_first_solution: bool = False,
        ) -> SolverResult:
            pass_solver = BnBSolver(instance)
            return pass_solver.solve(
                max_nodes=max_nodes,
                order_ready_fn=order_ready_fn,
                time_limit_s=time_limit_s,
                lb_spec=lb_spec,
                dominance=dominance,
                target_makespan=target_makespan,
                stop_on_first_solution=stop_on_first_solution,
            )

        t0 = time.perf_counter()
        strategy = build_search_strategy(
            search_strategy,
            initial_lower_bound=lbs_initial_lower_bound,
        )
        strategy_result = strategy.run(
            solver_fn=solver_pass,
            instance=instance,
            max_nodes=max_nodes,
            order_ready_fn=order_fn,
            time_limit_s=time_limit_s,
            lb_spec=lb_spec,
            dominance=dominance_spec,
        )
        elapsed = time.perf_counter() - t0

        mk = strategy_result.best_makespan
        if strategy_result.strategy_name == "lbs":
            lb = strategy_result.known_lower_bound
        elif strategy_result.last_solver_result is not None:
            lb = compute_global_lower_bound(strategy_result.last_solver_result)
        else:
            lb = strategy_result.known_lower_bound

        rows.append(
            {
                "instance": path.name,
                "makespan": None if mk is None else int(mk),
                "lowerbound": lb,
                "cpu_time_s": float(elapsed),
                "solved": mk is not None,
            }
        )
        if optimal_by_name is not None:
            opt = optimal_by_name.get(normalize_instance_key(path.name))
            rows[-1]["optimal_makespan"] = opt
            if mk is not None and opt is not None and opt > 0:
                rows[-1]["gap_opt_pct"] = (float(mk) - float(opt)) / float(opt) * 100.0
            else:
                rows[-1]["gap_opt_pct"] = None

        if progress_every > 0:
            is_last = idx == len(paths)
            if len(paths) == 1:
                if is_last or (idx % progress_every == 0):
                    print(f"[{idx}/{len(paths)}] processed {path.name}", file=sys.stderr)
            else:
                # Keep multi-instance logs compact by default:
                # if progress_every == 1, print only the final aggregate line.
                should_print = is_last if progress_every == 1 else (is_last or (idx % progress_every == 0))
                if should_print:
                    print(f"[{idx}/{len(paths)}] processed", file=sys.stderr)

    def fmt_int(val: Optional[int]) -> str:
        return "-" if val is None else str(int(val))

    def fmt_time_s(val: object) -> str:
        return f"{float(val):.3f}"

    def fmt_gap(val: Optional[float]) -> str:
        return "-" if val is None else f"{val:.1f}%"

    headers = ["Instance", "Makespan", "Lowerbound"]
    include_opt_cols = optimal_by_name is not None
    if include_opt_cols:
        headers.extend(["Optimal", "GapOpt[%]"])
    headers.append("CPU-Time[sec.]")
    table_rows: List[List[str]] = []
    for row in rows:
        vals = [
            str(row["instance"]),
            fmt_int(row["makespan"]),  # type: ignore[arg-type]
            fmt_int(row["lowerbound"]),  # type: ignore[arg-type]
        ]
        if include_opt_cols:
            vals.append(fmt_int(row.get("optimal_makespan")))  # type: ignore[arg-type]
            vals.append(fmt_gap(row.get("gap_opt_pct")))  # type: ignore[arg-type]
        vals.append(fmt_time_s(row["cpu_time_s"]))
        table_rows.append(vals)

    col_widths = [len(h) for h in headers]
    for row in table_rows:
        for i, cell in enumerate(row):
            col_widths[i] = max(col_widths[i], len(cell))

    def fmt_row(values: List[str]) -> str:
        pieces: List[str] = []
        for idx, value in enumerate(values):
            if idx == 0:
                pieces.append(value.ljust(col_widths[idx]))
            else:
                pieces.append(value.rjust(col_widths[idx]))
        return "  ".join(pieces)

    rendered_lines: List[str] = []

    def emit(line: str = "") -> None:
        rendered_lines.append(line)
        print(line)

    emit("")
    emit(fmt_row(headers))
    emit("-" * (sum(col_widths) + 2 * (len(col_widths) - 1)))
    for row in table_rows:
        emit(fmt_row(row))

    if len(rows) > 1:
        def avg_int_field(key: str) -> Optional[float]:
            vals = [float(row[key]) for row in rows if row.get(key) is not None]
            if not vals:
                return None
            return sum(vals) / len(vals)

        def avg_float_field(key: str) -> Optional[float]:
            vals = [float(row[key]) for row in rows]
            if not vals:
                return None
            return sum(vals) / len(vals)

        avg_makespan = avg_int_field("makespan")
        avg_lowerbound = avg_int_field("lowerbound")
        avg_cpu_time = avg_float_field("cpu_time_s")

        avg_headers = list(headers)
        avg_row = ["Average"]
        avg_row.append("-" if avg_makespan is None else f"{avg_makespan:.3f}")
        avg_row.append("-" if avg_lowerbound is None else f"{avg_lowerbound:.3f}")
        if include_opt_cols:
            avg_opt = avg_int_field("optimal_makespan")
            avg_gap_opt = None
            gap_vals = [float(row["gap_opt_pct"]) for row in rows if row.get("gap_opt_pct") is not None]
            if gap_vals:
                avg_gap_opt = sum(gap_vals) / len(gap_vals)
            avg_row.append("-" if avg_opt is None else f"{avg_opt:.3f}")
            avg_row.append("-" if avg_gap_opt is None else f"{avg_gap_opt:.3f}%")
        avg_row.append("-" if avg_cpu_time is None else f"{avg_cpu_time:.3f}")
        avg_col_widths = [len(h) for h in avg_headers]
        for i, cell in enumerate(avg_row):
            avg_col_widths[i] = max(avg_col_widths[i], len(cell))

        def fmt_avg(values: List[str]) -> str:
            parts: List[str] = []
            for idx, value in enumerate(values):
                if idx == 0:
                    parts.append(value.ljust(avg_col_widths[idx]))
                else:
                    parts.append(value.rjust(avg_col_widths[idx]))
            return "  ".join(parts)

        emit("")
        emit("Average Summary")
        emit(fmt_avg(avg_headers))
        emit("-" * (sum(avg_col_widths) + 2 * (len(avg_col_widths) - 1)))
        emit(fmt_avg(avg_row))

    emit("")
    total_instances = len(rows)
    emit("Comparison Summary")
    emit(f"Instances total: {total_instances}")
    if include_opt_cols:
        with_opt = [row for row in rows if row.get("optimal_makespan") is not None]
        solved_with_opt = [row for row in with_opt if row.get("makespan") is not None]
        solved_match = [
            row
            for row in solved_with_opt
            if int(row["makespan"]) == int(row["optimal_makespan"])
        ]
        solved_count = len(solved_match)
        solved_denom = len(with_opt)
        solved_rate = (solved_count / solved_denom * 100.0) if solved_denom > 0 else 0.0
        emit(f"Solved: {solved_count}/{solved_denom} ({solved_rate:.1f}%)")
        gap_vals = [
            float(row["gap_opt_pct"])
            for row in solved_with_opt
            if row.get("gap_opt_pct") is not None
        ]
        if gap_vals:
            mean_gap = statistics.mean(gap_vals)
            median_gap = statistics.median(gap_vals)
            emit(f"Mean gap to optimal: {mean_gap:.2f}%")
            emit(f"Median gap to optimal: {median_gap:.2f}%")
        else:
            emit("Gap-to-optimal stats: unavailable (no solved instances with baseline).")
    else:
        solved_rows = [row for row in rows if row.get("solved")]
        solved_count = len(solved_rows)
        solved_rate = (solved_count / total_instances * 100.0) if total_instances > 0 else 0.0
        emit(f"Solved: {solved_count}/{total_instances} ({solved_rate:.1f}%)")

    emit(f"Note: branching order used = {branch_order}")
    emit(f"Note: search strategy used = {search_strategy}")
    emit(f"\nNote: lower bound used = {format_lower_bound_spec(lb_spec)}")
    emit(f"Note: dominance used = {format_dominance_spec(dominance_spec)}")
    if include_opt_cols:
        emit(f"Note: optimal baseline JSON used = {optimal_json}")

    if output_path is not None:
        out_path = Path(output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text("\n".join(rendered_lines) + "\n")
        print(f"Saved results to {out_path}")


if __name__ == "__main__":
    main()
