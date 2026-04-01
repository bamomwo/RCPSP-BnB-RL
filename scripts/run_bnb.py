import argparse
import json
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


def validate_and_resolve(
    args: argparse.Namespace,
    cfg: Dict[str, Any],
) -> tuple[Optional[int], Optional[float], str, Optional[str], str, int, List[str], object]:
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

    return (
        max_nodes,
        time_limit_s,
        branch_order,
        (None if policy_path is None else str(policy_path)),
        policy_device,
        policy_max_resources,
        lb_spec,
        dominance_spec,
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
    max_nodes, time_limit_s, branch_order, policy_path, policy_device, policy_max_resources, lb_spec, dominance_spec = validate_and_resolve(args, cfg)
    paths = resolve_paths(args, cfg)

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
        solver = BnBSolver(instance)
        order_fn = None
        if use_policy:
            order_fn = make_order_fn(
                "policy",
                instance=instance,
                model=policy_model,
                max_resources=policy_max_resources,
                device=device,
                predecessors=solver.predecessors,
            )
        elif branch_order == "lower_bound":
            order_fn = make_order_fn(
                "lower_bound",
                instance=instance,
                predecessors=solver.predecessors,
                lb_id=lb_spec,
            )

        t0 = time.perf_counter()
        result = solver.solve(
            max_nodes=max_nodes,
            order_ready_fn=order_fn,
            time_limit_s=time_limit_s,
            lb_spec=lb_spec,
            dominance=dominance_spec,
        )
        elapsed = time.perf_counter() - t0

        mk = result.best_makespan
        lb = compute_global_lower_bound(result)

        rows.append(
            {
                "instance": path.name,
                "makespan": None if mk is None else int(mk),
                "lowerbound": lb,
                "cpu_time_s": float(elapsed),
            }
        )

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

    headers = ["Instance", "Makespan", "Lowerbound", "CPU-Time[sec.]"]
    table_rows: List[List[str]] = []
    for row in rows:
        table_rows.append(
            [
                str(row["instance"]),
                fmt_int(row["makespan"]),  # type: ignore[arg-type]
                fmt_int(row["lowerbound"]),  # type: ignore[arg-type]
                fmt_time_s(row["cpu_time_s"]),
            ]
        )

    col_widths = [len(h) for h in headers]
    for row in table_rows:
        for i, cell in enumerate(row):
            col_widths[i] = max(col_widths[i], len(cell))

    def fmt_row(values: List[str]) -> str:
        return (
            values[0].ljust(col_widths[0])
            + "  "
            + values[1].rjust(col_widths[1])
            + "  "
            + values[2].rjust(col_widths[2])
            + "  "
            + values[3].rjust(col_widths[3])
        )

    print("")
    print(fmt_row(headers))
    print("-" * (sum(col_widths) + 2 * (len(col_widths) - 1)))
    for row in table_rows:
        print(fmt_row(row))

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

        avg_headers = headers
        avg_row = [
            "Average",
            "-" if avg_makespan is None else f"{avg_makespan:.3f}",
            "-" if avg_lowerbound is None else f"{avg_lowerbound:.3f}",
            "-" if avg_cpu_time is None else f"{avg_cpu_time:.3f}",
        ]
        avg_col_widths = [len(h) for h in avg_headers]
        for i, cell in enumerate(avg_row):
            avg_col_widths[i] = max(avg_col_widths[i], len(cell))

        def fmt_avg(values: List[str]) -> str:
            return (
                values[0].ljust(avg_col_widths[0])
                + "  "
                + values[1].rjust(avg_col_widths[1])
                + "  "
                + values[2].rjust(avg_col_widths[2])
                + "  "
                + values[3].rjust(avg_col_widths[3])
            )

        print("")
        print("Average Summary")
        print(fmt_avg(avg_headers))
        print("-" * (sum(avg_col_widths) + 2 * (len(avg_col_widths) - 1)))
        print(fmt_avg(avg_row))

    print(f"Note: branching order used = {branch_order}")
    print(f"\nNote: lower bound used = {format_lower_bound_spec(lb_spec)}")
    print(f"Note: dominance used = {format_dominance_spec(dominance_spec)}")


if __name__ == "__main__":
    main()
