from __future__ import annotations

import argparse
from typing import Optional, Sequence

from rcpsp_bb_rl.bnb.branching import (
    ReadyOrderFn,
    SerialBranchingScheme,
)
from rcpsp_bb_rl.bnb.lower_bounds import lower_bound
from rcpsp_bb_rl.bnb.precedence import build_predecessors, compute_ready_set
from rcpsp_bb_rl.bnb.solver import (
    BBNode,
    BnBSolver,
    ScheduleEntry,
    SolverResult,
    current_makespan,
    earliest_feasible_start,
    resource_feasible,
    solve_serial,
)


def _print_summary(result: SolverResult) -> None:
    if result.best_makespan is None:
        print("No feasible schedule found.")
    else:
        print(f"Best makespan: {result.best_makespan}")
    print(f"Nodes expanded: {result.nodes_expanded}")
    print(f"Nodes pruned: {result.nodes_pruned}")


def _add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--instance", required=True, help="Path to RCPSP instance file")
    parser.add_argument("--max-nodes", type=int, default=2000)
    parser.add_argument("--time-limit-s", type=float, default=None)


def _run_serial_from_args(args: argparse.Namespace) -> int:
    from rcpsp_bb_rl.data.parsing import load_instance

    instance = load_instance(args.instance)
    result = solve_serial(
        instance,
        max_nodes=args.max_nodes,
        time_limit_s=args.time_limit_s,
    )
    _print_summary(result)
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    """
    Minimal CLI entrypoint for running B&B.

    Usage:
      python -m rcpsp_bb_rl.bnb.core serial --instance ...
    """
    parser = argparse.ArgumentParser(description="Run RCPSP B&B from rcpsp_bb_rl.bnb.core")
    subparsers = parser.add_subparsers(dest="solver", required=True)

    serial_parser = subparsers.add_parser("serial", help="Use serial branching solver")
    _add_common_args(serial_parser)
    serial_parser.set_defaults(run=_run_serial_from_args)

    args = parser.parse_args(argv)
    return args.run(args)


if __name__ == "__main__":
    raise SystemExit(main())

