import argparse
import sys
from pathlib import Path

import torch

# Ensure the src directory is importable when running from the repo root.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from rcpsp_bb_rl.bnb.policy_guidance import make_policy_order_fn  # noqa: E402
from rcpsp_bb_rl.bnb.core import BnBSolver  # noqa: E402
from rcpsp_bb_rl.data.parsing import load_instance  # noqa: E402
from rcpsp_bb_rl.models import load_policy_checkpoint  # noqa: E402
from rcpsp_bb_rl.visualization.bnb_tree import render_bnb_tree  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a simple B&B solver on an RCPSP instance.")
    parser.add_argument(
        "--instance",
        default="data/j60rcp/J601_1.RCP",
        help="Path to the RCPSP instance file.",
    )
    parser.add_argument(
        "--max-nodes",
        type=int,
        default=2000,
        help="Maximum number of nodes to expand before stopping.",
    )
    parser.add_argument(
        "--policy",
        default=None,
        help="Optional path to a policy checkpoint (.pt) to guide branching.",
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
        "--output-dir",
        default="reports/figures",
        help="Directory to write the B&B tree visualization.",
    )
    parser.add_argument(
        "--format",
        default="pdf",
        choices=["pdf", "png", "svg"],
        help="Output format for the rendered tree.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    instance = load_instance(args.instance)
    solver = BnBSolver(instance)
    order_fn = None

    if args.policy:
        device = torch.device(args.policy_device)
        policy = load_policy_checkpoint(args.policy, device=device)
        order_fn = make_policy_order_fn(
            instance=instance,
            model=policy,
            max_resources=args.policy_max_resources,
            device=device,
            predecessors=solver.predecessors,
        )

    result = solver.solve(max_nodes=args.max_nodes, order_ready_fn=order_fn)

    if result.best_makespan is None:
        print("No feasible schedule found within the node limit.")
    else:
        print(f"Best makespan: {result.best_makespan}")

    outpath = render_bnb_tree(result.nodes, result.edges, output_dir=args.output_dir, name="bnb_tree", fmt=args.format)
    print(f"Wrote B&B tree visualization to {outpath}")


if __name__ == "__main__":
    main()
