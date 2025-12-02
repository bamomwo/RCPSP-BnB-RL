import argparse
import sys
from pathlib import Path


# Ensure the src directory is importable when running from the repo root.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from rcpsp_bb_rl.data.parsing import load_instance  # noqa: E402
from rcpsp_bb_rl.visualization.precedence import (  # noqa: E402
    build_precedence_graph,
    render_graphviz,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize an RCPSP precedence graph."
    )
    parser.add_argument(
        "--instance",
        default="data/j30rcp/J301_1.RCP",
        help="Path to the RCPSP instance file.",
    )
    parser.add_argument(
        "--output-dir",
        default="reports/figures",
        help="Directory to save the rendered graph.",
    )
    parser.add_argument(
        "--name",
        default="rcpsp_graph",
        help="Base filename (without extension) for the rendered graph.",
    )
    parser.add_argument(
        "--format",
        default="pdf",
        choices=["pdf", "png", "svg"],
        help="Output format for graphviz render.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    instance = load_instance(args.instance)
    graph = build_precedence_graph(instance.activities)
    outpath = render_graphviz(
        graph,
        output_dir=args.output_dir,
        name=args.name,
        fmt=args.format,
    )
    print(f"Wrote precedence graph to {outpath}")


if __name__ == "__main__":
    main()
