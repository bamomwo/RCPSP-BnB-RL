"""
Build the search-effort estimator dataset.

Two subcommands, run in order:

  build-features  Walk an instance folder, compute x(I) per instance, write
                  features.csv. No solver needed.

  join            Join features.csv with a labels.csv produced by
                  run_bnb.py --emit-csv, compute y = log(1 + nodes), and write
                  the final data.csv consumed by train_estimator.py.

Examples:
  python3 scripts/build_estimator_dataset.py build-features \
      --root data/estimator --out data/estimator/features.csv

  python3 scripts/build_estimator_dataset.py join \
      --features data/estimator/features.csv \
      --labels data/estimator/labels.csv \
      --out data/estimator/data.csv
"""

import argparse
import sys
from pathlib import Path

# Ensure the src directory is importable when running from the repo root.
# This file now lives at src/rcpsp_bb_rl/ml/estimator/; the repo root is 4 levels up.
PROJECT_ROOT = Path(__file__).resolve().parents[4]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from rcpsp_bb_rl.ml.estimator.dataset import (  # noqa: E402
    DEFAULT_PATTERNS,
    build_features_csv,
    join_features_labels,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the estimator dataset.")
    sub = parser.add_subparsers(dest="command", required=True)

    p_feat = sub.add_parser("build-features", help="Compute features.csv from an instance folder.")
    p_feat.add_argument("--root", required=True, help="Folder of RCPSP instances.")
    p_feat.add_argument("--out", required=True, help="Output features.csv path.")
    p_feat.add_argument(
        "--patterns",
        nargs="+",
        default=list(DEFAULT_PATTERNS),
        help="Glob patterns for instance files (default: *.rcp).",
    )

    p_join = sub.add_parser("join", help="Join features.csv + labels.csv into data.csv.")
    p_join.add_argument("--features", required=True, help="features.csv path.")
    p_join.add_argument("--labels", required=True, help="labels.csv path (from run_bnb.py --emit-csv).")
    p_join.add_argument("--out", required=True, help="Output data.csv path.")

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "build-features":
        n = build_features_csv(args.root, args.out, patterns=tuple(args.patterns))
        print(f"Wrote {n} feature rows to {args.out}")
    elif args.command == "join":
        n = join_features_labels(args.features, args.labels, args.out)
        print(f"Wrote {n} joined rows to {args.out}")


if __name__ == "__main__":
    main()
