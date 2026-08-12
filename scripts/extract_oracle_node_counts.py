"""Extract an episode-indexed node-count reference from a PPO training log.

The resulting JSONL is a matched-reference dataset for the oracle-alpha
ablation: episode *k* in a rerun with the same seed must use the recorded node
count for episode *k*, after verifying that the selected instance is identical.

The node count is intentionally retained for both ``search_exhausted`` and
``time_limit`` episodes.  It measures the B&B work actually observed inside the
locked PPO episode horizon, rather than the (often unavailable) nodes required
to prove an instance optimal.

Example:
    python3 scripts/extract_oracle_node_counts.py \
        --log models/train-div-2048/policy_ppo_train_log.txt \
        --output data/oracles/train_div_2048_seed42_nodes.jsonl \
        --seed 42
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any


EPISODE_RE = re.compile(
    r"^\[Episode (?P<episode>\d+)\] Done  "
    r"Instance=(?P<instance>\S+)  "
    r"Reason=(?P<done_reason>\S+)  "
    r"Steps=(?P<steps>\d+)  "
    r"Nodes=(?P<nodes>\d+)  "
    r"Best_Ms=(?P<best_makespan>\S+)  "
    r"Inc_Improves=(?P<incumbent_improvements>\d+)  "
    r"N_est=(?P<reference_n_est>\d+)  "
    r"alpha=(?P<reference_alpha>\S+)  "
    r"rewards=\(G_root:(?P<g_root>[+-]?\d+(?:\.\d+)?), "
    r"G_cost:(?P<g_cost>[+-]?\d+(?:\.\d+)?), "
    r"G_first_incum:(?P<g_first_incum>[+-]?\d+(?:\.\d+)?), "
    r"G_incum_impro:(?P<g_incum_impro>[+-]?\d+(?:\.\d+)?)\)  "
    r"elapsed=(?P<elapsed_s>\d+)s$"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _optional_int(token: str) -> int | None:
    return None if token == "-" else int(token)


def parse_records(log_path: Path) -> list[dict[str, Any]]:
    """Parse and validate every episode-completion line in ``log_path``."""
    records: list[dict[str, Any]] = []
    seen_episodes: set[int] = set()

    for line_number, raw_line in enumerate(log_path.read_text().splitlines(), start=1):
        if not raw_line.startswith("[Episode ") or "] Done  " not in raw_line:
            continue
        match = EPISODE_RE.fullmatch(raw_line)
        if match is None:
            raise ValueError(
                f"{log_path}:{line_number}: malformed episode-completion line:\n{raw_line}"
            )

        raw = match.groupdict()
        episode = int(raw["episode"])
        if episode in seen_episodes:
            raise ValueError(f"{log_path}:{line_number}: duplicate episode {episode}")
        seen_episodes.add(episode)

        steps = int(raw["steps"])
        nodes = int(raw["nodes"])
        # This reference is only valid for the current one-decision/one-expanded-
        # node environment. A mismatch signals a changed logging/solver contract,
        # so fail rather than silently emit an ambiguous oracle target.
        if nodes != steps:
            raise ValueError(
                f"{log_path}:{line_number}: episode {episode} has steps={steps} "
                f"but nodes={nodes}; expected equality for this run."
            )

        records.append(
            {
                "episode": episode,
                "instance": raw["instance"],
                "nodes": nodes,
                "steps": steps,
                "done_reason": raw["done_reason"],
                "best_makespan": _optional_int(raw["best_makespan"]),
                "incumbent_improvements": int(raw["incumbent_improvements"]),
                "reference_n_est": int(raw["reference_n_est"]),
                "reference_alpha": float(raw["reference_alpha"]),
                "reference_rewards": {
                    "root": float(raw["g_root"]),
                    "cost": float(raw["g_cost"]),
                    "first_incumbent": float(raw["g_first_incum"]),
                    "incumbent_improvement": float(raw["g_incum_impro"]),
                },
                "elapsed_s": int(raw["elapsed_s"]),
            }
        )

    if not records:
        raise ValueError(f"No episode-completion records found in {log_path}")

    actual = [record["episode"] for record in records]
    expected = list(range(1, len(records) + 1))
    if actual != expected:
        first_bad = next(
            (idx for idx, (got, want) in enumerate(zip(actual, expected), start=1) if got != want),
            min(len(actual), len(expected)) + 1,
        )
        raise ValueError(
            f"Episode ids must be contiguous 1..{len(records)}; "
            f"first mismatch at position {first_bad}."
        )
    return records


def write_reference(
    records: list[dict[str, Any]],
    *,
    log_path: Path,
    output_path: Path,
    seed: int,
) -> Path:
    """Atomically write JSONL records and a provenance sidecar JSON file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    with tmp_path.open("w") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True, separators=(",", ":")))
            handle.write("\n")
    tmp_path.replace(output_path)

    reasons = Counter(str(record["done_reason"]) for record in records)
    metadata = {
        "schema_version": 1,
        "purpose": "Episode-indexed horizon-bounded B&B node-count reference for oracle alpha.",
        "key": ["episode", "instance"],
        "seed": seed,
        "episode_count": len(records),
        "done_reason_counts": dict(sorted(reasons.items())),
        "source_log": str(log_path),
        "source_log_sha256": _sha256(log_path),
        "node_count_definition": (
            "Realized BnBSolver.nodes_expanded at episode completion, including "
            "time_limit episodes; it is the 120-second training-horizon effort target."
        ),
        "validation": {
            "episode_ids_contiguous": True,
            "nodes_equal_steps": True,
            "parser": "scripts/extract_oracle_node_counts.py",
        },
    }
    meta_path = output_path.with_suffix(output_path.suffix + ".meta.json")
    meta_tmp_path = meta_path.with_suffix(meta_path.suffix + ".tmp")
    meta_tmp_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    meta_tmp_path.replace(meta_path)
    return meta_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", required=True, type=Path, help="PPO training log to parse.")
    parser.add_argument("--output", required=True, type=Path, help="Output .jsonl reference path.")
    parser.add_argument("--seed", required=True, type=int, help="Seed used for the reference PPO run.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.log.is_file():
        raise FileNotFoundError(f"Training log not found: {args.log}")
    records = parse_records(args.log)
    meta_path = write_reference(
        records,
        log_path=args.log,
        output_path=args.output,
        seed=args.seed,
    )
    print(f"Wrote {len(records)} episode references to {args.output}")
    print(f"Metadata: {meta_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
