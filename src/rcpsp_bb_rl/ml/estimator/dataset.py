"""
Dataset assembly for the search-effort estimator.

Two stages, run in order:

  1. build-features: walk an instance folder, compute x(I) for each instance via
     features.py, and write features.csv (instance name + 18 feature columns).
     No solver needed — fast.

  2. join: read features.csv and a labels.csv produced by run_bnb.py
     (--emit-csv), join by instance name, compute y = log(1 + nodes), carry the
     solved flag, and write the final data.csv consumed by train_estimator.py.

Keeping the two stages separate means features.csv can be generated and
inspected immediately while the slow baseline solver produces labels.csv in
parallel. The join keys on instance filename and fails loudly on mismatch.
"""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from rcpsp_bb_rl.data.dataset import list_instance_paths
from rcpsp_bb_rl.data.parsing import load_instance
from rcpsp_bb_rl.ml.estimator.features import FEATURE_NAMES, extract_feature_dict

DEFAULT_PATTERNS: Sequence[str] = ("*.RCP", "*.rcp")

# Columns present in data.csv that are NOT model inputs. `instance` is metadata,
# `nodes` is the pre-log node count, `y` is the target, and `solved` is withheld
# on purpose: exposing whether the search exhausted leaks information about the
# target the model is trying to predict.
NON_FEATURE_COLUMNS: Sequence[str] = ("instance", "nodes", "y", "solved")
TARGET_COLUMN = "y"


# ---------------------------------------------------------------------------
# Stage 1: feature emission
# ---------------------------------------------------------------------------

def build_features_csv(
    root: Path | str,
    out_path: Path | str,
    patterns: Sequence[str] = DEFAULT_PATTERNS,
) -> int:
    """
    Compute features for every instance under root and write features.csv.

    Columns: instance, <FEATURE_NAMES...>. One row per instance. Returns the
    number of rows written.
    """
    paths = list_instance_paths(root, patterns=tuple(patterns))
    if not paths:
        raise FileNotFoundError(f"No instances found under {root} with patterns {patterns}")

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    header = ["instance"] + list(FEATURE_NAMES)
    n_written = 0
    with out_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for path in paths:
            instance = load_instance(path)
            feats = extract_feature_dict(instance)
            row = [path.name] + [feats[name] for name in FEATURE_NAMES]
            writer.writerow(row)
            n_written += 1

    return n_written


# ---------------------------------------------------------------------------
# Stage 2: join features + labels -> data.csv
# ---------------------------------------------------------------------------

def _read_csv_rows(path: Path | str) -> List[Dict[str, str]]:
    with Path(path).open(newline="") as f:
        return list(csv.DictReader(f))


def join_features_labels(
    features_csv: Path | str,
    labels_csv: Path | str,
    out_path: Path | str,
) -> int:
    """
    Join features.csv with labels.csv (from run_bnb.py --emit-csv) by instance
    name and write the training dataset data.csv.

    labels.csv must provide columns: instance, nodes, solved. The label is
    y = log(1 + nodes); solved=1 means the search exhausted (true node count),
    solved=0 means it timed out at the horizon (the count is a floor).

    Output columns: instance, <FEATURE_NAMES...>, nodes, y, solved.
    Returns the number of joined rows written.
    """
    feat_rows = _read_csv_rows(features_csv)
    label_rows = _read_csv_rows(labels_csv)

    feat_by_name = {r["instance"]: r for r in feat_rows}
    label_by_name = {r["instance"]: r for r in label_rows}

    feat_names = set(feat_by_name)
    label_names = set(label_by_name)
    missing_labels = feat_names - label_names
    missing_feats = label_names - feat_names
    if missing_labels:
        raise ValueError(
            f"{len(missing_labels)} instance(s) in features but not labels "
            f"(e.g. {sorted(missing_labels)[:5]}). Did run_bnb.py cover all instances?"
        )
    if missing_feats:
        raise ValueError(
            f"{len(missing_feats)} instance(s) in labels but not features "
            f"(e.g. {sorted(missing_feats)[:5]}). Did build-features cover all instances?"
        )

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    header = ["instance"] + list(FEATURE_NAMES) + ["nodes", "y", "solved"]
    n_written = 0
    with out_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for name in sorted(feat_by_name):
            frow = feat_by_name[name]
            lrow = label_by_name[name]
            nodes = int(float(lrow["nodes"]))
            y = math.log1p(nodes)
            solved = _parse_bool(lrow["solved"])
            row = (
                [name]
                + [frow[fn] for fn in FEATURE_NAMES]
                + [nodes, y, int(solved)]
            )
            writer.writerow(row)
            n_written += 1

    return n_written


def _parse_bool(value: str) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y", "t"}


# ---------------------------------------------------------------------------
# Stage 3: load data.csv -> train/val arrays for the estimator
# ---------------------------------------------------------------------------

@dataclass
class EstimatorSplit:
    """A train/val split of the estimator dataset (raw, unscaled features)."""
    x_train: np.ndarray      # [n_train, NUM_FEATURES]
    y_train: np.ndarray      # [n_train]
    x_val: np.ndarray        # [n_val, NUM_FEATURES]
    y_val: np.ndarray        # [n_val]
    feature_names: List[str]
    instances_train: List[str]
    instances_val: List[str]
    nodes_train: np.ndarray  # [n_train] raw node counts (for node-space metrics)
    nodes_val: np.ndarray    # [n_val]


def load_estimator_dataset(
    data_csv: Path | str,
    val_frac: float = 0.2,
    seed: int = 42,
) -> EstimatorSplit:
    """
    Load the joined data.csv, select the 18 feature columns as X and `y` as the
    target, and split randomly into train/val.

    The NON_FEATURE_COLUMNS (instance, nodes, y, solved) are excluded from X;
    `solved` in particular is withheld so the model gets no hint about the target.
    Features are returned RAW (unscaled) — the caller fits the Standardizer on the
    train split only, so no statistics leak from val into training.
    """
    rows = _read_csv_rows(data_csv)
    if not rows:
        raise ValueError(f"No rows found in {data_csv}")

    # Freeze the feature order to FEATURE_NAMES (the estimator's input contract),
    # not to whatever column order the CSV happens to have.
    header = set(rows[0].keys())
    missing = [c for c in FEATURE_NAMES if c not in header]
    if missing:
        raise ValueError(
            f"data.csv is missing expected feature columns: {missing}. "
            f"Was it produced by join_features_labels against the current features.py?"
        )
    if TARGET_COLUMN not in header:
        raise ValueError(f"data.csv is missing the target column '{TARGET_COLUMN}'.")

    x = np.array(
        [[float(r[name]) for name in FEATURE_NAMES] for r in rows],
        dtype=np.float64,
    )
    y = np.array([float(r[TARGET_COLUMN]) for r in rows], dtype=np.float64)
    nodes = np.array([float(r["nodes"]) for r in rows], dtype=np.float64)
    instances = [r["instance"] for r in rows]

    n = len(rows)
    if not (0.0 < val_frac < 1.0):
        raise ValueError(f"val_frac must be in (0, 1); got {val_frac}")

    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    n_val = max(1, int(round(n * val_frac)))
    val_idx = perm[:n_val]
    train_idx = perm[n_val:]

    return EstimatorSplit(
        x_train=x[train_idx],
        y_train=y[train_idx],
        x_val=x[val_idx],
        y_val=y[val_idx],
        feature_names=list(FEATURE_NAMES),
        instances_train=[instances[i] for i in train_idx],
        instances_val=[instances[i] for i in val_idx],
        nodes_train=nodes[train_idx],
        nodes_val=nodes[val_idx],
    )
