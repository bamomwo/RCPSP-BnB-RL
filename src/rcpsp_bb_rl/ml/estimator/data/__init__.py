"""
Dataset construction for the search-effort estimator.

features.py maps an instance to the static feature vector x(I); dataset.py builds
features.csv, joins it with run_bnb.py labels into data.csv, and splits it for
training. Neither module imports torch, so both are usable from plain data tooling.
"""

from rcpsp_bb_rl.ml.estimator.data.dataset import (
    DEFAULT_PATTERNS,
    EstimatorSplit,
    build_features_csv,
    join_features_labels,
    load_estimator_dataset,
)
from rcpsp_bb_rl.ml.estimator.data.features import (
    FEATURE_NAMES,
    NUM_FEATURES,
    extract_feature_dict,
    extract_features,
)

__all__ = [
    "FEATURE_NAMES",
    "NUM_FEATURES",
    "extract_feature_dict",
    "extract_features",
    "DEFAULT_PATTERNS",
    "build_features_csv",
    "join_features_labels",
    "load_estimator_dataset",
    "EstimatorSplit",
]
