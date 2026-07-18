"""Smoke test for the estimator feature extractor."""

import sys
from pathlib import Path

# Ensure the src directory is importable when running from the repo root.
# This file now lives at src/rcpsp_bb_rl/ml/estimator/; the repo root is 4 levels up.
PROJECT_ROOT = Path(__file__).resolve().parents[4]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from rcpsp_bb_rl.data.parsing import load_instance  # noqa: E402
from rcpsp_bb_rl.ml.estimator import (  # noqa: E402
    FEATURE_NAMES,
    NUM_FEATURES,
    extract_feature_dict,
    extract_features,
)

print(f"NUM_FEATURES = {NUM_FEATURES}")

files = sorted(Path("data/train/1kNetRes").glob("*.rcp"))[:3]
for f in files:
    inst = load_instance(f)
    d = extract_feature_dict(inst)
    vec = extract_features(inst)
    assert vec.shape == (NUM_FEATURES,), vec.shape
    print(f"\n=== {f.name} (acts={inst.num_activities}, res={inst.num_resources}) ===")
    for name in FEATURE_NAMES:
        print(f"  {name:24s} {d[name]:.4f}")
