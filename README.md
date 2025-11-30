# RCPSP Branch-and-Bound RL

Lightweight ML project scaffold for training and evaluating a branching policy for RCPSP branch-and-bound.

## Layout
- `src/rcpsp_bb_rl/`: Python package for data loading, modeling, and utilities.
- `scripts/`: Entry points for experiments and tooling (add more as needed).
- `configs/`: Experiment/training configs (placeholder).
- `data/`: Input data; `raw` holds source `.RCP` instances, `processed` for derived artifacts. Data is gitignored.
- `notebooks/`: Exploratory analysis.
- `models/`: Saved checkpoints.
- `reports/figures/`: Generated visualizations.

## Quickstart
Visualize an RCPSP precedence graph (ensure dependencies like `networkx` and `graphviz` are installed):

```bash
PYTHONPATH=src python scripts/visualize_rcpsp.py \
  --instance data/j30rcp/J301_1.RCP \
  --output-dir reports/figures \
  --format pdf
```

The script parses the instance, builds the precedence DAG, and renders it with Graphviz. Add future training/evaluation entry points under `scripts/` and reuse shared code from `src/rcpsp_bb_rl/`.
