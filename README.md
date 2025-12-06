# RCPSP Branch-and-Bound RL

An ML project for training and evaluating a branching policy for RCPSP branch-and-bound:
- Naive B&B: depth-first with deterministic ready ordering.
- Teacher: CP-SAT produces optimal trajectories used for supervision.
- Learned policy: trained on optimal trajectories and plugged into BnBSolver to decide branching (via `--policy`).
- Reporting: compare native vs policy-guided runs and save summaries under `reports/bnb_reports`.

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
python scripts/visualize_rcpsp.py --instance data/j30rcp/J301_1.RCP --output-dir reports/figures --format pdf
```

Run the simple B&B solver and visualize its search tree:

```bash
python scripts/run_bnb.py --instance data/j30rcp/J301_1.RCP --max-nodes 2000 --output-dir reports/figures
```
Log teacher trajectories from CP-SAT (if you need to generate training data):

```bash
PYTHONPATH=src python scripts/log_teacher_traces.py --root data/j30rcp --pattern '*.RCP' --output-dir data/trajectories
```

Train the supervised policy on the logged trajectories:

```bash
PYTHONPATH=src python scripts/train_bc.py --trajectories-dir data/trajectories --output models/policy.pt
```

Run the solver with a learned branching policy (expects `models/policy.pt` saved by `scripts/train_bc.py`):

```bash
PYTHONPATH=src python scripts/run_bnb.py --instance data/j30rcp/J301_1.RCP --policy models/policy.pt --policy-max-resources 4 --policy-device cpu
```


Batch-report B&B results across a directory of instances (native only by default):

```bash
python scripts/report_bnb.py --root data/j30rcp --pattern '*.RCP' --max-nodes 2000
```

Compare native vs policy-guided branching and save a text summary:

```bash
PYTHONPATH=src python scripts/report_bnb.py --root data/j30rcp --pattern '*.RCP' --max-nodes 2000 --policy models/policy.pt --output-dir reports/bnb_reports --output-name summary.txt
```

Evaluate only OR-Tools (CP-SAT) on the first 20 instances under `data/1kNetRes`, with a 1-second per-instance time limit:

```bash
python scripts/report_bnb.py --root data/1kNetRes --pattern "*.rcp" --limit 20 --only-ortools --ortools-time-limit 1
```
This skips the native/policy B&B runs and reports the OR-Tools makespan/time columns only.

The scripts parse instances, build precedence graphs, and can render search trees with Graphviz. Add future training/evaluation entry points under `scripts/` and reuse shared code from `src/rcpsp_bb_rl/`.
