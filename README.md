# RCPSP Branch-and-Bound RL

This project learns an activity-ordering policy for exact depth-first branch and
bound on the Resource-Constrained Project Scheduling Problem (RCPSP). Behavioral
cloning (BC) initializes the actor. The first primal-policy experiment fine-tunes
that actor using complete greedy schedules and direct sibling comparisons.

## Minimal J30 experiment

Training uses **30 real activities** (the usual J30 files also contain two dummy
endpoints). Each episode records one greedy stem and snapshots every state with
two or more ready activities. It forces each alternative sibling once, pools all
resulting partial schedules, and completes them greedily with batched inference.
The stem outcome supplies its chosen sibling's result under the same greedy
continuation rule. Training does not run DFS, use an incumbent or dominance table,
or construct a critic. Every selected sibling reaches a complete schedule; no
known optimal makespan is required for training.

The loss is tie-aware RankNet: average strict sibling-pair losses within each
informative state, then average across those states. Equal makespans create no
preference. A small entropy term and measured categorical `KL(old || current)`
limit collapse and excessive updates. The KL guard checks the full collected
batch after each optimizer step and stops further steps when the mean exceeds
`target_kl`; it does not undo the step that crossed the threshold. All-tied batches
skip optimization. Greedy collection and optimization disable dropout.

Run from the repository root using the project environment:

```bash
/Users/mac/Documents/teak/env/bin/python scripts/train_ppo.py --config config/train_ppo.json --log
```

The historical filename is retained, but this trainer uses actor-only RankNet,
not PPO. `config/train_ppo.json` starts from `models/policy_bc.pt`, collects 32
instances per update, and writes artifacts under `models/primal-policy/`. Set
`device` to `cpu` for a local run; unavailable CUDA also falls back to CPU.
`total_updates` and `episodes_per_update` control the experiment budget. There
are no fork-depth caps, inference chunks, or secondary stems in this version.
Updates use minibatches of 128 informative states and two epochs. For an
illustrative batch with about 272 informative states, this gives three
minibatches per epoch and six optimizer steps, unless the KL guard stops early.

For a quick end-to-end validation, copy the config and set `total_updates: 1`,
`episodes_per_update: 1`, `max_instances: 1`, `eval_limit: 1`,
`eval_max_nodes: 100`, and `tensorboard: false`. Use a separate `save_path` and
`checkpoint_dir` for each experiment; evaluation records append in the output directory.

## Observation and checkpoint contract

Training and new-actor DFS evaluation share `ml/observations.py`. It computes
bounds and resource-feasible earliest starts with no incumbent, uses the sum of
activity durations for latest-start horizons, and fixes the reserved incumbent
gap feature to zero. It ignores the solver's cached feasibility mask and bound.
The existing resource-set widths remain **17 global / 18 candidate** features.
Snapshots, candidate ordering, and resource features depend on the partial
schedule, not search history. The actor is constructed with `policy_only=True`;
legacy BC/shared-value and separate-critic checkpoints load through the strict
`load_actor_checkpoint` extractor without changing actor weights.

Saved checkpoints record `observation_schema: incumbent_free_v1` and the training
configuration. The existing B&B policy loader recognizes this schema. Old BC and
RL checkpoints retain legacy observations unless explicitly overridden. Solver
bound pruning, DFS backtracking, and dominance remain unchanged. Feature
invariance removes incumbent leakage for the same partial schedule; it does not
make training and search visit the same states.

## Controlled exact-search evaluation

Training begins without evaluation. When `eval_root` is configured, the trainer
evaluates the updated actor at each evaluation interval and after the final
update. Omitting `eval_root` uses `data/eval/intrain/J30_10`; explicitly setting
`eval_root: null` disables evaluation. Evaluation uses exact DFS with `dominance="set_based"`.
The separate `compare_policies` helper remains available to compare original BC,
BC with incumbent-free observations, and a trained actor under identical budgets.

`evaluation.json` is a readable JSON array with one flat record per evaluation:

```json
[
  {"update": 10, "solved": 0.1, "primal_opt": 0.4, "primal_gap": 2.27, "dive1_gap": 10.75, "nodes": 103957}
]
```

`solved` is the fraction proven optimal across all evaluation instances.
`primal_opt` is the fraction reaching a known optimum among instances with a
known optimum, including failures to find a schedule. Both gaps are percentages
relative to the known optimum; unknown references produce null metrics. Gap
means use instances with both a schedule and a known optimum. `nodes` is the mean
number of expanded nodes, retained at full precision in JSON and rounded for
console display. The console shows counts as well as percentages:

```text
[Eval 10] solved=1/10 (10%)  primal_opt=4/10 (40%)  primal_gap=2.27%  dive1_gap=10.75%  nodes=103957
```

Training prints aligned episode rows and an update summary, with evaluation
blocks separated by dotted lines and blank lines. No `training.jsonl` is written.
For example:

```text
[Episode 1524] SP_inst245001.rcp stem=96 best=91 lb=84 forks=29 signal=27 beat=5
[Update 10] states=750 info=260 pair_acc=0.68 impr=2.4 loss=0.6965->0.6758 KL=0.00291 steps=6 wall=14.92s
```

`best` is the minimum makespan across the stem and all completed forks. `lb` is
the root lower bound. `beat` counts fork states where an alternative strictly
improves on the stem, once per state. `signal` includes both those states and
states where the stem is best or tied for best but other siblings are worse.
Matching `lb` certifies optimality; a gap above a loose lower bound does not
establish suboptimality.

`pair_acc` is pre-update accuracy pooled across all strictly ordered sibling
pairs; makespan ties are excluded and equal actor scores receive half credit.
It is `n/a` when there are no strict pairs. `impr` is the mean `stem - best` in
makespan units across all collected instances, including zero improvements.
These diagnostics describe current greedy continuations. Persistent trends can
guide exploration experiments; they do not independently establish
generalization or global optimality.

Add `--log` to mirror stdout and stderr to `<policy_stem>_train_log.txt` beside the
checkpoint (for `policy.pt`, this is `policy_train_log.txt`). Each invocation
overwrites that text log; evaluation records append to `evaluation.json`.
Optional TensorBoard retains the detailed training diagnostics in `tb/`.

To deploy a checkpoint in the existing runner, set `policy_path` in a copy of
`config/run_bnb.json` and run:

```bash
python scripts/run_bnb.py --config config/run_bnb.json --root data/eval/intrain/J30_10
```

## Layout and verification

- `src/rcpsp_bnb_rl/bnb/`: exact solver, scheduling, lower bounds, dominance.
- `src/rcpsp_bnb_rl/ml/il/`: teacher traces and behavioral cloning features.
- `src/rcpsp_bnb_rl/ml/models/`: transformer and checkpoint loading.
- `src/rcpsp_bnb_rl/ml/rl/single_dive.py`: cloneable schedules and fork collection.
- `src/rcpsp_bnb_rl/ml/rl/primal_training.py`: ranking optimization and KL checks.
- `src/rcpsp_bnb_rl/ml/rl/evaluation.py`: exact-search controlled comparisons.

The earlier DFS environment and tree-return helpers remain available for old
experiments; the new trainer does not use them. Stale tests for the absent
closed-tree ranking API have been replaced with tests of the new contract.

```bash
PYTHONPATH=src /Users/mac/Documents/teak/env/bin/python -m unittest discover -s tests -v
```

Tests cover incumbent-free features, strict checkpoint compatibility, cloning,
solver first-dive parity, independently replayed sibling outcomes, batching,
tie-aware state averaging, KL stopping, target/proof reporting, and a complete
J30 training/update/checkpoint/evaluation cycle with no startup evaluation. These checks validate the
implementation; sustained training runs are needed to assess policy gains.
