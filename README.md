# RCPSP Branch-and-Bound RL

This project learns an activity-ordering policy for exact depth-first branch and
bound on the Resource-Constrained Project Scheduling Problem (RCPSP). Behavioral
cloning (BC) can initialize the actor, or training can start from random weights.
Primal-policy training uses complete greedy schedules and direct sibling comparisons.

## Primal-policy training

Training accepts variable project sizes, including J30 through RG300. Each
episode records one greedy stem and selects up to `max_forks_per_dive` states
with two or more ready activities, spread across the stem's depths. When the
budget covers every split, all splits are selected. It forces each alternative
sibling once and completes the resulting partial schedules greedily with batched inference.
The stem outcome supplies its chosen sibling's result under the same greedy
continuation rule. Training does not run DFS, use an incumbent or dominance table,
or construct a critic. Every selected sibling reaches a complete schedule; no
known optimal makespan is required for training.

The loss is tie-aware RankNet: average strict sibling-pair losses within each
informative state, then average across those states. Equal makespans create no
preference. A small entropy term and measured categorical `KL(old || current)`
limit collapse and excessive updates. The KL guard checks every collected
state in bounded batches after each optimizer step and stops further steps when the mean exceeds
`target_kl`; it does not undo the step that crossed the threshold. All-tied batches
skip optimization. Greedy collection and optimization disable dropout.

Run from the repository root using the project environment:

```bash
/Users/mac/Documents/teak/env/bin/python scripts/train_rl.py --config config/train_rl.json --log
```

The trainer uses actor-only RankNet. `config/train_rl.json` starts from random weights, collects 32
instances per update, and writes artifacts under `models/primal-policy/`. Set
`device` to `cpu` for a local run; unavailable CUDA also falls back to CPU.

Set `"bc_checkpoint": null` to train from scratch. The seeded random Actor uses
`d_model: 64`, `n_heads: 4`, `n_layers: 2`, and `ffn_dim: 128` by default;
these architecture settings can be changed in the training config. A checkpoint
path to a newly trained BC actor loads its saved architecture and weights instead.
Both paths use 14 global / 21 candidate / 7 resource / 8 interaction inputs, disabled
dropout, and incumbent-free observations. Startup logs and saved metadata record
the initialization source. An invalid checkpoint path raises an error.

`total_updates` and `episodes_per_update` control the experiment budget. There
are no secondary stems. Collection and diagnostics use these tunable limits:

| Configuration key | Default | Meaning |
|---|---:|---|
| `max_forks_per_dive` | 32 | Maximum parent states evaluated per stem; `null` evaluates all splits. |
| `max_active_forks` | 128 | Maximum live alternative schedules held in the rollout pool. |
| `inference_chunk_size` | 128 | Maximum continuation observations passed to one Actor forward call. |
| `metrics_batch_size` | 128 | Maximum states per forward call when measuring ranking and KL. |

The default fork budget keeps every J30 split while selecting a subset on larger
projects. Every sibling at a selected state is evaluated, including groups
larger than the active pool. Only selected full snapshots are retained; pending
siblings are created as pool slots become available. Actor parameters stay fixed
throughout collection, so delaying a continuation does not change its rule.
Inference chunks bound device tensors separately from the CPU scheduling pool.
The limits bound state and batch counts; wider ready sets and longer resource
profiles still increase the memory and CPU work of each state.

Ranking summaries run before and after optimization. Each optimizer step checks
only categorical KL, across all collected states in bounded batches, without
constructing all sibling-pair comparisons again. Chunk aggregation weights
RankNet by informative states, accuracy by strict pairs, and KL/entropy by all
states. These are the same metric definitions as the unchunked calculation.

Updates use minibatches of 128 informative states and two epochs. For an
illustrative batch with about 272 informative states, this gives three
minibatches per epoch and six optimizer steps, unless the KL guard stops early.

For a quick end-to-end validation, copy the config and set `total_updates: 1`,
`episodes_per_update: 1`, `max_instances: 1`, `eval_limit: 1`,
`eval_max_nodes: 100`, and `tensorboard: false`. Use a separate `save_path` and
`checkpoint_dir` for each experiment; evaluation records append in the output directory.

## Observation and checkpoint contract

BC, rollout training, and DFS evaluation share `ml/observations.py` and the
**resource_set** packing features. Bounds and resource-feasible earliest starts
are recomputed without an incumbent or the solver's cached mask. Latest starts
use `max(local lower bound, dynamic critical path)` as an aspiration horizon;
signed slack and time windows can be negative without declaring infeasibility.
This horizon never changes solver pruning.

The resource profile is shared with scheduling. Resource tokens and global
occupancy summaries query `min_est_ready`; interactions query each candidate's
own EST before placing it. Total resource energy and root critical-path length
are cached per instance. Energy pressure uses root CP and retains values above
one. Empty resource sets use one all-zero sentinel token.

The exact column order is declared by `SET_*_FEATURE_NAMES` in
`src/rcpsp_bnb_rl/ml/il/featurize.py`:

| Input | Width | Ordered columns |
|---|---:|---|
| Global | 14 | frac_scheduled, makespan_so_far_norm, lb_norm, lb_gap_norm, rem_dur_sum_norm, rem_dur_mean_norm, rem_dur_max_norm, num_ready_norm, n_res_norm, max_energy_pressure, mean_occ_at_min_est, max_occ_at_min_est, min_est_ready_norm, max_dyn_rem_pressure |
| Candidate | 21 | duration_norm, res_intensity, total_resource_load, head_norm, tail_norm, head_dur_tail_norm, is_critical, num_succ_norm, num_pred_norm, succ_dur_sum_norm, est_norm, est_minus_head_norm, finish_norm, conflict_count_norm, slack_to_lb_signed, ls_norm, time_window_signed, is_urgent, makespan_ext_norm, rel_est_norm, bottleneck_weighted_desc_load |
| Resource | 7 | is_real_resource, occ_frac_at_min_est, avail_frac_at_min_est, rem_energy_norm, energy_pressure, dyn_rem_energy_pressure, exclusive_load_pressure |
| Candidate-resource | 8 | demand_frac, conflict_load_norm, has_demand, rem_after_demand_at_est_frac, desc_demand_energy_frac, direct_succ_demand_frac, is_heavy_demand, mean_rem_after_demand_interval_frac |

Time quantities use `S=max(1,sum(durations))`; activity counts use total activities
`N`, including dummy endpoints. `n_res_norm=min(1,R/N)`. Ready conflict counts use
ready-set size; demand and occupancy use capacity (zero-capacity denominators
use one). Remaining energy uses `cap*S`; total-energy pressure uses
`cap*max(1,root_cp)`. Occupancy alone is clamped to [0,1]; signed slack/windows
and energy pressure are not clamped. Urgency is `time_window_signed < 1/N`.

Dynamic pressure divides remaining resource energy by `cap*H`, where
`H=max(1,target_horizon-min_est_ready)`. Exclusive-load pressure divides the
remaining duration of activities with demand strictly above half capacity by
`H`. Descendant energy uses `cap*root_cp`; direct-successor demand uses `cap`.
Both downstream aggregates intersect with unscheduled activities, and direct
successors may still have other unsatisfied predecessors. The candidate's
derived bottleneck load sums descendant energy fractions weighted by dynamic
resource pressure. Interval residual capacity averages the live pre-action
profile over `[EST,EST+duration)`, falling back to the start-point residual for
zero duration. New pressure and downstream features return zero when capacity
or their critical-path denominator is zero.

Checkpoints record both `observation_schema` and `feature_schema` as
`resource_set`, together with the input widths and training configuration.
Old resource-set checkpoints are rejected; retrain BC from the existing raw
teacher trajectories or use scratch training. There is one feature path and
no automatic weight migration. Teacher action labels are preserved while inputs
are recomputed through the shared builder; configure the same `lower_bound` in
BC and rollout training. Solver bound pruning, DFS, and dominance are unchanged.

## Controlled exact-search evaluation

Training begins without evaluation. When `eval_root` is configured, the trainer
evaluates the updated actor at each evaluation interval and after the final
update. Omitting `eval_root` uses `data/eval/intrain/J30_10`; explicitly setting
`eval_root: null` disables evaluation. Evaluation uses exact DFS with `dominance="set_based"`.
Training evaluation allows at most **5,000 expanded nodes per instance**, with
`eval_max_nodes: 5000` and `eval_time_limit_s: null`. Search can finish earlier
when optimality is proved. A time limit does not truncate this node budget.
For a separate time-limited experiment, explicitly set `eval_max_nodes: null`
and a positive `eval_time_limit_s`; if both limits are set, search stops at the first.
The `compare_policies` helper compares BC and trained actors under
identical budgets and observations.

`evaluation.json` is a readable JSON array with one flat record per evaluation:

```json
[
  {"update": 10, "solved": 0.2, "primal_opt": 0.4, "primal_gap": 1.819, "dive1_gap": 5.171, "nodes": 4187.1}
]
```

The evaluator computes five metrics. Its in-memory result contains the same
five summary fields and per-instance rows with the instance name plus those
fields; gaps use percentage units throughout. Reporting adds the update number
and formats counts for the console. The active search budget is recorded in the
training configuration saved with each checkpoint.

`solved` is the fraction with optimality proved across all evaluation instances.
`primal_opt` is the fraction reaching a known optimum among instances with a
known optimum, including failures to find a schedule. Finding that makespan
within a node or time limit does not itself count as a proof. Both gaps are percentages
relative to the known optimum; unknown references produce null metrics. Gap
means use instances with both a schedule and a known optimum. `nodes` is the mean
number of expanded nodes, retained at full precision in JSON and rounded for
console display. The console shows counts as well as percentages:

```text
[Eval 10] solved=2/10 (20%, proven)  primal_opt=4/10 (40%, found)  primal_gap=1.82%  dive1_gap=5.17%  nodes=4187
```

The standalone `run_bnb.py` summary likewise reports `Proven optimal` separately
from `Optimal makespan found`. Its labels CSV retains `solved`, which means
optimality was proved.

Training prints aligned episode rows and an update summary, with evaluation
blocks separated by dotted lines and blank lines. No `training.jsonl` is written.
For example:

```text
[Episode 1524] SP_inst245001.rcp stem=96 best=91 lb=84 forks=29 signal=27 beat=5
[Update 10] states=750 info=260 pair_acc=0.68 impr=2.4 loss=0.6965->0.6758 KL=0.00291 steps=6 wall=14.92s
```

`forks=32/298` means 32 evaluated parent states out of 298 eligible stem splits;
the denominator is omitted when every split is evaluated. `best` is the minimum
makespan across the stem and all completed forks at selected states. `lb` is
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
With a fork budget, `best`, `beat`, and `impr` describe the evaluated subset.

For evaluation on larger projects, explicitly set `eval_root` and provide
`eval_optimal_json` in the existing proven-optimum format, or set that reference
to `null` when optima are unavailable. The default evaluation remains the J30
set. The merged J60/J120 baseline files include unproven upper bounds and must
not be treated as known optima. Exact DFS and its pruning are unchanged.

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
- `src/rcpsp_bnb_rl/ml/rl/env.py`: cloneable schedules and fork collection.
- `src/rcpsp_bnb_rl/ml/rl/primal_training.py`: ranking optimization and KL checks.
- `src/rcpsp_bnb_rl/ml/rl/evaluation.py`: exact-search controlled comparisons.

The codebase contains one Actor model, one packing feature schema, BC training,
sibling-ranking training, and exact-search evaluation. Actor forwards return
logits directly. `load_policy_checkpoint` and `save_policy_checkpoint` are the
checkpoint API for both training stages and evaluation.

```bash
PYTHONPATH=src /Users/mac/Documents/teak/env/bin/python -m unittest discover -s tests -v
```

Tests cover incumbent-free features, strict checkpoint compatibility, cloning,
solver first-dive parity, independently replayed sibling outcomes, batching,
tie-aware state averaging, KL stopping, target/proof reporting, and a complete
J30 training/update/checkpoint/evaluation cycle with no startup evaluation. These checks validate the
implementation; sustained training runs are needed to assess policy gains.
