# RCPSP Branch-and-Bound RL

A transformer policy learns child visitation order inside a serial branch-and-bound solver for the Resource-Constrained Project Scheduling Problem. Training starts with behavioral cloning of optimal CP-SAT trajectories, then fine-tunes the actor from partially explored search trees.

## Layout

- `src/rcpsp_bnb_rl/bnb/`: solver, scheduling, lower bounds, dominance, and search strategies.
- `src/rcpsp_bnb_rl/ml/il/`: teacher trajectories, features, and imitation learning.
- `src/rcpsp_bnb_rl/ml/models/`: actor transformer and checkpoint loading.
- `src/rcpsp_bnb_rl/ml/rl/`: branching environment, censored subtree outcomes, and pairwise losses.
- `scripts/`: training, solver evaluation, and search visualization.
- `config/`: JSON configurations.
- `data/`, `models/`, `results/`: instances, checkpoints, and evaluation artifacts.

## Training and evaluation

Use the project's Python environment (`/Users/mac/Documents/teak/env/bin/python` in this checkout).

```bash
python scripts/log_teacher_traces.py --root data/train/1kNetRes --pattern '*.rcp' --output-dir data/train/Traj-Opt/1kNetRes
python scripts/train_bc.py --config config/train_bc.json
python scripts/train_ppo.py --config config/train_ppo.json --log
python scripts/run_bnb.py --config config/run_bnb.json --root data/eval/intrain/J30_10
python -m unittest discover -s tests
```

The historical `train_ppo.py` command now runs actor-only censored pairwise RankNet training. There are no critic updates, value targets, PPO ratios, or step penalties. The supplied BC configuration may need its trajectory paths adjusted to the locally generated data.

## Censored pairwise training

The rollout policy samples the first feasible child categorically and ranks remaining children by actor logits. Evaluation uses greedy actor order. Training and evaluation both use set-based dominance and finish the active dive when a node or wall-clock budget expires. These are fixed behaviors rather than configuration switches. Every visited complete schedule is recorded, including schedules that do not improve the incumbent.

A bottom-up pass tracks each subtree's best observed makespan, closure, and valid lower bound. Pending children are masked. Dominance dead ends generate no labels; discarded dominance children prevent strengthening a parent's lower bound through its observed children.

At a split, a realized schedule beats a strictly worse realized schedule (empirical unless the loser's lower bound certifies the preference). It also beats a strictly greater valid lower bound (certified). Equality against a bound gets a weaker `tie_break_bound` search preference. Equal realized schedules, overlapping censored outcomes, two bounds, and unknown outcomes generate no preference. `preference_mode="certified_only"` excludes empirical and equality labels.

For each active split, training averages `weight * softplus(-(winner_logit - loser_logit))` over valid pairs, then averages over split nodes. Strict-pair weight is `1 + pair_gap_alpha * relative_gap`; equality weight is `tie_break_weight`. A small `entropy_coef` and optional `bc_kl_coef` regularize the actor. `epochs` defaults to 2. Exact categorical `KL(old || new)` is checked before and after minibatch steps and over the full retained batch after each epoch; exceeding `target_kl` stops further optimization. This is a stopping guard, so the final step can exceed the threshold.

Only splits with valid pairs enter updates; forced decisions never enter the loss. Batch sizes count active split nodes, not search steps or individual pairs. The supplied defaults are sized for the CPU trial, which produced about 9 active splits per episode (median 7):

| Setting | Default | Meaning |
|---|---:|---|
| `min_batch_size` | 64 | Minimum retained active splits collected before an update. Episodes finish before checking, so a batch can exceed 64. |
| `target_mb_size` | 32 | Maximum active splits in an optimizer minibatch. Minibatches are balanced to avoid a tiny remainder. |
| `min_episodes` | 4 | Minimum distinct training instances contributing retained splits to the batch. Empty-label episodes do not count; the effective minimum is capped by dataset size. |

Every active split and all of its valid pairs are retained. There is no subsampler or per-episode cap. Both collection requirements must be met, so the batch can exceed 64 when an episode supplies many active splits or while waiting for four contributing instances. These are initial settings for sparse supervision, not an empirically optimized choice. Training ends by updating any remaining nonempty partial batch, even below the normal thresholds; an empty final batch produces no optimizer update.

The supplied fine-tuning config omits architecture settings because the BC checkpoint defines its architecture. The script still accepts `d_model`, `n_heads`, `n_layers`, and `ffn_dim` for explicit experiments with `bc_checkpoint: null`. Dropout is fixed at zero because rollout, update, and KL calculations use deterministic actor forwards; categorical action sampling supplies exploration. The learning rate remains `5e-5`, with two epochs and `target_kl: 0.02`.

Checkpoints contain only actor parameters. Existing BC and previous PPO checkpoints load their actor weights directly; obsolete critic/value payloads are discarded. Both resource-set and legacy fixed-resource actor schemas remain supported for checkpoint loading.

## Diagnostics and evaluation

The six-line startup header summarizes the run, device, datasets, search budgets, collection, and optimizer. Each completed episode prints two lines with its instance, nodes expanded, makespan, stopping reason, runtime, cumulative steps, total/active splits, pair counts, and batch progress. Three-line update summaries show retained splits and pairs, contributing instances, timings, before/after loss and accuracy, KL, entropy, pair types, and active splits at each actual decision depth. For 32 activities this includes every depth from `d0` to `d31`, including zero counts; `d8` means eight activities have been scheduled. Terminal schedules at depth 32 are not decision states. A batch mixing instance sizes shows depths through the largest instance's final decision depth. Evaluation prints one line with exact optimum matches, mean gap to known optima, proof count, mean nodes, and runtime. `opt_matched` uses only instances with a known optimal makespan and is independent of proof status or configured targets; it displays `n/a` when no optima are available.

The trainer writes `.updates.jsonl` next to `save_path`, including total and active split counts at each actual depth (`d0_total_split`, `d0_active_split`, and so on), certified/empirical/equality/censored counts, ranking accuracy and loss by pair type before and after updates, categorical KL from the rollout actor and BC actor, and early-stop statistics. TensorBoard receives `train/*` and `eval/*` scalar metrics when enabled. TensorBoard gaps and all accuracy rates use fractions (0.05 means 5%); the evaluation JSON's `mean_gap` is a percentage.

With `eval_root`, the trainer evaluates the current actor every `eval_every_updates` and at training end, using set-based dominance, `eval_max_nodes`, and `eval_time_limit_s`. Collection starts immediately; baseline comparisons can be run separately after training. `bc_checkpoint` initializes the actor and provides the frozen reference for BC KL monitoring and optional regularization. The `.eval.json` file is a flat list with one compact entry per evaluated checkpoint, without instance details or training configuration:

```json
[
  {"step": 6534000, "update": 7, "solved": 0.1, "optimal_primal": 0.2, "mean_gap": 6.81, "mean_nodes": 18003.3},
  {"step": 7466806, "update": 8, "solved": 0.1, "optimal_primal": 0.2, "mean_gap": 5.97, "mean_nodes": 18003.4}
]
```

`step` and `update` are numeric environment-step and update counts. `solved` is the fraction proven optimal across all evaluation instances; `optimal_primal` is the fraction matching the known optimum among instances with a known optimum. `mean_gap` is the mean percentage gap to known optima among runs that found a schedule (`6.81` means 6.81%), and `mean_nodes` averages expansions across all evaluation instances. Missing optimum comparisons are `null`. Known optima come from `eval_optimal_json`.

Detailed evaluation metrics remain available in TensorBoard, including target/proof effort and runtime. `eval_target_makespans` can override per-instance targets by lowercase filename stem for those metrics; targets do not change `optimal_primal`. Training configuration remains in each checkpoint.

Long training and benchmark runs are separate from the unit and smoke tests.
