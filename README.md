# RCPSP Branch-and-Bound RL

A learned branching policy for the Resource-Constrained Project Scheduling Problem (RCPSP), trained to replace classic branching order methods inside a Branch-and-Bound (B&B) solver.

## Background & Approach

RCPSP is the problem of scheduling a set of activities with known durations, precedence constraints, and resource requirements to minimize project makespan (Blazewicz et al., 1983). We utlize the branch and bound procedure in solving this problem where we specifcially learn a branching policy to optimize search. A robust branching policy drastically reduces the search tree, enables aggressive pruning and faster convergence. We show that a machine learning policy outperforms classic branching order methods leading to better optimal search. 

We train a transformer-based branching policy in two stages: supervised imitation learning on optimal trajectories from OR-Tools CP-SAT, followed by PPO reinforcement learning to improve efficient search navigation.

## Layout

```
src/rcpsp_bnb_rl/
  bnb/        B&B core: solver, branching, lower bounds, dominance, search strategy
  ml/
    models/   BranchingTransformer (policy + value heads)
    il/       Featurization, trajectory generation, teacher policy
    rl/       BranchingEnv (PPO environment), policy guidance
scripts/
  log_teacher_traces.py   Generate optimal trajectories via CP-SAT
  train_bc.py             Train imitation learning policy
  train_ppo.py        PPO fine-tuning from BC checkpoint
  run_bnb.py              Run and evaluate the B&B solver
config/                   JSON configs for solver, BC training, PPO training
data/
  train/                  Training instances and CP-SAT traces (including Traj-Opt)
  eval/                   Held-out evaluation sets (J30, J60, J90, J120)
models/                   Saved checkpoints
```


## Quickstart

Generate teacher trajectories from training instances:

```bash
python scripts/log_teacher_traces.py --root data/train/1kNetRes --pattern "*.rcp" --output-dir data/train/Traj-Opt/1kNetRes
```

Train the imitation learning policy:

```bash
python scripts/train_bc.py --config config/train_bc.json
```

Fine-tune with PPO:

```bash
python scripts/train_ppo.py --config config/train_ppo.json
```

Run the solver with the learned policy:

```bash
python scripts/run_bnb.py --config config/run_bnb.json --root data/eval/intrain/J30_10
```

The supplied BC config refers to an older `data/train/trajectories/trainM`
layout; set its trajectory and instance directories to matching paths before
running it. The separate PPO config loads the existing `models/policy_bc.pt`
checkpoint and uses the data directories supplied in this checkout.

## Configuration

`config/run_bnb.json` — Solver settings: branching order, time limit, dominance rules, policy path.

`config/train_bc.json` — BC training: trajectory directory, model architecture, learning rate, epochs.

`config/train_ppo.json` — PPO with independent actor and critic transformers, 120-second training episodes, selective bootstrap at `critic_bootstrap_min_completion: 0.01`, and output under `models/cs/`.

The node-expansion cost uses the configured fixed `alpha` for every instance.
Critic returns use running mean/variance normalization, with the value head
rescaled when those statistics change to preserve its raw predictions. Actor
advantages are standardized over the valid rollout batch before PPO updates.

### Separate actor and critic experiment

Run the separate architecture with:

```bash
python scripts/train_ppo.py --config config/train_ppo.json --log
```

The PPO trainer on this branch always uses separate actor and critic networks.
Both transformer trunks start as independent copies of the shared PPO initialization:
the actor preserves the BC policy exactly, and the critic preserves the same
initial value predictions as the shared baseline, including the newly expanded
value head. Input projections, resource encoders, attention layers, normalization,
and output heads share no parameters. Value-loss gradients update only the
critic; policy and entropy gradients update only the actor. Gradient clipping is
also independent, so critic gradient magnitude cannot shrink the actor update.
The critic still supplies the baseline used to calculate policy advantages.

With the supplied BC checkpoint, the actor has 74,433 parameters and the critic
has 79,617, for 154,050 in total. Collection and updates run both transformers;
evaluation runs only the actor. The additional training computation can affect
how many search nodes fit into a time-limited training episode.

For a matched shared baseline, run the shared trainer from the baseline branch
with the same configuration and different `save_path` and `checkpoint_dir`
values. Keep the BC checkpoint, seed, training/evaluation data, rewards,
budgets, and PPO settings identical. The separate configuration uses
`data/train/1kNetRes` for training and `data/eval/intrain/J30_10` for evaluation.

Separate checkpoints retain both networks and the value-normalization state.
The existing `load_policy_checkpoint` and B&B runners load the deployment actor;
`load_separate_actor_critic_checkpoint` restores both networks for further PPO
training. As in the existing trainer, starting from a checkpoint initializes a
new optimizer and new training counters rather than restoring an exact training
session. The supplied configuration enables selective critic bootstrapping and
writes artifacts to `models/cs/`.

Run the architecture and PPO integration tests with:

```bash
PYTHONPATH=src python -m unittest discover -s tests -p 'test_separate_actor_critic.py'
PYTHONPATH=src python -m unittest discover -s tests -p 'test_single_action_ppo.py'
```

The test files are local, ignored artifacts in this checkout; run these commands
when that local test directory is available.

### PPO action and search order

PPO samples one feasible activity from the policy's categorical distribution.
During training, DFS explores that activity first and then the remaining ready
activities in ascending activity-ID order. The buffer stores only the selected
candidate index; PPO likelihood ratios and entropy refer to that single choice.
The tail is deterministic and independent of policy logits.

Evaluation continues to sort all ready activities by descending policy score,
matching the historical single-action training/evaluation setup. Subtree rewards,
value normalization, resource-set features, and stratified episode sampling use
the current implementations. Training artifacts are written under
`models/cs/` by `config/train_ppo.json`.

Run the action and replay regression tests with:

```bash
PYTHONPATH=src python -m unittest discover -s tests -p 'test_single_action_ppo.py'
```

### Selective critic bootstrapping

`config/train_ppo.json` sets
`critic_bootstrap_min_completion: 0.01`. At truncation, the trainer estimates the
remaining subtree return at pending nodes with at least this fraction of real
activities scheduled. Identifiable zero-duration, zero-demand global dummy
source/sink activities are excluded: for J30, the gate is the first of 30 real
tasks. Interior zero-duration activities still count. Set the option to `null`
for a comparison without bootstrapping; configs that omit it disable bootstrapping.

The trainer reconstructs frontier observations with the final incumbent,
frontier lower bound, feasible candidates, and node stagnation. It runs only
the critic, without gradients or action sampling, and converts normalized values
back to raw return units using the collection-time statistics. Boundary values
are fixed before subsampling and throughout the PPO update. A fresh BC critic
waits until the first PPO update; a restored PPO critic with value-normalization
metadata can supply estimates immediately. This avoids using an untrained value
head, but does not establish that one update is enough for accurate estimates.

Each frontier value contributes once to ancestor targets, alongside every
observed expansion cost and incumbent bonus. An ancestor becomes usable only
when every unfinished descendant is covered; an uncovered shallow sibling still
blocks it. Actual solver closure and the observed episode return are preserved.
Pending nodes supply boundary values, not new actor transitions. Cheap residuals
are resolved directly: bound-prunable and empty-ready nodes contribute zero,
and eligible zero-feasible states contribute `-alpha` conditional on
being expanded with the final incumbent. An intervening incumbent improvement
could instead cause bound pruning. Complete
pending schedules receive conditional incumbent bonuses in DFS frontier order,
so first-incumbent bonuses are not repeatedly counted among those schedules.

Both `tree_gamma_cost` and `tree_gamma_bonus` must be `1.0`, and
`tree_keep_open` must be `false`. The single critic predicts the combined cost
and bonus return, so it cannot independently discount unknown reward channels.

Console `[Bootstrap]` lines and TensorBoard `bootstrap/*` scalars report pending
and eligible frontier counts, exact zeros, expansion/terminal estimates, neural estimates,
uncovered nodes, recovered decisions by depth, their retention after sampling,
and the fraction of retained targets using frontier contributions. Estimated
values are not added to reported observed episode rewards.

Completion is a heuristic for value reliability. The existing subsampler still
drops forced-choice states from critic training, even though many eligible
frontier states are forced; `bootstrap/critic_forced` reports this exposure.
Final-context estimates also approximate future incumbent interactions between
frontier branches. Check depth-specific value accuracy and accumulated ancestor
error before interpreting training gains. Shallow-depth subsampling is applied
after these complete-tree targets have been calculated.

```bash
PYTHONPATH=src python -m unittest discover -s tests -p 'test_critic_bootstrap*.py'
```

### Shallow-depth rollout subsampling

The separate actor/critic configuration uses `shallow_depth_threshold: 0.5`
and `shallow_priority_per_cell: 128`. A shallow state has strictly fewer than
half of the real tasks scheduled when its action is taken. Identifiable dummy
source/sink activities do not count; in J30, 0 through 14 of 30 real tasks
scheduled is shallow. The completion cutoff transfers across instance sizes.

After the full-tree backup, discard invalid targets and states with at most
one feasible action. Let `K` be the smaller of the episode cap and the number
of remaining decisions. Selection is without replacement, in this order:

1. Include the union of the pre-first-incumbent prefix and the windows around
   incumbent improvements. If this union alone fills or exceeds `K`, randomly
   trim it to `K`, as before; no shallow additions are made.
2. Using the original time-by-depth grid, top up the protected shallow
   decisions in each occupied shallow cell to
   `min(shallow_available_in_cell, shallow_priority_per_cell)`. Existing shallow
   prefix/window decisions count toward that cell's allowance. Randomly draw
   any additional decisions within each cell, without replacement. If these
   additions exceed the remaining capacity, distribute that capacity across
   their cells with the same stratified sampler.
3. Spend the leftover capacity on all remaining decisions using the original
   stratified method. Three episode-progress bands and four relative-depth
   bands give up to 12 cells. Grid depth remains `solver_depth / n_activities`;
   real-task completion is used only for shallow eligibility. Small cells are
   retained in full and unused shares are redistributed. Random remainder
   allocation uses the entire cap when enough decisions exist.

There is no fixed shallow percentage. With six occupied shallow cells and a
limit of 128 per cell, at most 768 additional shallow decisions are protected,
fewer when event priorities already cover them or cells contain fewer choices.
The ordinary stratified stage can still select more shallow states. For an
illustrative episode with abundant decisions in all 12 cells, six shallow
cells, no event priorities, and cap 8192, the shallow additions use 768 slots.
The remaining 7424 slots split roughly equally between shallow and deeper
cells: about 4480/8192, or 54.7%, of the final sample is shallow.

Shallow additions use free capacity after event priorities. If they consume
that capacity, no slots remain for the ordinary fill. Uncapped episodes and
episodes that fit the cap keep every usable decision without extra sampling.
`shallow_priority_per_cell: 0` disables the additions; configurations that omit
the setting default to zero. Return targets are calculated before sampling;
the retained state distribution determines their weight in the PPO losses.

Console `[Subsample]` lines report the shallow count, sum of per-cell priority
targets, available count, actual retained fraction, and selected cell counts.
The priority target is not a guaranteed final shallow count when capacity is
exhausted by event priorities.
TensorBoard `subsample/shallow_*` and `subsample/cell/*` scalars record the
same distribution. `included`, `shallow_added`, and `stratified` form a
non-overlapping accounting of retained event priorities, additional shallow
priority choices, and general choices; their counts sum to `kept`.

```bash
PYTHONPATH=src python -m unittest discover -s tests -p 'test_shallow_subsampling.py'
```

### Evaluation during GPU PPO training

In `scripts/train_ppo.py`, set `eval_every_updates` to a positive integer
to evaluate and save a checkpoint after that many complete PPO updates. This
setting overrides `eval_every_steps`; omitting it or using `null` keeps
evaluation based on environment steps, checked after each PPO update.

`config/train_ppo.json` uses `eval_every_updates: 1` for the
120-second training episodes. Each evaluation still uses its own
`eval_time_limit_s` budget. Evaluation JSON records and checkpoint metadata
include both `step` and `update`; the best model is selected by mean gap.
The final evaluation is skipped when the latest policy was already evaluated,
including when training ends with a partially collected batch and no new update.

### Incumbent rewards in PPO

`scripts/train_ppo.py` scores an incumbent of makespan `C` using the fixed
root lower bound `L`. The first incumbent earns `beta1 * L / C`; each later
incumbent earns `beta2 * (previous_makespan - C) / previous_makespan`.
`beta1` and `beta2` are independent, finite, non-negative weights, and
`tree_gamma_bonus = 1.0` keeps these gains undiscounted. The improvement reward
depends on the chronology of incumbent updates and does not telescope. For
example, with `L = 80`, `beta1 = 1`, and `beta2 = 2`, a history of `100` earns
`0.8`; `160 -> 125 -> 100` earns
`0.5 + 2*(35/160) + 2*(25/125) = 1.3375`. The separate config uses
`beta1 = 1`, `beta2 = 5`, and unit cost/bonus discounts.
