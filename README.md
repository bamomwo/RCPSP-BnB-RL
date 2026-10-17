# RCPSP Branch-and-Bound RL

A learned branching policy for the Resource-Constrained Project Scheduling Problem (RCPSP), trained to replace classic branching order methods inside a Branch-and-Bound (B&B) solver.

## Background & Approach

RCPSP is the problem of scheduling a set of activities with known durations, precedence constraints, and resource requirements to minimize project makespan (Blazewicz et al., 1983). We utlize the branch and bound procedure in solving this problem where we specifcially learn a branching policy to optimize search. A robust branching policy drastically reduces the search tree, enables aggressive pruning and faster convergence. We show that a machine learning policy outperforms classic branching order methods leading to better optimal search. 

We train a transformer-based branching policy in two stages: supervised imitation learning on optimal trajectories from OR-Tools CP-SAT, followed by PPO reinforcement learning to improve efficient search navigation.

## Layout

```
src/rcpsp_bb_rl/
  bnb/        B&B core: solver, branching, lower bounds, dominance, search strategy
  ml/
    models/   BranchingTransformer (policy + value heads)
    il/       Featurization, trajectory generation, teacher policy
    rl/       BranchingEnv (PPO environment), policy guidance
scripts/
  log_teacher_traces.py   Generate optimal trajectories via CP-SAT
  train_bc.py             Train imitation learning policy
  train_ppo_gpu.py        PPO fine-tuning from BC checkpoint
  run_bnb.py              Run and evaluate the B&B solver
config/                   JSON configs for solver, BC training, PPO training
data/
  train/                  Training instances (1kNetRes, keepopt)
  eval/                   Held-out evaluation sets (J30, J60, J90, J120)
  trajectories/           CP-SAT teacher traces (JSONL)
models/                   Saved checkpoints
```

## Dependencies

Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install torch numpy ortools
```

## Quickstart

Generate teacher trajectories from training instances:

```bash
python scripts/log_teacher_traces.py --root data/train/1kNetRes --pattern "*.rcp" --output-dir data/trajectories/1kNetRes
```

Train the imitation learning policy:

```bash
python scripts/train_bc.py --config config/train_bc.json
```

Fine-tune with PPO:

```bash
python scripts/train_ppo_gpu.py --config config/train_ppo_gpu_single.json
```

Run the solver with the learned policy:

```bash
python scripts/run_bnb.py --config config/run_bnb.json
```

## Configuration

`config/run_bnb.json` — Solver settings: branching order, time limit, dominance rules, policy path.

`config/train_bc.json` — BC training: trajectory directory, model architecture, learning rate, epochs.

`config/train_ppo_gpu_single.json` — PPO training: categorical single-action PPO with the current reward, rollout, and update improvements.
`config/train_ppo_gpu_oracle.json` — existing oracle configuration, with its original output paths. The trainer on this branch uses single-action PPO with either config.
`config/train_ppo_gpu_separate_actor_critic.json` — PPO with independent actor and critic transformers, using the oracle training hyperparameters and the dataset paths present in this checkout.

### Separate actor and critic experiment

Run the separate architecture with:

```bash
python scripts/train_ppo_gpu.py --config config/train_ppo_gpu_separate_actor_critic.json --log
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
`data/oracle/oracle_beta2_5.jsonl` and `data/eval/intrain/J30_10`; older configs
may need their dataset paths updated before running the comparison.

Separate checkpoints retain both networks and the value-normalization state.
The existing `load_policy_checkpoint` and B&B runners load the deployment actor;
`load_separate_actor_critic_checkpoint` restores both networks for further PPO
training. As in the existing trainer, starting from a checkpoint initializes a
new optimizer and new training counters rather than restoring an exact training
session. Artifacts go to `models/train-separate-actor-critic/`.

Run the architecture and PPO integration tests with:

```bash
python -m unittest discover -s tests -p 'test_separate_actor_critic.py'
python -m unittest discover -s tests -p 'test_single_action_ppo.py'
```

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
`models/train-single-action-ppo/` by the supplied config.

Run the action and replay regression tests with:

```bash
python -m unittest discover -s tests -p 'test_single_action_ppo.py'
```

### Evaluation during GPU PPO training

In `scripts/train_ppo_gpu.py`, set `eval_every_updates` to a positive integer
to evaluate and save a checkpoint after that many complete PPO updates. This
setting overrides `eval_every_steps`; omitting it or using `null` keeps
evaluation based on environment steps, checked after each PPO update.

`config/train_ppo_gpu_oracle.json` uses `eval_every_updates: 1` for the
600-second training episodes. Each evaluation still uses its own
`eval_time_limit_s` budget. Evaluation JSON records and checkpoint metadata
include both `step` and `update`; the best model is selected by mean gap.
The final evaluation is skipped when the latest policy was already evaluated,
including when training ends with a partially collected batch and no new update.

### Incumbent rewards in PPO

`scripts/train_ppo_gpu.py` uses the fixed root lower bound `L` to score an
incumbent of makespan `C` as `Q(C) = L / C`. The first incumbent earns
`beta1 * Q(first)`; each later incumbent earns
`beta2 * (Q(new) - Q(previous))`. `beta1` and `beta2` are independent,
finite, non-negative weights. `tree_gamma_bonus = 1.0` keeps quality gains
undiscounted, and intermediate incumbents cannot inflate the improvement
channel: its cumulative value is `beta2 * (Q(final) - Q(first))`. For example,
with `L = 80`, `beta1 = 1`, and `beta2 = 2`, the histories `100` and
`160 -> 125 -> 100` earn total incumbent rewards `0.8` and `0.8 + 2*(0.8 -
0.5) = 1.1`, respectively. This lets training choose between first-incumbent
quality and later refinement according to the configured objective. Separate
first/incumbent-improvement logs are retained.

Both `config/train_ppo_gpu.json` and `config/train_ppo_gpu_oracle.json` already
use these settings. Node-cost rewards retain their existing scaling and
discount. The new bonuses apply to subsequent PPO training; existing
checkpoints are not changed by updating the reward code.
