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
  train_ppo.py            PPO fine-tuning from BC checkpoint
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
python scripts/train_ppo.py --config config/train_ppo.json
```

Run the solver with the learned policy:

```bash
python scripts/run_bnb.py --config config/run_bnb.json
```

## Configuration

`config/run_bnb.json` — Solver settings: branching order, time limit, dominance rules, policy path.

`config/train_bc.json` — BC training: trajectory directory, model architecture, learning rate, epochs.

`config/train_ppo.json` — PPO training: BC checkpoint, reward coefficients, rollout and update settings.

### Incumbent rewards in PPO

`scripts/train_ppo_gpu.py` uses the fixed root lower bound `L` to score an
incumbent of makespan `C` as `Q(C) = L / C`. The first incumbent earns
`beta1 * Q(first)`; each later incumbent earns
`beta2 * (Q(new) - Q(previous))`. Training requires `beta1 == beta2 >= 0`
and `tree_gamma_bonus = 1.0`, so the cumulative incumbent reward equals
`beta1 * Q(final)` regardless of intermediate solutions. For example, with
`L = 80` and both weights equal to 1, both `100` and `160 -> 125 -> 100`
earn a total of `0.8`. Separate first/incumbent-improvement logs are retained.

Both `config/train_ppo_gpu.json` and `config/train_ppo_gpu_oracle.json` already
use these settings. Node-cost rewards retain their existing scaling and
discount. The new bonuses apply to subsequent PPO training; existing
checkpoints are not changed by updating the reward code.
