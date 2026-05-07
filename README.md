# RCPSP Branch-and-Bound RL

A learned branching policy for the Resource-Constrained Project Scheduling Problem (RCPSP), trained to replace classic branching order methods inside a Branch-and-Bound (B&B) solver.

## Background & Approach

RCPSP is the problem of scheduling a set of activities with known durations, precedence constraints, and resource requirements to minimize project makespan (Blazewicz et al., 1983). We utlize the branch and bound procedure in solving this problem where we specifcially learn a branching policy to optimize search. A robust branching policy drastically reduces the search tree, enables aggressive pruning and faster convergence. We show that a machine learning policy outperforms classic branching order methods leading to better optimal search. 

We train a transformer-based branching policy in two stages: supervised imitation learning on optimal trajectories from OR-Tools CP-SAT, followed by PPO reinforcement learning to improve late-search navigation.

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
