
# AGENTS.md

## Project overview
This project explore training a branching policy for efficient search in the branch and bound algorithm for the Resource Consttraint Project Scheduling Problem (RCPSP). The methodology consist of a two-stage training pipeline, where we first train a model through supervised imitation learning and then initialize RL training from that behavorial model. Below is the general outline of the projects. 

config/ - Contains hyperparameters for model training
models/ - Contains policy checkpoints from training
results/ - Contains evaluated results
scripts/ - Contains preprocessing and training scripts
src/rcpsp_bnb_rl/bnb - Contain the implementation of the branch and bound algorithm
src/rcpsp_bnb_rl/ml/il - Contains the implementation of supervised imitation learning
src/rcpsp_bnb_rl/ml/rl - Contains the implementation of reinforcement learning
src/rcpsp_bnb_rl/ml/models - Defines the architectural network for the model. 
znotes/ - Contains general implementation notes and general ideas

## Before making changes
- Read the relevant code and existing tests.
- For complex changes, propose a plan first.
- When making implementation, you should implement according to the plan in coherent bits (never write full implementaion on very complex task unless otherwise instructed.)

## Coding standards
- Follow existing naming conventions.
- Prefer small, focused changes.
- Avoid introducing unnecessary dependencies.
- Avoid excessive and verbose comments (you are encourage to use brief comments to note key implementation detail)


## Safety
- Never commit credentials or secrets.
- Do not deploy or push changes without approval.

## Environment
- All project dependencies are contained in the environment `Documents/teak/env/bin/`. 
