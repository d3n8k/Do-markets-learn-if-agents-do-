# Do Markets Learn if Agents Do?
Simulating endogenous market efficiency through multi-agent reinforcement learning (MARL) in non-stationary environments

## Overview
This repository contains the codebase for an independent research project exploring how multi-agent Q-learning interactions affect price impact and rolling volatility in simulated markets. A core focus of this work is resolving policy entropy collapse in baseline learning models by introducing dynamic hyperparameter adjustments and realistic market frictions.

## Repository Structure

`market_simulation.py`: The baseline Phase 1 & 2 market simulation featuring a simplified 6-state discretization and static learning rates.
`robustness_note.py`: The advanced v2 model acting as a robustness check. It implements an annealed learning rate ($\eta_t$), turnover-based trading costs, and a 27-state inventory-aware space based on volatility terciles.

## Execution
The scripts are fully self-contained and require standard Python scientific libraries (`numpy`, `matplotlib`, `scipy`). 

To run the robustness experiments and generate all associated figures (Fig 1-3) and tables:
```bash
python robustness_note.py
