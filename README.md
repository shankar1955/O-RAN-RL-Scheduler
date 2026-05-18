# O-RAN RL Scheduler
**Q-Learning-Based Dynamic Resource Allocation in O-RAN Small-Cell Networks**

**Author:** Shankar M, B.E. EEE, Chennai Institute of Technology
**Status:** 🔧 Work in Progress

---

## Overview
This project implements and benchmarks a Q-learning-based xApp for dynamic resource block (RB) allocation in an O-RAN small-cell deployment. The goal is to demonstrate that an AI-driven scheduling policy outperforms classical rule-based schedulers in spectral efficiency and inter-cell fairness.

The longer-term objective is to integrate this policy into a Near-RT RIC running on the O-RAN Software Community (OSC) stack, connected to a real gNB via the E2 interface.

---

## Architecture

```
┌─────────────────────────────────┐
│        Near-RT RIC (OSC)        │  ← Planned: Docker deployment
│   ┌─────────────────────────┐   │
│   │    Q-Learning xApp      │   │  ← This repo (policy logic)
│   └─────────────────────────┘   │
│            E2 Interface         │  ← Planned: srsRAN / OAI gNB
└─────────────────────────────────┘
         ↕
┌─────────────────────────────────┐
│     Simulated O-RAN Env         │  ← oran_env.py (current)
│  3 Cells | 10 UEs | 25 RBs     │
└─────────────────────────────────┘
```

---

## Project Structure
```
oran_rl_scheduler/
├── env/
│   └── oran_env.py          # Small-cell network simulation (3 cells, 10 UEs)
├── agent/
│   ├── q_agent.py           # Tabular Q-learning agent
│   └── baseline.py          # Rule-based baselines (RR, Max-CQI, PF)
├── utils/                   # Plotting and metrics (in progress)
├── results/                 # Training logs and Q-table checkpoints
└── train.py                 # Main training + evaluation script
```

---

## Current Progress
- [x] O-RAN small-cell environment (Rayleigh channel, Poisson traffic, Jain fairness reward)
- [x] Tabular Q-learning agent with ε-greedy exploration
- [x] Rule-based baselines: Round Robin, Max-CQI, Proportional Fair
- [x] Training loop with episode logging (SE, Jain index, reward)
- [ ] Results plotting and benchmark comparison figures
- [ ] DQN upgrade (neural function approximator for continuous state space)
- [ ] OSC Near-RT RIC Docker deployment
- [ ] E2 interface integration with srsRAN/OAI gNB
- [ ] xApp registration and policy push via E2SM-RC

---

## Planned Next Steps (pre-NTUST arrival)
1. Complete DQN upgrade replacing tabular Q-table
2. Deploy OSC Near-RT RIC stack locally (Docker)
3. Connect simulated E2 node to RIC using OSC E2 simulator
4. Migrate xApp policy into OSC xApp SDK framework

---

## Requirements
```
numpy
python >= 3.8
```

---

## Run
```bash
python train.py
```
