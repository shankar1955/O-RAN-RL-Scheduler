"""
Training Script — Q-Learning xApp for O-RAN Resource Allocation
Author: Shankar M, Chennai Institute of Technology

Status: Work in Progress
- [x] Environment simulation (oran_env.py)
- [x] Tabular Q-learning agent (q_agent.py)
- [x] Rule-based baselines (baseline.py)
- [x] Training loop with logging
- [ ] DQN upgrade (planned — neural function approximator)
- [ ] OSC Near-RT RIC E2 interface integration (planned)
- [ ] Real gNB testbed connection via srsRAN (planned)
"""

import numpy as np
import os
import json
from env.oran_env import ORANEnv
from agent.q_agent import QLearningAgent
from agent.baseline import RoundRobinScheduler, ProportionalFairScheduler

os.makedirs("results", exist_ok=True)

# ── Hyperparameters ──────────────────────────────────────────────
EPISODES       = 500
LOG_INTERVAL   = 50

# ── Init ─────────────────────────────────────────────────────────
env   = ORANEnv()
agent = QLearningAgent()
rr    = RoundRobinScheduler()
pf    = ProportionalFairScheduler()

log = {"q_learning": [], "round_robin": [], "prop_fair": []}


def run_baseline(scheduler, label, episodes=100):
    rewards, se_list, jain_list = [], [], []
    for ep in range(episodes):
        state = env.reset()
        ep_reward, ep_se, ep_jain = 0, 0, 0
        done = False
        while not done:
            action = scheduler.allocate(env.ues)
            state, reward, done, info = env.step(action)
            ep_reward += reward
            ep_se     += info["spectral_efficiency"]
            ep_jain   += info["jain_fairness"]
        rewards.append(ep_reward)
        se_list.append(ep_se / 200)
        jain_list.append(ep_jain / 200)
    print(f"[{label}] Avg Reward: {np.mean(rewards):.3f} | "
          f"SE: {np.mean(se_list):.4f} | Jain: {np.mean(jain_list):.4f}")
    return np.mean(rewards), np.mean(se_list), np.mean(jain_list)


# ── Train Q-Learning Agent ────────────────────────────────────────
print("=" * 55)
print("  O-RAN RL Scheduler — Training (Shankar M, CIT)")
print("=" * 55)

for episode in range(1, EPISODES + 1):
    state = env.reset()
    state_idx = agent.discretise_state(state)
    ep_reward, ep_se, ep_jain = 0, 0, 0
    done = False

    while not done:
        action_idx = agent.select_action(state_idx)
        allocation = agent.action_to_rb_allocation(action_idx)

        next_state, reward, done, info = env.step(allocation)
        next_state_idx = agent.discretise_state(next_state)

        agent.update(state_idx, action_idx, reward, next_state_idx, done)

        state_idx  = next_state_idx
        ep_reward += reward
        ep_se     += info["spectral_efficiency"]
        ep_jain   += info["jain_fairness"]

    agent.decay_epsilon()
    log["q_learning"].append({
        "episode": episode,
        "reward": round(ep_reward, 4),
        "se":     round(ep_se / 200, 4),
        "jain":   round(ep_jain / 200, 4),
        "epsilon": round(agent.epsilon, 4)
    })

    if episode % LOG_INTERVAL == 0:
        recent = log["q_learning"][-LOG_INTERVAL:]
        avg_r  = np.mean([e["reward"] for e in recent])
        avg_se = np.mean([e["se"]     for e in recent])
        avg_jf = np.mean([e["jain"]   for e in recent])
        print(f"Ep {episode:>4} | Reward: {avg_r:7.3f} | "
              f"SE: {avg_se:.4f} | Jain: {avg_jf:.4f} | ε: {agent.epsilon:.3f}")

agent.save("results/q_table.pkl")

# ── Baselines ─────────────────────────────────────────────────────
print("\n── Baseline Evaluation ──")
run_baseline(rr, "Round Robin",        episodes=100)
run_baseline(pf, "Proportional Fair",  episodes=100)

# ── Save logs ─────────────────────────────────────────────────────
with open("results/training_log.json", "w") as f:
    json.dump(log, f, indent=2)
print("\n[Done] Logs saved to results/training_log.json")
