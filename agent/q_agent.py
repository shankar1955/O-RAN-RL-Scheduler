"""
Q-Learning Agent for O-RAN Resource Block Scheduling
Discretised state-action space for tractable tabular Q-learning
Author: Shankar M, Chennai Institute of Technology
"""

import numpy as np
import pickle


class QLearningAgent:
    """
    Tabular Q-learning agent.
    State: discretised [avg_CQI_bin, avg_queue_bin] per cell (3x3x3x3 = 81 states)
    Action: per-cell allocation strategy index (3 strategies x 3 cells)

    NOTE: Full per-UE deep RL (DQN) is the planned next step.
    This tabular version validates the reward structure and convergence
    before moving to a neural function approximator.
    """

    def __init__(self, n_states=243, n_actions=27,
                 alpha=0.1, gamma=0.95,
                 epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.05):

        self.n_states = n_states
        self.n_actions = n_actions
        self.alpha = alpha           # learning rate
        self.gamma = gamma           # discount factor
        self.epsilon = epsilon       # exploration rate
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

        # Q-table: rows = states, cols = actions
        self.q_table = np.zeros((n_states, n_actions))

        self.episode_rewards = []
        self.episode_fairness = []
        self.episode_se = []

    def discretise_state(self, state):
        """
        Compress the 20-dim continuous state into a single integer index.
        Per-cell: avg CQI bin (0-2) and avg queue bin (0-2) → 3^6 = 729
        Simplified to 3^5 = 243 for stability.
        """
        cqi_vals = state[0::2]    # even indices = CQI
        q_vals = state[1::2]      # odd indices = queue

        # 3 cells → average CQI and queue per cell
        cell_states = []
        for c in range(3):
            ue_indices = [i for i in range(10) if i % 3 == c]
            avg_cqi = np.mean([cqi_vals[i] for i in ue_indices])
            avg_q = np.mean([q_vals[i] for i in ue_indices])
            cqi_bin = int(np.clip(avg_cqi * 3, 0, 2))
            q_bin = int(np.clip(avg_q * 3, 0, 2))
            cell_states.extend([cqi_bin, q_bin])

        # Encode to single integer (base-3)
        idx = 0
        for i, v in enumerate(cell_states[:5]):
            idx += v * (3 ** i)
        return min(idx, self.n_states - 1)

    def action_to_rb_allocation(self, action_idx):
        """
        Maps action index to per-UE RB allocation vector.
        3 strategies per cell:
          0 = proportional fair (PF)
          1 = max-CQI greedy
          2 = round-robin equal share
        """
        strategies = []
        tmp = action_idx
        for _ in range(3):
            strategies.append(tmp % 3)
            tmp //= 3

        # TODO: implement full per-UE RB allocation from strategy
        # For now returns a flat equal split as placeholder
        allocation = [2] * 10   # 2 RBs per UE as base
        return allocation

    def select_action(self, state_idx):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.n_actions)
        return np.argmax(self.q_table[state_idx])

    def update(self, state_idx, action, reward, next_state_idx, done):
        target = reward
        if not done:
            target += self.gamma * np.max(self.q_table[next_state_idx])
        td_error = target - self.q_table[state_idx, action]
        self.q_table[state_idx, action] += self.alpha * td_error

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save(self, path="results/q_table.pkl"):
        with open(path, "wb") as f:
            pickle.dump(self.q_table, f)
        print(f"[Agent] Q-table saved to {path}")

    def load(self, path="results/q_table.pkl"):
        with open(path, "rb") as f:
            self.q_table = pickle.load(f)
        print(f"[Agent] Q-table loaded from {path}")
