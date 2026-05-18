"""
O-RAN Small-Cell Network Environment
3 cells, 10 UEs, variable traffic loads
Author: Shankar M, Chennai Institute of Technology
"""

import numpy as np


NUM_CELLS = 3
NUM_UES = 10
NUM_RBS = 25          # Resource Blocks per cell (10 MHz LTE-like)
MAX_QUEUE = 100       # Max packets in UE buffer


class UE:
    def __init__(self, ue_id, cell_id):
        self.ue_id = ue_id
        self.cell_id = cell_id
        self.rsrp = np.random.uniform(-110, -70)   # dBm
        self.cqi = self._rsrp_to_cqi(self.rsrp)
        self.queue = np.random.randint(5, 30)       # packets pending
        self.traffic_type = np.random.choice(["video", "voip", "data"])

    def _rsrp_to_cqi(self, rsrp):
        # Simplified RSRP -> CQI mapping (CQI: 1-15)
        cqi = int((rsrp + 110) / 40 * 14) + 1
        return np.clip(cqi, 1, 15)

    def update_channel(self):
        # Rayleigh-like random walk for channel variation
        self.rsrp += np.random.normal(0, 2)
        self.rsrp = np.clip(self.rsrp, -120, -60)
        self.cqi = self._rsrp_to_cqi(self.rsrp)

    def update_queue(self, rbs_allocated):
        # Drain queue based on RBs allocated and CQI
        throughput = rbs_allocated * self.cqi * 0.1   # simplified Mbps
        self.queue = max(0, self.queue - int(throughput))
        # Arrival: Poisson traffic
        arrival = np.random.poisson(lam=3 if self.traffic_type == "video" else 1)
        self.queue = min(MAX_QUEUE, self.queue + arrival)


class ORANEnv:
    """
    Simulates a 3-cell O-RAN small-cell deployment.
    The scheduler (xApp) observes per-UE CQI and queue state,
    then allocates resource blocks across UEs in each cell.
    """

    def __init__(self):
        self.ues = [UE(i, i % NUM_CELLS) for i in range(NUM_UES)]
        self.timestep = 0
        self.state_dim = NUM_UES * 2    # [CQI, queue_norm] per UE
        self.action_dim = NUM_UES       # RB share per UE (discrete: 0,1,2,3)

    def reset(self):
        self.ues = [UE(i, i % NUM_CELLS) for i in range(NUM_UES)]
        self.timestep = 0
        return self._get_state()

    def _get_state(self):
        state = []
        for ue in self.ues:
            state.append(ue.cqi / 15.0)                  # normalised CQI
            state.append(ue.queue / MAX_QUEUE)            # normalised queue
        return np.array(state, dtype=np.float32)

    def step(self, action):
        """
        action: list of RB allocations per UE (integers, sum <= NUM_RBS per cell)
        Returns: next_state, reward, done, info
        """
        cell_rb_used = {c: 0 for c in range(NUM_CELLS)}
        throughputs = []

        for i, ue in enumerate(self.ues):
            rbs = int(action[i])
            cell_rb_used[ue.cell_id] += rbs

        # Penalise if any cell exceeds RB budget
        overload_penalty = 0
        for c, used in cell_rb_used.items():
            if used > NUM_RBS:
                overload_penalty += (used - NUM_RBS) * 0.5

        # Apply allocation and collect throughput
        for i, ue in enumerate(self.ues):
            rbs = int(action[i])
            tp = rbs * ue.cqi * 0.1
            throughputs.append(tp)
            ue.update_queue(rbs)
            ue.update_channel()

        # Spectral efficiency reward
        se_reward = np.sum(throughputs) / (NUM_CELLS * NUM_RBS)

        # Jain's Fairness Index
        tp_arr = np.array(throughputs) + 1e-6
        jain = (np.sum(tp_arr) ** 2) / (NUM_UES * np.sum(tp_arr ** 2))

        reward = se_reward + jain - overload_penalty

        self.timestep += 1
        done = self.timestep >= 200

        info = {
            "spectral_efficiency": se_reward,
            "jain_fairness": jain,
            "throughputs": throughputs,
            "cell_rb_used": cell_rb_used
        }

        return self._get_state(), reward, done, info

    def get_state_dim(self):
        return self.state_dim

    def get_action_dim(self):
        return self.action_dim
