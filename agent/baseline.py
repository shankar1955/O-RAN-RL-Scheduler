"""
Rule-Based Baseline Scheduler
Used as benchmark comparison against Q-learning agent.
Author: Shankar M, Chennai Institute of Technology
"""

import numpy as np

NUM_RBS = 25
NUM_UES = 10


class RoundRobinScheduler:
    """Equal RB distribution — no channel awareness."""
    def allocate(self, ues):
        rbs_each = NUM_RBS // NUM_UES
        return [rbs_each] * NUM_UES


class MaxCQIScheduler:
    """Greedy: allocate all RBs to highest CQI UE per cell."""
    def allocate(self, ues):
        allocation = [0] * NUM_UES
        cells = {}
        for i, ue in enumerate(ues):
            cells.setdefault(ue.cell_id, []).append((i, ue.cqi))

        for cell_id, ue_list in cells.items():
            ue_list.sort(key=lambda x: x[1], reverse=True)
            # Give all RBs to top CQI UE in cell
            allocation[ue_list[0][0]] = NUM_RBS

        return allocation


class ProportionalFairScheduler:
    """
    Proportional Fair: balances instantaneous CQI against
    historical average throughput.
    """
    def __init__(self):
        self.avg_tp = {i: 1.0 for i in range(NUM_UES)}
        self.alpha = 0.1   # moving average weight

    def allocate(self, ues):
        allocation = [0] * NUM_UES
        cells = {}
        for i, ue in enumerate(ues):
            cells.setdefault(ue.cell_id, []).append((i, ue))

        for cell_id, ue_list in cells.items():
            # PF metric: CQI / avg_throughput
            pf_scores = [(idx, ue.cqi / self.avg_tp[ue.ue_id])
                         for idx, ue in ue_list]
            pf_scores.sort(key=lambda x: x[1], reverse=True)

            # Distribute RBs proportionally
            total_score = sum(s for _, s in pf_scores)
            for idx, score in pf_scores:
                rbs = int((score / total_score) * NUM_RBS)
                allocation[idx] = max(1, rbs)

        # Update moving average
        for i, ue in enumerate(ues):
            tp = allocation[i] * ue.cqi * 0.1
            self.avg_tp[ue.ue_id] = (1 - self.alpha) * self.avg_tp[ue.ue_id] + self.alpha * tp

        return allocation
