"""
MOTEvaluator — nuscenes_pmbm_tracker.py의 MOTEvaluator + hungarian_match 재사용.
MOTA = 1 - (FP + FN + IDSW) / N_GT
MOTP = mean matched center distance [m]
"""

import numpy as np
from scipy.optimize import linear_sum_assignment


def hungarian_match(gt_pos, det_pos, gate):
    """gt_pos, det_pos: list of [x,y]"""
    if len(gt_pos) == 0 or len(det_pos) == 0:
        return [], list(range(len(gt_pos))), list(range(len(det_pos)))
    cost = np.linalg.norm(
        np.array(gt_pos)[:, None, :] - np.array(det_pos)[None, :, :], axis=2
    )
    ri, ci = linear_sum_assignment(cost)
    matched, matched_g, matched_d = [], set(), set()
    for r, c in zip(ri, ci):
        if cost[r, c] <= gate:
            matched.append((r, c))
            matched_g.add(r)
            matched_d.add(c)
    unmatched_g = [i for i in range(len(gt_pos))  if i not in matched_g]
    unmatched_d = [j for j in range(len(det_pos)) if j not in matched_d]
    return matched, unmatched_g, unmatched_d


class MOTEvaluator:
    def __init__(self, threshold: float = 2.0):
        self.thr       = threshold
        self.tp = self.fp = self.fn = self.idsw = 0
        self.sum_dist  = 0.0
        self.n_dist    = 0
        self.n_frames  = 0
        self._prev_match: dict = {}

    def update(self, det_pos, det_ids, gt_pos, gt_ids):
        self.n_frames += 1
        matched, unmatched_g, unmatched_d = hungarian_match(
            gt_pos, det_pos, self.thr
        )
        new_match = {}
        for gi, di in matched:
            new_match[gt_ids[gi]] = det_ids[di]
            d = np.linalg.norm(np.array(gt_pos[gi]) - np.array(det_pos[di]))
            self.sum_dist += d
            self.n_dist   += 1
        for gi, di in matched:
            gid = gt_ids[gi]
            if gid in self._prev_match and self._prev_match[gid] != det_ids[di]:
                self.idsw += 1
        self.tp  += len(matched)
        self.fn  += len(unmatched_g)
        self.fp  += len(unmatched_d)
        self._prev_match = new_match

    @property
    def mota(self) -> float:
        n_gt = max(self.tp + self.fn, 1)
        return 1.0 - (self.fp + self.fn + self.idsw) / n_gt

    @property
    def motp(self) -> float:
        return self.sum_dist / max(self.n_dist, 1)

    def summary(self) -> dict:
        n_gt    = max(self.tp + self.fn, 1)
        n_frame = max(self.n_frames, 1)
        return {
            "MOTA":      round(self.mota, 4),
            "MOTP":      round(self.motp, 4),
            "FP/frame":  round(self.fp / n_frame, 2),
            "FN/frame":  round(self.fn / n_frame, 2),
            "IDSW":      self.idsw,
            "TP":        self.tp,
            "FP":        self.fp,
            "FN":        self.fn,
        }
