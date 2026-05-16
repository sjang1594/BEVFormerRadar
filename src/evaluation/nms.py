"""
Detection decoding + Center-distance NMS.

decode_predictions():
    heatmap peak → 3D box (x, y, z, w, l, h, yaw, score, cls)

center_nms():
    BEV 중심거리 기반 NMS
"""

from __future__ import annotations
import math
import torch
import torch.nn.functional as F
import numpy as np


def _heatmap_peaks(heatmap: torch.Tensor, score_thr: float) -> torch.Tensor:
    """
    Local maxima extraction via max-pooling.
    heatmap: (C, H, W)
    Returns: (N, 3) — [cls, row, col]
    """
    # 3×3 max-pool로 local max 마스크
    hm = heatmap.unsqueeze(0)                            # (1,C,H,W)
    hm_pool = F.max_pool2d(hm, kernel_size=3, stride=1, padding=1)
    peak_mask = (hm == hm_pool) & (hm > score_thr)      # (1,C,H,W)
    peak_mask = peak_mask.squeeze(0)                     # (C,H,W)
    return peak_mask.nonzero(as_tuple=False)             # (N,3) [cls,row,col]


def decode_predictions(
    preds:      dict,
    cfg:        dict,
    score_thr:  float | None = None,
) -> list[dict]:
    """
    preds: DetectionHead 출력 (B, ...)
    Returns: list of B dicts, each with keys:
        boxes   (M, 7) [x,y,z,w,l,h,yaw]
        scores  (M,)
        labels  (M,)
    """
    if score_thr is None:
        score_thr = cfg["eval"]["score_threshold"]

    x_min    = cfg["bev"]["x_min"]
    y_min    = cfg["bev"]["y_min"]
    cell     = cfg["bev"]["cell_size"]
    B        = preds["heatmap"].shape[0]
    results  = []

    for b in range(B):
        hm  = preds["heatmap"][b].detach().cpu()   # (C,H,W)
        off = preds["offset"][b].detach().cpu()    # (2,H,W)
        sz  = preds["size"][b].detach().cpu()      # (3,H,W)
        yaw = preds["yaw"][b].detach().cpu()       # (2,H,W)

        peaks = _heatmap_peaks(hm, score_thr)      # (N,3)
        if len(peaks) == 0:
            results.append({
                "boxes":  torch.zeros((0, 7)),
                "scores": torch.zeros((0,)),
                "labels": torch.zeros((0,), dtype=torch.long),
            })
            continue

        cls_idx = peaks[:, 0]
        rows    = peaks[:, 1]
        cols    = peaks[:, 2]
        scores  = hm[cls_idx, rows, cols]

        # 위치 디코딩
        x = x_min + (cols.float() + off[0, rows, cols]) * cell
        y = y_min + (rows.float() + off[1, rows, cols]) * cell
        z = torch.zeros_like(x)

        w = sz[0, rows, cols]
        l = sz[1, rows, cols]
        h = sz[2, rows, cols]
        yaw_rad = torch.atan2(yaw[0, rows, cols], yaw[1, rows, cols])

        boxes = torch.stack([x, y, z, w, l, h, yaw_rad], dim=-1)  # (N,7)
        results.append({"boxes": boxes, "scores": scores, "labels": cls_idx})

    return results


def center_nms(boxes: np.ndarray, scores: np.ndarray,
               nms_radius: float) -> np.ndarray:
    """
    BEV 중심거리 기반 NMS.
    boxes:  (N, 7) [x,y,z,w,l,h,yaw]
    scores: (N,)
    Returns: keep indices (M,)
    """
    if len(boxes) == 0:
        return np.array([], dtype=np.int64)

    order = scores.argsort()[::-1]
    keep  = []
    suppressed = np.zeros(len(boxes), dtype=bool)

    for i in order:
        if suppressed[i]:
            continue
        keep.append(i)
        cx, cy = boxes[i, 0], boxes[i, 1]
        dist = np.sqrt((boxes[:, 0] - cx) ** 2 + (boxes[:, 1] - cy) ** 2)
        suppressed |= (dist < nms_radius)
        suppressed[i] = False  # 자기 자신은 유지

    return np.array(keep, dtype=np.int64)
