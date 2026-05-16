"""
Detection Loss — Focal Loss (heatmap) + L1 (offset / size / yaw).

L = L_focal(α=2,β=4) + 0.25×L_offset + 0.25×L_size + 0.25×L_yaw
"""

from __future__ import annotations
import math
import torch
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# GT map 생성
# ─────────────────────────────────────────────────────────────────────────────

def build_gt_maps(
    gt_boxes:   torch.Tensor,   # (M, 7) [x,y,z,w,l,h,yaw] ego
    gt_labels:  torch.Tensor,   # (M,)
    cfg:        dict,
) -> tuple:
    """
    Returns:
        heatmap : (num_cls, H, W)  float32  Gaussian GT
        offset  : (2, H, W)        float32  sub-cell offset at center
        size    : (3, H, W)        float32  w,l,h at center
        yaw_map : (2, H, W)        float32  sin,cos at center
        center_mask: (H, W)        bool     True at GT center cells
    """
    num_cls      = cfg["model"]["num_classes"]
    H, W         = cfg["bev"]["h"], cfg["bev"]["w"]
    x_min        = cfg["bev"]["x_min"]               # -50.0
    y_min        = cfg["bev"]["y_min"]               # -50.0
    cell         = cfg["bev"]["cell_size"]           # 0.5
    sigma_factor = cfg["model"]["heatmap_sigma_factor"]  # 2.0

    heatmap     = torch.zeros(num_cls, H, W, dtype=torch.float32)
    offset_map  = torch.zeros(2, H, W, dtype=torch.float32)
    size_map    = torch.zeros(3, H, W, dtype=torch.float32)
    yaw_map     = torch.zeros(2, H, W, dtype=torch.float32)
    center_mask = torch.zeros(H, W, dtype=torch.bool)

    if len(gt_boxes) == 0:
        return heatmap, offset_map, size_map, yaw_map, center_mask

    # 그리드 좌표축 (Gaussian 계산용)
    ys = torch.arange(H, dtype=torch.float32)
    xs = torch.arange(W, dtype=torch.float32)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")  # (H, W)

    for i in range(len(gt_boxes)):
        x, y, z, w, l, h, yaw = gt_boxes[i].tolist()
        cls = int(gt_labels[i].item())

        # ego 좌표 → BEV 그리드 좌표 (float)
        cx = (x - x_min) / cell   # column
        cy = (y - y_min) / cell   # row

        # BEV 범위 밖이면 스킵
        if not (0 <= cx < W and 0 <= cy < H):
            continue

        # Gaussian sigma (grid cell 단위)
        w_cells = w / cell
        l_cells = l / cell
        sigma   = max(max(w_cells, l_cells) / sigma_factor, 1.0)

        # Gaussian 렌더링
        dist2 = (grid_y - cy) ** 2 + (grid_x - cx) ** 2
        gauss = torch.exp(-dist2 / (2 * sigma ** 2))
        heatmap[cls] = torch.maximum(heatmap[cls], gauss)

        # 중심 셀 (정수)
        ri, ci = int(cy), int(cx)
        ri = min(ri, H - 1)
        ci = min(ci, W - 1)

        # sub-pixel offset으로 인해 gauss < 1.0이 되는 경우를 방지 (pos_mask 정확도)
        heatmap[cls, ri, ci] = 1.0

        center_mask[ri, ci] = True
        offset_map[0, ri, ci] = cx - ci    # x sub-cell
        offset_map[1, ri, ci] = cy - ri    # y sub-cell
        size_map[0, ri, ci]   = w
        size_map[1, ri, ci]   = l
        size_map[2, ri, ci]   = h
        yaw_map[0, ri, ci]    = math.sin(yaw)
        yaw_map[1, ri, ci]    = math.cos(yaw)

    return heatmap, offset_map, size_map, yaw_map, center_mask


# ─────────────────────────────────────────────────────────────────────────────
# Loss 함수
# ─────────────────────────────────────────────────────────────────────────────

def focal_loss(pred: torch.Tensor, gt: torch.Tensor,
               alpha: float = 2.0, beta: float = 4.0) -> torch.Tensor:
    """
    CornerNet Focal Loss.
    pred: (B, C, H, W) — sigmoid 적용 후
    gt:   (B, C, H, W) — Gaussian GT heatmap [0,1]
    """
    pos_mask = (gt == 1.0).float()
    neg_mask = (gt < 1.0).float()

    pos_loss = pos_mask * (1 - pred).pow(alpha) * torch.log(pred.clamp(min=1e-6))
    neg_loss = neg_mask * (1 - gt).pow(beta) * pred.pow(alpha) * torch.log((1 - pred).clamp(min=1e-6))

    n_pos = pos_mask.sum().clamp(min=1.0)
    return -(pos_loss.sum() + neg_loss.sum()) / n_pos


def detection_loss(preds: dict, batch: dict, cfg: dict) -> dict:
    """
    Args:
        preds:  DetectionHead 출력 dict (heatmap, offset, size, yaw)
        batch:  DataLoader 배치 dict
        cfg:    config.yaml dict
    Returns:
        losses dict: total, heatmap, offset, size, yaw
    """
    device = preds["heatmap"].device
    B      = preds["heatmap"].shape[0]

    w_off  = cfg["train"]["loss_offset_weight"]
    w_size = cfg["train"]["loss_size_weight"]
    w_yaw  = cfg["train"]["loss_yaw_weight"]

    total_focal  = torch.tensor(0.0, device=device)
    total_offset = torch.tensor(0.0, device=device)
    total_size   = torch.tensor(0.0, device=device)
    total_yaw    = torch.tensor(0.0, device=device)
    n_valid = 0

    for b in range(B):
        gt_boxes  = batch["gt_boxes"][b].to(device)   # (M, 7)
        gt_labels = batch["gt_labels"][b].to(device)  # (M,)

        hm_gt, off_gt, sz_gt, yaw_gt, mask = build_gt_maps(gt_boxes, gt_labels, cfg)
        hm_gt  = hm_gt.to(device)
        off_gt = off_gt.to(device)
        sz_gt  = sz_gt.to(device)
        yaw_gt = yaw_gt.to(device)
        mask   = mask.to(device)   # (H, W) bool

        # Focal loss (heatmap)
        total_focal += focal_loss(preds["heatmap"][b].unsqueeze(0),
                                   hm_gt.unsqueeze(0))

        if mask.any():
            n_valid += 1
            # L1 losses at GT center cells
            m = mask.unsqueeze(0)  # (1, H, W)

            total_offset += F.l1_loss(preds["offset"][b][m.expand(2,-1,-1)],
                                       off_gt[m.expand(2,-1,-1)])
            total_size   += F.l1_loss(preds["size"][b][m.expand(3,-1,-1)],
                                       sz_gt[m.expand(3,-1,-1)])
            total_yaw    += F.l1_loss(preds["yaw"][b][m.expand(2,-1,-1)],
                                       yaw_gt[m.expand(2,-1,-1)])

    n_valid = max(n_valid, 1)
    l_offset = total_offset / n_valid
    l_size   = total_size   / n_valid
    l_yaw    = total_yaw    / n_valid
    l_focal  = total_focal  / B

    total = l_focal + w_off * l_offset + w_size * l_size + w_yaw * l_yaw

    return {
        "total":   total,
        "heatmap": l_focal,
        "offset":  l_offset,
        "size":    l_size,
        "yaw":     l_yaw,
    }
