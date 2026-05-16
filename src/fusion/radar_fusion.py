"""
Radar Fusion — 레이더 포인트 → BEV 래스터화 → BEV 쿼리 bias 가산.

레이더 포인트 (N, 5) [x, y, z, vx, vy] ego frame
  → BEV 그리드 래스터화 (B, 5, H, W)
  → Flatten (B, N_bev, 5) → Linear → (B, N_bev, C)
  → BEV 쿼리에 bias로 가산
"""

import torch
import torch.nn as nn


class RadarFusion(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        radar_ch  = 5                               # [x,y,z,vx,vy]
        bev_ch    = cfg["model"]["bev_channels"]    # 256
        self.H    = cfg["bev"]["h"]                 # 200
        self.W    = cfg["bev"]["w"]                 # 200
        self.x_min = cfg["bev"]["x_min"]            # -50.0
        self.y_min = cfg["bev"]["y_min"]            # -50.0
        self.cell  = cfg["bev"]["cell_size"]        # 0.5

        self.proj = nn.Linear(radar_ch, bev_ch)

    def rasterize(self, radar_pts_list: list, device: torch.device) -> torch.Tensor:
        """
        radar_pts_list: list of (N_i, 5) tensors (per batch)
        Returns: (B, 5, H, W)  — 셀당 평균값 (없는 셀은 0)
        """
        B = len(radar_pts_list)
        H, W = self.H, self.W
        bev = torch.zeros(B, 5, H, W, device=device)

        for b, pts in enumerate(radar_pts_list):
            if pts.shape[0] == 0:
                continue
            pts = pts.to(device)   # (N, 5)

            # ego 좌표 → BEV 그리드 인덱스
            col = ((pts[:, 0] - self.x_min) / self.cell).long()  # x → col
            row = ((pts[:, 1] - self.y_min) / self.cell).long()  # y → row

            # 범위 내 포인트만
            valid = (col >= 0) & (col < W) & (row >= 0) & (row < H)
            col, row, pts = col[valid], row[valid], pts[valid]

            if col.shape[0] == 0:
                continue

            # 셀당 합산 + 카운트 (mean 계산용)
            flat_idx = row * W + col   # (N,)
            for c in range(5):
                bev[b, c].reshape(-1).scatter_add_(0, flat_idx, pts[:, c])
            count = torch.zeros(H * W, device=device)
            count.scatter_add_(0, flat_idx, torch.ones(flat_idx.shape[0], device=device))
            count = count.clamp(min=1.0).reshape(H, W)
            bev[b] /= count.unsqueeze(0)   # (5, H, W) / (H, W) → mean

        return bev   # (B, 5, H, W)

    def forward(
        self,
        bev_queries:     torch.Tensor,   # (B, N_bev, C)
        radar_pts_list:  list,           # list of (N_i, 5)
    ) -> torch.Tensor:                   # (B, N_bev, C)

        device = bev_queries.device
        B      = bev_queries.shape[0]

        radar_bev = self.rasterize(radar_pts_list, device)          # (B, 5, H, W)
        radar_flat = radar_bev.permute(0, 2, 3, 1)                  # (B, H, W, 5)
        radar_flat = radar_flat.reshape(B, self.H * self.W, 5)      # (B, N_bev, 5)

        bias = self.proj(radar_flat)                                 # (B, N_bev, C)
        return bev_queries + bias
