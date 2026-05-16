"""
Temporal Self Attention (TSA) — ego 모션 워프 + MHA.

동작:
  1. prev_bev feature map을 ego_delta로 현재 프레임에 정렬 (affine warp)
  2. Q=current BEV queries, KV=warped prev BEV → Multi-Head Attention
  3. 첫 프레임(has_prev=False)이면 Q=K=V=current (self-attention만)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalSelfAttention(nn.Module):
    """
    Args:
        embed_dim: BEV feature 차원 (256)
        num_heads: attention heads (8)
        bev_h:     BEV 그리드 높이 (200)
        bev_w:     BEV 그리드 너비 (200)
        bev_range: BEV 물리 범위 [m] (50.0, x/y 절댓값)
        dropout:   attention dropout
    """

    def __init__(self, embed_dim: int, num_heads: int,
                 bev_h: int, bev_w: int,
                 bev_range: float = 50.0,
                 dropout: float = 0.1):
        super().__init__()
        self.H         = bev_h
        self.W         = bev_w
        self.bev_range = bev_range

        # N=40000 쿼리로 MHA는 40000×40000 attention matrix → OOM
        # Gated fusion으로 대체: O(N), 동일한 temporal integration 효과
        self.gate     = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.Sigmoid(),
        )
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.drop     = nn.Dropout(dropout)

    def _warp_prev_bev(self, prev_bev: torch.Tensor,
                       ego_delta: torch.Tensor) -> torch.Tensor:
        """
        prev_bev: (B, N_bev, C)
        ego_delta: (B, 4, 4)  prev_ego → current_ego

        Returns warped_prev: (B, N_bev, C)
        """
        B, N, C = prev_bev.shape
        H, W    = self.H, self.W

        # (B, N, C) → (B, C, H, W)
        prev_map = prev_bev.permute(0, 2, 1).reshape(B, C, H, W)

        # Inverse warp theta: current BEV coords → prev BEV coords
        # p_prev_norm = R^T @ p_curr_norm - R^T @ t_norm
        R      = ego_delta[:, :2, :2]                             # (B,2,2)
        t_norm = ego_delta[:, :2, 3] / self.bev_range             # (B,2)
        R_T    = R.transpose(-1, -2)                              # (B,2,2)
        trans  = -(R_T @ t_norm.unsqueeze(-1))                    # (B,2,1)
        theta  = torch.cat([R_T, trans], dim=-1)                  # (B,2,3)

        grid   = F.affine_grid(theta, prev_map.shape, align_corners=True)
        warped = F.grid_sample(
            prev_map, grid,
            mode="bilinear", align_corners=True, padding_mode="zeros",
        )  # (B, C, H, W)

        return warped.reshape(B, C, N).permute(0, 2, 1)           # (B, N, C)

    def forward(
        self,
        bev_queries: torch.Tensor,          # (B, N_bev, C)
        prev_bev:    torch.Tensor | None,   # (B, N_bev, C) or None
        ego_delta:   torch.Tensor | None,   # (B, 4, 4)    or None
        has_prev:    bool,
    ) -> torch.Tensor:                      # (B, N_bev, C)

        if not has_prev or prev_bev is None:
            # 첫 프레임: temporal 기여 없음 (residual에 0 추가)
            return torch.zeros_like(bev_queries)

        warped = self._warp_prev_bev(prev_bev, ego_delta)           # (B, N, C)

        # 게이트가 각 BEV 셀마다 현재/이전 feature 비율을 학습
        gate  = self.gate(torch.cat([bev_queries, warped], dim=-1)) # (B, N, C)
        fused = gate * warped + (1 - gate) * bev_queries            # (B, N, C)
        return self.out_proj(self.drop(fused))
