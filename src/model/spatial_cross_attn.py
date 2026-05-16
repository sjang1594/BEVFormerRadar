"""
Spatial Cross Attention (SCA) — BEV 쿼리 × 카메라 이미지 feature.

동작 순서:
  1. BEV 쿼리 레퍼런스 포인트를 6개 카메라에 투영 (project_bev_to_image)
  2. P4 feature map에서 grid_sample로 이미지 feature 추출
  3. 유효 카메라에 대해 dot-product attention → 가중 합산
  4. Linear projection으로 BEV 쿼리 업데이트

메모리 최적화: N_bev=40000 쿼리를 chunk_size=5000 단위로 처리
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.geometry.projections import project_bev_to_image


class SpatialCrossAttention(nn.Module):
    """
    Args:
        embed_dim:  BEV feature 차원 (256)
        num_heads:  attention heads (사용하지 않음, 확장 예약)
        chunk_size: 청크 크기 (메모리 최적화)
        dropout:    attention dropout
    """

    def __init__(self, embed_dim: int, num_heads: int = 8,
                 chunk_size: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.embed_dim  = embed_dim
        self.chunk_size = chunk_size
        self.scale      = embed_dim ** -0.5

        self.q_proj   = nn.Linear(embed_dim, embed_dim)
        self.k_proj   = nn.Linear(embed_dim, embed_dim)
        self.v_proj   = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout  = nn.Dropout(dropout)

    def forward(
        self,
        bev_queries:    torch.Tensor,   # (B, N_bev, C)
        img_feats:      torch.Tensor,   # (B*6, C, H_feat, W_feat)
        ref_pts:        torch.Tensor,   # (N_bev, n_z, 3)  ego frame
        cam_intrinsics: torch.Tensor,   # (B, 6, 3, 3)
        cam_extrinsics: torch.Tensor,   # (B, 6, 4, 4)
        img_h: int,
        img_w: int,
    ) -> torch.Tensor:                  # (B, N_bev, C)

        B, N_bev, C = bev_queries.shape
        n_cam        = 6
        n_z          = ref_pts.shape[1]
        output       = torch.zeros_like(bev_queries)

        for start in range(0, N_bev, self.chunk_size):
            end      = min(start + self.chunk_size, N_bev)
            chunk_n  = end - start

            chunk_q   = bev_queries[:, start:end, :]   # (B, chunk_n, C)
            chunk_ref = ref_pts[start:end]              # (chunk_n, n_z, 3)

            # ── 1. BEV → 이미지 투영 ──────────────────────────────────────────
            uv_norm, valid = project_bev_to_image(
                chunk_ref, cam_intrinsics, cam_extrinsics, img_h, img_w
            )
            # uv_norm: (B, 6, chunk_n*n_z, 2)
            # valid:   (B, 6, chunk_n*n_z) bool

            # ── 2. grid_sample ────────────────────────────────────────────────
            uv = uv_norm.reshape(B * n_cam, 1, chunk_n * n_z, 2)
            sampled = F.grid_sample(
                img_feats, uv,
                mode="bilinear", align_corners=True, padding_mode="zeros",
            )  # (B*6, C, 1, chunk_n*n_z)
            sampled = sampled.squeeze(2)                            # (B*6, C, chunk_n*n_z)
            sampled = sampled.reshape(B, n_cam, C, chunk_n, n_z)

            # ── 3. z anchor 평균 (valid 가중 평균) ────────────────────────────
            valid_z   = valid.reshape(B, n_cam, chunk_n, n_z).float()  # (B,6,chunk,n_z)
            z_count   = valid_z.sum(dim=-1).clamp(min=1.0)             # (B,6,chunk)
            sampled   = (sampled * valid_z.unsqueeze(2)).sum(dim=-1)   # (B,6,C,chunk)
            sampled   = sampled / z_count.unsqueeze(2)                 # (B,6,C,chunk)
            sampled   = sampled.permute(0, 3, 1, 2)                    # (B,chunk,6,C)

            valid_any = valid_z.any(dim=-1).permute(0, 2, 1)           # (B,chunk,6)

            # ── 4. Attention ──────────────────────────────────────────────────
            # 유효하지 않은 카메라의 V를 0으로 마스킹
            V = self.v_proj(sampled) * valid_any.float().unsqueeze(-1) # (B,chunk,6,C)
            K = self.k_proj(sampled)                                    # (B,chunk,6,C)
            Q = self.q_proj(chunk_q)                                    # (B,chunk,C)

            # dot-product score: (B, chunk, 6)
            attn = (Q.unsqueeze(2) * K).sum(-1) * self.scale

            # 유효하지 않은 카메라 → -inf
            attn = attn.masked_fill(~valid_any, float("-inf"))

            # 모든 카메라가 유효하지 않은 위치 → NaN 방지 (0으로 대체 → softmax 균등, V=0 → out=0)
            all_invalid = ~valid_any.any(dim=-1, keepdim=True)         # (B,chunk,1)
            attn = torch.where(all_invalid.expand_as(attn), torch.zeros_like(attn), attn)

            attn_w = torch.softmax(attn, dim=-1)                       # (B,chunk,6)
            attn_w = self.dropout(attn_w)

            # 가중 합산: (B, chunk, C)
            out = (attn_w.unsqueeze(-1) * V).sum(dim=2)
            output[:, start:end, :] = self.out_proj(out)

        return output
