"""
BEVFormerLayer + BEVFormerEncoder.

BEVFormerLayer (Pre-LN):
    x = x + TSA(LN(x), prev_bev, ego_delta, has_prev)
    x = x + SCA(LN(x), img_feats, ref_pts, cam_K, cam_E)
    x = x + FFN(LN(x))

BEVFormerEncoder:
    2 × BEVFormerLayer
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .temporal_self_attn import TemporalSelfAttention
from .spatial_cross_attn import SpatialCrossAttention


class FFN(nn.Module):
    def __init__(self, embed_dim: int, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class BEVFormerLayer(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        C          = cfg["model"]["bev_channels"]       # 256
        num_heads  = cfg["model"]["num_heads"]          # 8
        ffn_dim    = cfg["model"]["ffn_hidden_dim"]     # 1024
        dropout    = cfg["model"]["dropout"]            # 0.1
        chunk_size = cfg["model"]["sca_chunk_size"]     # 5000
        bev_h      = cfg["bev"]["h"]                    # 200
        bev_w      = cfg["bev"]["w"]                    # 200
        bev_range  = cfg["bev"]["x_max"]                # 50.0

        self.tsa   = TemporalSelfAttention(C, num_heads, bev_h, bev_w, bev_range, dropout)
        self.sca   = SpatialCrossAttention(C, num_heads, chunk_size, dropout)
        self.ffn   = FFN(C, ffn_dim, dropout)
        self.norm1 = nn.LayerNorm(C)
        self.norm2 = nn.LayerNorm(C)
        self.norm3 = nn.LayerNorm(C)

    def forward(
        self,
        bev_queries:    torch.Tensor,         # (B, N_bev, C)
        img_feats:      torch.Tensor,         # (B*6, C, H_f, W_f)
        ref_pts:        torch.Tensor,         # (N_bev, n_z, 3)
        cam_intrinsics: torch.Tensor,         # (B, 6, 3, 3)
        cam_extrinsics: torch.Tensor,         # (B, 6, 4, 4)
        img_h: int,
        img_w: int,
        prev_bev:    torch.Tensor | None,
        ego_delta:   torch.Tensor | None,
        has_prev:    bool,
    ) -> torch.Tensor:

        bev_queries = bev_queries + self.tsa(
            self.norm1(bev_queries), prev_bev, ego_delta, has_prev
        )
        bev_queries = bev_queries + self.sca(
            self.norm2(bev_queries), img_feats, ref_pts,
            cam_intrinsics, cam_extrinsics, img_h, img_w
        )
        bev_queries = bev_queries + self.ffn(self.norm3(bev_queries))
        return bev_queries


class BEVFormerEncoder(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        n_layers = cfg["model"]["num_encoder_layers"]   # 2
        self.layers = nn.ModuleList(
            [BEVFormerLayer(cfg) for _ in range(n_layers)]
        )

    def forward(
        self,
        bev_queries:    torch.Tensor,
        img_feats:      torch.Tensor,
        ref_pts:        torch.Tensor,
        cam_intrinsics: torch.Tensor,
        cam_extrinsics: torch.Tensor,
        img_h: int,
        img_w: int,
        prev_bev:    torch.Tensor | None,
        ego_delta:   torch.Tensor | None,
        has_prev:    bool,
    ) -> torch.Tensor:

        for layer in self.layers:
            bev_queries = layer(
                bev_queries, img_feats, ref_pts,
                cam_intrinsics, cam_extrinsics,
                img_h, img_w,
                prev_bev, ego_delta, has_prev,
            )
        return bev_queries
