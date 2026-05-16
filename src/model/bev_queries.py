"""
BEV Query Grid — 학습 가능한 BEV 쿼리 + ego frame 레퍼런스 포인트.
"""

import torch
import torch.nn as nn
from src.geometry.projections import build_bev_ref_points


class BEVQueryGrid(nn.Module):
    """
    BEV 그리드의 학습 가능한 쿼리 임베딩.

    Attributes:
        embed:   nn.Embedding(H*W, C)  — 학습 가능한 BEV 쿼리
        ref_pts: (H*W, n_z, 3) buffer  — ego frame 레퍼런스 포인트 (고정)
    """

    def __init__(self, cfg: dict):
        super().__init__()
        H = cfg["bev"]["h"]
        W = cfg["bev"]["w"]
        C = cfg["model"]["bev_channels"]

        self.H = H
        self.W = W
        self.N = H * W   # 40000

        self.embed = nn.Embedding(self.N, C)
        nn.init.normal_(self.embed.weight, std=0.02)

        # 레퍼런스 포인트: (H, W, n_z, 3) → (N, n_z, 3)
        ref = build_bev_ref_points(cfg, device=torch.device("cpu"))  # (H, W, n_z, 3)
        ref = ref.reshape(self.N, -1, 3)                             # (N, n_z, 3)
        self.register_buffer("ref_pts", ref)

    def get_queries(self, B: int) -> torch.Tensor:
        """(B, N_bev, C) 학습 가능한 BEV 쿼리."""
        return self.embed.weight.unsqueeze(0).expand(B, -1, -1)

    def get_ref_pts(self) -> torch.Tensor:
        """(N_bev, n_z, 3) ego frame 레퍼런스 포인트."""
        return self.ref_pts
