"""
BEVFormer — 전체 모델 통합.

Forward:
  imgs (B,6,3,H,W) + 카메라 파라미터 + prev 상태
    → Backbone + FPN → img_feats (B*6, 256, H/16, W/16)
    → BEVQueryGrid → queries (B, N_bev, 256)
    → BEVFormerEncoder (TSA+SCA+FFN) × 2
    → bev_feats (B, N_bev, 256)
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .backbone import ResNet50Backbone
from .fpn import FPNNeck
from .bev_queries import BEVQueryGrid
from .encoder import BEVFormerEncoder
from src.fusion.radar_fusion import RadarFusion


class BEVFormer(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        self.cfg = cfg
        self.use_radar = cfg["train"]["use_radar_fusion"]

        self.backbone = ResNet50Backbone(frozen=cfg["model"]["backbone_frozen"])
        self.fpn      = FPNNeck(out_channels=cfg["model"]["fpn_out_channels"])
        self.bev_grid = BEVQueryGrid(cfg)
        self.encoder  = BEVFormerEncoder(cfg)
        if self.use_radar:
            self.radar_fusion = RadarFusion(cfg)

        self.img_h = cfg["data"]["img_h"]   # 448
        self.img_w = cfg["data"]["img_w"]   # 800

    def extract_img_feats(self, imgs: torch.Tensor) -> torch.Tensor:
        """
        imgs: (B, 6, 3, H, W)
        Returns: (B*6, C, H/16, W/16)
        """
        B, n_cam, C, H, W = imgs.shape
        imgs_flat = imgs.reshape(B * n_cam, C, H, W)
        feats = self.backbone(imgs_flat)
        p4    = self.fpn(feats)
        return p4

    def forward(
        self,
        imgs:           torch.Tensor,        # (B, 6, 3, H, W)
        cam_intrinsics: torch.Tensor,        # (B, 6, 3, 3)
        cam_extrinsics: torch.Tensor,        # (B, 6, 4, 4)
        ego_to_world:   torch.Tensor,        # (B, 4, 4)
        prev_bev:       torch.Tensor | None = None,
        prev_ego_to_world: torch.Tensor | None = None,
        radar_pts_list: list | None = None,  # list of (N_i, 5)
    ) -> torch.Tensor:                       # (B, N_bev, C)

        B = imgs.shape[0]
        device = imgs.device

        # ── 1. Image feature extraction ────────────────────────────────────
        img_feats = self.extract_img_feats(imgs)   # (B*6, 256, 28, 50)

        # ── 2. BEV queries + ref points ────────────────────────────────────
        bev_queries = self.bev_grid.get_queries(B).to(device)   # (B, N, C)
        ref_pts     = self.bev_grid.get_ref_pts().to(device)    # (N, n_z, 3)

        # ── 2b. Radar fusion bias ───────────────────────────────────────────
        if self.use_radar and radar_pts_list is not None:
            bev_queries = self.radar_fusion(bev_queries, radar_pts_list)

        # ── 3. Ego delta (prev → current) ──────────────────────────────────
        has_prev  = prev_bev is not None and prev_ego_to_world is not None
        ego_delta = None
        if has_prev:
            # T_prev2curr = inv(ego_to_world_curr) @ ego_to_world_prev
            curr_w2e  = torch.inverse(ego_to_world)            # (B,4,4)
            ego_delta = curr_w2e @ prev_ego_to_world           # (B,4,4)

        # ── 4. BEVFormer Encoder ───────────────────────────────────────────
        bev_feats = self.encoder(
            bev_queries, img_feats, ref_pts,
            cam_intrinsics, cam_extrinsics,
            self.img_h, self.img_w,
            prev_bev, ego_delta, has_prev,
        )  # (B, N_bev, C)

        return bev_feats
