"""
CenterPoint-style Detection Head.

Input : BEV features (B, N_bev, C)
Output: dict
    heatmap : (B, num_classes, H, W)  sigmoid  — 객체 중심 확률
    offset  : (B, 2, H, W)            raw       — sub-cell x,y offset
    size    : (B, 3, H, W)            raw       — w,l,h [m]
    yaw     : (B, 2, H, W)            raw       — sin(yaw), cos(yaw)
"""

import torch
import torch.nn as nn


class DetectionHead(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        in_ch   = cfg["model"]["bev_channels"]   # 256
        num_cls = cfg["model"]["num_classes"]     # 3
        self.H  = cfg["bev"]["h"]                # 200
        self.W  = cfg["bev"]["w"]                # 200

        # 공유 feature refinement
        self.shared = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(in_ch),
            nn.ReLU(inplace=True),
        )

        # 각 출력 branch
        self.heatmap_head = nn.Conv2d(in_ch, num_cls, 1)
        self.offset_head  = nn.Conv2d(in_ch, 2, 1)
        self.size_head    = nn.Conv2d(in_ch, 3, 1)
        self.yaw_head     = nn.Conv2d(in_ch, 2, 1)

        # heatmap bias 초기화 (focal loss 초기 수렴 안정화)
        nn.init.constant_(self.heatmap_head.bias, -2.19)   # log(0.1/0.9)

    def forward(self, bev_feats: torch.Tensor) -> dict:
        """bev_feats: (B, N_bev, C)"""
        B, N, C = bev_feats.shape
        x = bev_feats.permute(0, 2, 1).reshape(B, C, self.H, self.W)
        x = self.shared(x)

        return {
            "heatmap": torch.sigmoid(self.heatmap_head(x)),  # (B,3,H,W)
            "offset":  self.offset_head(x),                  # (B,2,H,W)
            "size":    self.size_head(x),                    # (B,3,H,W)
            "yaw":     self.yaw_head(x),                     # (B,2,H,W)
        }
