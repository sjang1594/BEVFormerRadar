"""
FPN Neck — C3/C4/C5 → single P4 feature map (256ch).

lateral conv: 1×1, reduces channel to out_ch
top-down:     C5 → upsample → add to C4 → 3×3 conv → P4
              C4(merged) → upsample → add to C3 → (unused, but computed for symmetry)

Output: P4  (B*6, 256, H/16, W/16)  → used by SCA
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FPNNeck(nn.Module):
    """
    FPN: C3(512) + C4(1024) + C5(2048) → P4(256).

    Forward input:  dict { 'C3', 'C4', 'C5' }  from ResNet50Backbone
    Forward output: Tensor (B*6, out_channels, H/16, W/16)
    """

    IN_CHANNELS = {"C3": 512, "C4": 1024, "C5": 2048}

    def __init__(self, out_channels: int = 256):
        super().__init__()
        self.out_channels = out_channels

        # lateral 1×1 convs
        self.lat_c3 = nn.Conv2d(self.IN_CHANNELS["C3"], out_channels, 1)
        self.lat_c4 = nn.Conv2d(self.IN_CHANNELS["C4"], out_channels, 1)
        self.lat_c5 = nn.Conv2d(self.IN_CHANNELS["C5"], out_channels, 1)

        # output 3×3 conv for P4
        self.out_conv = nn.Conv2d(out_channels, out_channels, 3, padding=1)

    def forward(self, features: dict) -> torch.Tensor:
        c3 = self.lat_c3(features["C3"])   # (B*6, 256, H/8,  W/8)
        c4 = self.lat_c4(features["C4"])   # (B*6, 256, H/16, W/16)
        c5 = self.lat_c5(features["C5"])   # (B*6, 256, H/32, W/32)

        # top-down: C5 → C4
        p4 = c4 + F.interpolate(c5, size=c4.shape[-2:], mode="nearest")
        p4 = self.out_conv(p4)             # (B*6, 256, H/16, W/16)

        return p4
