"""
ResNet50 Backbone — C3/C4/C5 feature extraction via forward hooks.
Backbone is frozen (pretrained ImageNet weights).
"""

import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights
from typing import Dict


class ResNet50Backbone(nn.Module):
    """
    ResNet50에서 layer2(C3), layer3(C4), layer4(C5) 출력을 추출.

    Forward input:  (B*6, 3, H, W)
    Forward output: dict { 'C3': (B*6, 512, H/8, W/8),
                           'C4': (B*6, 1024, H/16, W/16),
                           'C5': (B*6, 2048, H/32, W/32) }
    """

    def __init__(self, frozen: bool = True):
        super().__init__()
        net = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)

        self.stem   = nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool)
        self.layer1 = net.layer1   # C2: stride 4,  channels 256
        self.layer2 = net.layer2   # C3: stride 8,  channels 512
        self.layer3 = net.layer3   # C4: stride 16, channels 1024
        self.layer4 = net.layer4   # C5: stride 32, channels 2048

        if frozen:
            for p in self.parameters():
                p.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        x = self.stem(x)
        x = self.layer1(x)
        c3 = self.layer2(x)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)
        return {"C3": c3, "C4": c4, "C5": c5}
