"""
BEVFormerDataset — PyTorch Dataset

각 __getitem__은 sample_token 하나를 로드해 numpy → torch tensor 변환 후 반환.
"""

import numpy as np
import torch
from torch.utils.data import Dataset
from typing import List, Optional

from .nuscenes_loader import NuScenesLoader
from .transforms import Compose


class BEVFormerDataset(Dataset):
    """
    Args:
        cfg:          config.yaml 딕셔너리
        scene_indices: 사용할 장면 인덱스 목록 (train: [0..5], val: [6,7])
        transform:    Compose 변환 (None이면 정규화만)
    """

    def __init__(self, cfg: dict, scene_indices: List[int],
                 transform: Optional[Compose] = None):
        self.loader    = NuScenesLoader(cfg)
        self.transform = transform
        self.cfg       = cfg

        # 모든 sample_token 목록 수집 (scene 순서 유지)
        self.tokens: List[str] = []
        for si in scene_indices:
            self.tokens.extend(self.loader.get_scene_tokens(si))

    def __len__(self) -> int:
        return len(self.tokens)

    def __getitem__(self, idx: int) -> dict:
        token = self.tokens[idx]
        sample = self.loader.get_sample(token)

        if self.transform is not None:
            sample = self.transform(sample)

        # numpy → torch tensor
        return {
            "imgs":           torch.from_numpy(sample["imgs"]),           # (6,3,H,W)
            "cam_intrinsics": torch.from_numpy(sample["cam_intrinsics"].astype(np.float32)),  # (6,3,3)
            "cam_extrinsics": torch.from_numpy(sample["cam_extrinsics"].astype(np.float32)),  # (6,4,4)
            "ego_to_world":   torch.from_numpy(sample["ego_to_world"].astype(np.float32)),    # (4,4)
            "radar_pts_ego":  torch.from_numpy(sample["radar_pts_ego"]),  # (N,5)
            "gt_boxes":       torch.from_numpy(sample["gt_boxes"]),        # (M,7)
            "gt_labels":      torch.from_numpy(sample["gt_labels"]),       # (M,)
            "timestamp":      sample["timestamp"],
            "sample_token":   sample["sample_token"],
            "prev_token":     sample["prev_token"],
        }


def collate_fn(batch: list) -> dict:
    """
    가변 길이 gt_boxes, gt_labels, radar_pts_ego를 리스트로 묶는 collate.
    imgs/intrinsics/extrinsics 등 고정 크기 텐서는 stack.
    """
    keys_stack = ["imgs", "cam_intrinsics", "cam_extrinsics", "ego_to_world"]
    keys_list  = ["gt_boxes", "gt_labels", "radar_pts_ego"]
    keys_meta  = ["timestamp", "sample_token", "prev_token"]

    result = {}
    for k in keys_stack:
        result[k] = torch.stack([b[k] for b in batch], dim=0)
    for k in keys_list:
        result[k] = [b[k] for b in batch]
    for k in keys_meta:
        result[k] = [b[k] for b in batch]

    return result
