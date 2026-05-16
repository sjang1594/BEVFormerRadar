"""
SyntheticDataset — NeuralSensorSim이 생성한 .npz 파일을 BEVFormerDataset과
동일한 포맷으로 반환하는 PyTorch Dataset.

각 샘플은 독립 프레임 (prev_token="") — temporal 연속성 없음.
"""

import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset


class SyntheticDataset(Dataset):
    """
    Args:
        npz_dir:   .npz 파일이 있는 디렉토리
        cfg:       config.yaml dict (img_mean, img_std 사용)
        augment:   True이면 랜덤 수평 플립 적용
    """

    def __init__(self, npz_dir: str, cfg: dict, augment: bool = False):
        self.files   = sorted(glob.glob(os.path.join(npz_dir, "sample_*.npz")))
        self.mean    = np.array(cfg["data"]["img_mean"], dtype=np.float32).reshape(3, 1, 1)
        self.std     = np.array(cfg["data"]["img_std"],  dtype=np.float32).reshape(3, 1, 1)
        self.augment = augment

        if not self.files:
            raise FileNotFoundError(f"No sample_*.npz found in {npz_dir}")

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> dict:
        s = np.load(self.files[idx])

        # uint8 [0,255] → float32 [0,1] → ImageNet 정규화
        imgs = s["imgs"].astype(np.float32) / 255.0   # (6, 3, H, W)
        imgs = (imgs - self.mean) / self.std

        # 수평 플립 (augment)
        if self.augment and np.random.rand() < 0.5:
            FLIP_ORDER = [0, 2, 1, 3, 5, 4]
            imgs = imgs[FLIP_ORDER, :, :, ::-1].copy()
            # cx flip
            K = s["cam_intrinsics"].copy()
            K = K[FLIP_ORDER]
            K[:, 0, 2] = s["imgs"].shape[-1] - 1 - K[:, 0, 2]
        else:
            K = s["cam_intrinsics"]

        radar = s["radar_pts"] if "radar_pts" in s.files else np.zeros((0, 5), dtype=np.float32)

        return {
            "imgs":           torch.from_numpy(imgs),                            # (6,3,H,W)
            "cam_intrinsics": torch.from_numpy(K.astype(np.float32)),            # (6,3,3)
            "cam_extrinsics": torch.from_numpy(s["cam_extrinsics"].astype(np.float32)),  # (6,4,4)
            "ego_to_world":   torch.from_numpy(s["ego_to_world"].astype(np.float32)),    # (4,4)
            "radar_pts_ego":  torch.from_numpy(radar),                           # (N,5)
            "gt_boxes":       torch.from_numpy(s["gt_boxes"].astype(np.float32)),# (M,7)
            "gt_labels":      torch.from_numpy(s["gt_labels"].astype(np.int64)), # (M,)
            "timestamp":      0.0,
            "sample_token":   f"synthetic_{idx:06d}",
            "prev_token":     "",   # 항상 첫 프레임 → temporal state 초기화
        }
