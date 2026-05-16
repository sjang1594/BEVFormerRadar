"""
이미지 전처리 — 정규화 + 수평 플립 증강 (학습 데이터 2× 확장)
"""

import numpy as np
from typing import Dict


class Normalize:
    """ImageNet 정규화: (pixel - mean) / std, channel-first."""

    def __init__(self, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        self.mean = np.array(mean, dtype=np.float32).reshape(3, 1, 1)
        self.std  = np.array(std,  dtype=np.float32).reshape(3, 1, 1)

    def __call__(self, sample: dict) -> dict:
        imgs = sample["imgs"]           # (6, 3, H, W) float32 in [0,1]
        sample["imgs"] = (imgs - self.mean) / self.std
        return sample


class RandomHorizontalFlip:
    """
    50% 확률로 카메라 이미지 + GT 박스를 수평 플립.

    카메라 순서도 뒤집음:
        [FRONT, FL, FR, BACK, BL, BR] → [FRONT, FR, FL, BACK, BR, BL]
        (인덱스: 0,1,2,3,4,5 → 0,2,1,3,5,4)

    GT 박스 처리:
        x_ego → -x_ego, yaw_ego → π - yaw_ego
    """

    FLIP_ORDER = [0, 2, 1, 3, 5, 4]   # CAM_FRONT 유지, L↔R 교환

    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, sample: dict) -> dict:
        if np.random.random() >= self.p:
            return sample

        # 이미지 플립 (좌우) + 카메라 순서 교환
        imgs = sample["imgs"]                           # (6,3,H,W)
        imgs = imgs[self.FLIP_ORDER]                    # 순서 교환
        imgs = imgs[:, :, :, ::-1].copy()              # 좌우 반전
        sample["imgs"] = imgs

        # intrinsic: cx 반전 (cx' = W - cx)
        W = imgs.shape[-1]
        K = sample["cam_intrinsics"].copy()             # (6,3,3)
        K = K[self.FLIP_ORDER]
        K[:, 0, 2] = W - 1 - K[:, 0, 2]
        sample["cam_intrinsics"] = K

        # extrinsic: cam→ego  (cam frame x축 반전: T_c2e_new = T_c2e @ flip4)
        ext = sample["cam_extrinsics"].copy()           # (6,4,4)
        ext = ext[self.FLIP_ORDER]
        flip4 = np.array([[-1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]], dtype=ext.dtype)
        ext = ext @ flip4                               # (6,4,4) @ (4,4) → (6,4,4)
        sample["cam_extrinsics"] = ext

        # GT 박스: x_ego 반전, yaw 반전
        gt_boxes = sample["gt_boxes"].copy()            # (M,7)
        if len(gt_boxes) > 0:
            gt_boxes[:, 0] = -gt_boxes[:, 0]           # x_ego
            gt_boxes[:, 6] = np.pi - gt_boxes[:, 6]   # yaw_ego
        sample["gt_boxes"] = gt_boxes

        # 레이더 포인트: x, vx 반전
        radar = sample["radar_pts_ego"].copy()          # (N,5)
        if len(radar) > 0:
            radar[:, 0] = -radar[:, 0]                 # x_ego
            radar[:, 3] = -radar[:, 3]                 # vx_ego
        sample["radar_pts_ego"] = radar

        return sample


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, sample: dict) -> dict:
        for t in self.transforms:
            sample = t(sample)
        return sample


def build_train_transform(cfg: dict):
    return Compose([
        Normalize(cfg["data"]["img_mean"], cfg["data"]["img_std"]),
        RandomHorizontalFlip(p=0.5),
    ])


def build_val_transform(cfg: dict):
    return Compose([
        Normalize(cfg["data"]["img_mean"], cfg["data"]["img_std"]),
    ])
