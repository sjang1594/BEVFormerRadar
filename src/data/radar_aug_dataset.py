"""
RadarAugDataset — augment_radar_real.py 출력 로딩 + 인라인 radar augmentation.

SyntheticDataset과 동일한 인터페이스. MixedDataset에 그대로 연결 가능.

A (사전생성): dynamic_compositor copy-paste → .npz (구조적 augmentation)
B (인라인):  match_real_density + radial noise jitter + flip (확률적 augmentation)

카메라: 실제 nuScenes 이미지 원본 → frozen ResNet50 domain gap 없음.
레이더: real radar + copy-pasted 동적 객체 + 인라인 perturbation.
"""

import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset


class RadarAugDataset(Dataset):
    """
    Args:
        npz_dir:            augment_radar_real.py 출력 디렉토리
        cfg:                config.yaml dict
        augment:            True → flip + radar jitter
        match_real_density: True → 레이더 포인트 수를 실제 분포에 맞춤
        radar_noise_std:    인라인 xyz 노이즈 σ [m] (0이면 추가 안 함)
    """

    # nuScenes mini 실제 레이더 밀도 (diagnose_density.py 측정값)
    _REAL_DENSITY_MEAN = 113.6
    _REAL_DENSITY_STD  = 32.0
    _REAL_DENSITY_MIN  = 53
    _REAL_DENSITY_MAX  = 198

    def __init__(self, npz_dir: str, cfg: dict,
                 augment: bool = False,
                 match_real_density: bool = True,
                 radar_noise_std: float = 0.05):
        self.files             = sorted(glob.glob(os.path.join(npz_dir, "sample_*.npz")))
        self.mean              = np.array(cfg["data"]["img_mean"], dtype=np.float32).reshape(3, 1, 1)
        self.std               = np.array(cfg["data"]["img_std"],  dtype=np.float32).reshape(3, 1, 1)
        self.augment           = augment
        self.match_real_density = match_real_density
        self.radar_noise_std   = radar_noise_std

        if not self.files:
            raise FileNotFoundError(f"No sample_*.npz found in {npz_dir}")

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> dict:
        s = np.load(self.files[idx])

        # ── 카메라: uint8 → float32 → ImageNet 정규화 ────────────────────────
        imgs = s["imgs"].astype(np.float32) / 255.0   # (6, 3, H, W)
        imgs = (imgs - self.mean) / self.std

        K = s["cam_intrinsics"].copy()   # (6, 3, 3)

        # ── [B] 인라인 augmentation: 수평 플립 ──────────────────────────────
        do_flip = self.augment and (np.random.rand() < 0.5)
        if do_flip:
            FLIP_ORDER = [0, 2, 1, 3, 5, 4]
            imgs = imgs[FLIP_ORDER, :, :, ::-1].copy()
            K = K[FLIP_ORDER].copy()
            K[:, 0, 2] = s["imgs"].shape[-1] - 1 - K[:, 0, 2]

        # ── 레이더 로드 ───────────────────────────────────────────────────────
        radar = (s["radar_pts"].copy()
                 if "radar_pts" in s.files
                 else np.zeros((0, 5), dtype=np.float32))

        # ── [B] 인라인 augmentation: density matching ────────────────────────
        if self.match_real_density and len(radar) > 0:
            n_target = int(np.clip(
                np.random.normal(self._REAL_DENSITY_MEAN, self._REAL_DENSITY_STD),
                self._REAL_DENSITY_MIN, self._REAL_DENSITY_MAX,
            ))
            if n_target < len(radar):
                idx_sub = np.random.choice(len(radar), n_target, replace=False)
                radar   = radar[idx_sub]

        # ── [B] 인라인 augmentation: xyz 노이즈 jitter ───────────────────────
        if self.radar_noise_std > 0 and len(radar) > 0:
            radar = radar.copy()
            radar[:, :3] += np.random.randn(len(radar), 3).astype(np.float32) * self.radar_noise_std

        # ── [B] 인라인 augmentation: flip 시 vx/vy 부호 반전 ────────────────
        if do_flip and len(radar) > 0:
            radar = radar.copy()
            radar[:, 3] = -radar[:, 3]   # vx sign flip (y-axis mirror)

        return {
            "imgs":           torch.from_numpy(imgs),
            "cam_intrinsics": torch.from_numpy(K.astype(np.float32)),
            "cam_extrinsics": torch.from_numpy(s["cam_extrinsics"].astype(np.float32)),
            "ego_to_world":   torch.from_numpy(s["ego_to_world"].astype(np.float32)),
            "radar_pts_ego":  torch.from_numpy(radar),
            "gt_boxes":       torch.from_numpy(s["gt_boxes"].astype(np.float32)),
            "gt_labels":      torch.from_numpy(s["gt_labels"].astype(np.int64)),
            "timestamp":      0.0,
            "sample_token":   f"radar_aug_{int(idx):06d}",
            "prev_token":     "",   # 항상 첫 프레임 → temporal state 초기화
        }
