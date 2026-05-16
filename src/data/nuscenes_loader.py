"""
nuScenes 데이터 로더 — 6 카메라 + 5 레이더 + GT 박스

기존 nuscenes_pmbm_tracker.py의 quat_to_rot, se3_transform, load_all_radars_world,
get_ego_pose_world, load_gt_world 패턴을 재사용.
카메라 로딩(intrinsic/extrinsic)을 추가.
"""

import os
import math
import numpy as np
from PIL import Image
from typing import Dict, List, Optional, Tuple

try:
    from nuscenes.nuscenes import NuScenes
    from nuscenes.utils.data_classes import RadarPointCloud
except ImportError:
    raise ImportError("nuscenes-devkit not installed. Run: pip install nuscenes-devkit")


# ─────────────────────────────────────────────────────────────────────────────
# nuScenes 카테고리 → 클래스 인덱스 (config.yaml과 일치)
# ─────────────────────────────────────────────────────────────────────────────
CATEGORY_MAP: Dict[str, int] = {}
for cat in ["vehicle.car", "vehicle.truck", "vehicle.bus.bendy", "vehicle.bus.rigid",
            "vehicle.construction", "vehicle.trailer"]:
    CATEGORY_MAP[cat] = 0  # vehicle
for cat in ["human.pedestrian.adult", "human.pedestrian.child",
            "human.pedestrian.wheelchair", "human.pedestrian.stroller"]:
    CATEGORY_MAP[cat] = 1  # pedestrian
for cat in ["vehicle.motorcycle", "vehicle.bicycle"]:
    CATEGORY_MAP[cat] = 2  # cyclist

CLASS_NAMES = ["vehicle", "pedestrian", "cyclist"]

RADAR_CHANNELS = [
    "RADAR_FRONT", "RADAR_FRONT_LEFT", "RADAR_FRONT_RIGHT",
    "RADAR_BACK_LEFT", "RADAR_BACK_RIGHT",
]
CAM_CHANNELS = [
    "CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT",
    "CAM_BACK", "CAM_BACK_LEFT", "CAM_BACK_RIGHT",
]


# ─────────────────────────────────────────────────────────────────────────────
# 기하학 유틸리티 (nuscenes_pmbm_tracker.py에서 재사용)
# ─────────────────────────────────────────────────────────────────────────────
def quat_to_rot(q) -> np.ndarray:
    """Unit quaternion [w,x,y,z] → 3×3 rotation matrix."""
    w, x, y, z = q
    return np.array([
        [1-2*(y*y+z*z),   2*(x*y-z*w),   2*(x*z+y*w)],
        [  2*(x*y+z*w), 1-2*(x*x+z*z),   2*(y*z-x*w)],
        [  2*(x*z-y*w),   2*(y*z+x*w), 1-2*(x*x+y*y)],
    ], dtype=np.float64)


def se3_transform(rot: np.ndarray, trans: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """pts: (N,3) → rot @ pts.T + trans."""
    return (rot @ pts.T).T + trans


def make_se3(rot: np.ndarray, trans: np.ndarray) -> np.ndarray:
    """Build 4×4 SE3 matrix from 3×3 rot and 3-vec trans."""
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = rot
    T[:3,  3] = trans
    return T


# ─────────────────────────────────────────────────────────────────────────────
# NuScenesLoader
# ─────────────────────────────────────────────────────────────────────────────
class NuScenesLoader:
    """
    nuScenes 데이터 로더.

    주요 메서드:
        get_sample(sample_token) → dict
        get_scene_tokens(scene_idx) → List[str]  (sample token 목록)
    """

    def __init__(self, cfg: dict):
        dataroot = cfg["data"]["dataroot"]
        # config.yaml에서 dataroot가 상대 경로면 이 파일 기준으로 해석
        if not os.path.isabs(dataroot):
            base = os.path.dirname(os.path.dirname(os.path.dirname(
                os.path.abspath(__file__))))  # BEVFormerRadar/
            dataroot = os.path.normpath(os.path.join(base, dataroot))

        self.nusc = NuScenes(
            version=cfg["data"]["version"],
            dataroot=dataroot,
            verbose=False,
        )
        self.cfg = cfg
        self.max_radar_range = cfg["data"]["max_radar_range"]
        self.cam_channels = cfg["data"]["cam_channels"]
        self.radar_channels = cfg["data"]["radar_channels"]

    # ── 씬 / 샘플 열거 ────────────────────────────────────────────────────────

    def get_scene_tokens(self, scene_idx: int) -> List[str]:
        """장면 내 모든 sample_token 목록 (시간 순)."""
        scene = self.nusc.scene[scene_idx]
        tokens = []
        tok = scene["first_sample_token"]
        while tok:
            tokens.append(tok)
            tok = self.nusc.get("sample", tok)["next"]
        return tokens

    def num_scenes(self) -> int:
        return len(self.nusc.scene)

    # ── 핵심: 단일 샘플 로딩 ─────────────────────────────────────────────────

    def get_sample(self, sample_token: str) -> dict:
        """
        단일 샘플을 로드해 dict 반환.

        Returns:
            imgs:           np.ndarray (6, 3, H, W) float32, [0,1]
            cam_intrinsics: np.ndarray (6, 3, 3)
            cam_extrinsics: np.ndarray (6, 4, 4)   cam→ego SE3
            ego_to_world:   np.ndarray (4, 4)
            radar_pts_ego:  np.ndarray (N, 5)      [x,y,z,vx,vy] ego frame
            gt_boxes:       np.ndarray (M, 7)      [x,y,z,w,l,h,yaw] ego frame
            gt_labels:      np.ndarray (M,)         0=vehicle,1=ped,2=cyclist
            timestamp:      float
            sample_token:   str
            prev_token:     str | ""  (빈 문자열이면 첫 프레임)
        """
        sample = self.nusc.get("sample", sample_token)

        # 1. Ego pose (world frame)
        ref_chan = self.cam_channels[0]  # CAM_FRONT를 ego 기준으로 사용
        ref_sd = self.nusc.get("sample_data", sample["data"][ref_chan])
        R_e2w, t_e2w = self._get_ego_pose(ref_sd["token"])
        ego_to_world = make_se3(R_e2w, t_e2w)

        # 2. 카메라 이미지 + intrinsic/extrinsic
        imgs, intrinsics, extrinsics = self._load_cameras(sample, R_e2w, t_e2w)

        # 3. 레이더 (ego frame)
        radar_pts_ego = self._load_radars_ego(sample)

        # 4. GT 박스 (ego frame)
        gt_boxes, gt_labels = self._load_gt_ego(sample, R_e2w, t_e2w)

        return {
            "imgs":           imgs,             # (6, 3, H, W) float32
            "cam_intrinsics": intrinsics,       # (6, 3, 3) float64
            "cam_extrinsics": extrinsics,       # (6, 4, 4) float64  cam→ego
            "ego_to_world":   ego_to_world,     # (4, 4) float64
            "radar_pts_ego":  radar_pts_ego,    # (N, 5) float32
            "gt_boxes":       gt_boxes,         # (M, 7) float32
            "gt_labels":      gt_labels,        # (M,)   int64
            "timestamp":      sample["timestamp"] * 1e-6,
            "sample_token":   sample_token,
            "prev_token":     sample.get("prev", ""),
        }

    # ── 내부 로더 ─────────────────────────────────────────────────────────────

    def _get_ego_pose(self, sample_data_token: str) -> Tuple[np.ndarray, np.ndarray]:
        """sample_data_token → (R_ego2world, t_ego2world)."""
        sd = self.nusc.get("sample_data", sample_data_token)
        ep = self.nusc.get("ego_pose", sd["ego_pose_token"])
        R = quat_to_rot(ep["rotation"])
        t = np.array(ep["translation"], dtype=np.float64)
        return R, t

    def _load_cameras(self, sample, R_e2w, t_e2w):
        """
        Returns:
            imgs:       (6, 3, H, W) float32 in [0,1]
            intrinsics: (6, 3, 3) float64
            extrinsics: (6, 4, 4) float64  cam→ego
        """
        cfg_d = self.cfg["data"]
        H, W = cfg_d["img_h"], cfg_d["img_w"]

        imgs       = np.zeros((6, 3, H, W), dtype=np.float32)
        intrinsics = np.zeros((6, 3, 3), dtype=np.float64)
        extrinsics = np.zeros((6, 4, 4), dtype=np.float64)

        for i, ch in enumerate(self.cam_channels):
            if ch not in sample["data"]:
                extrinsics[i] = np.eye(4)
                continue

            sd_token = sample["data"][ch]
            sd = self.nusc.get("sample_data", sd_token)
            cs = self.nusc.get("calibrated_sensor", sd["calibrated_sensor_token"])

            # intrinsic K
            K = np.array(cs["camera_intrinsic"], dtype=np.float64)  # (3,3) native res

            # extrinsic: cam → ego SE3
            R_c2e = quat_to_rot(cs["rotation"])
            t_c2e = np.array(cs["translation"], dtype=np.float64)

            # 이미지 리사이즈에 따른 K 스케일 조정
            img_path = os.path.join(self.nusc.dataroot, sd["filename"])
            try:
                img_pil = Image.open(img_path).convert("RGB")
                orig_w, orig_h = img_pil.size   # nuScenes: 1600×900
                scale_x = W / orig_w
                scale_y = H / orig_h
                img_pil = img_pil.resize((W, H), Image.BILINEAR)
                img_np = np.array(img_pil, dtype=np.float32) / 255.0   # (H,W,3)
                imgs[i] = img_np.transpose(2, 0, 1)                    # (3,H,W)
            except Exception:
                scale_x = scale_y = W / 1600.0

            K_scaled = K.copy()
            K_scaled[0, :] *= scale_x   # fx, cx
            K_scaled[1, :] *= scale_y   # fy, cy

            intrinsics[i] = K_scaled
            extrinsics[i] = make_se3(R_c2e, t_c2e)

        return imgs, intrinsics, extrinsics

    def _load_radars_ego(self, sample) -> np.ndarray:
        """
        5채널 레이더 포인트를 ego frame으로 변환해 (N, 5)로 반환.
        columns: [x, y, z, vx_comp, vy_comp]
        """
        all_pts = []
        for ch in self.radar_channels:
            if ch not in sample["data"]:
                continue
            sd_token = sample["data"][ch]
            sd = self.nusc.get("sample_data", sd_token)
            pc_path = os.path.join(self.nusc.dataroot, sd["filename"])
            try:
                pc = RadarPointCloud.from_file(pc_path)
            except Exception:
                continue

            pts  = pc.points[:3, :].T             # (N,3) sensor frame
            vx   = pc.points[8, :]               # compensated vx
            vy   = pc.points[9, :]               # compensated vy

            cs = self.nusc.get("calibrated_sensor", sd["calibrated_sensor_token"])
            R_s2e = quat_to_rot(cs["rotation"])
            t_s2e = np.array(cs["translation"], dtype=np.float64)

            pts_e = se3_transform(R_s2e, t_s2e, pts)          # (N,3) ego
            vel_s = np.stack([vx, vy, np.zeros_like(vx)], axis=1)
            vel_e = (R_s2e @ vel_s.T).T                        # (N,3) ego

            # 범위 필터 (ego 거리)
            r = np.hypot(pts_e[:, 0], pts_e[:, 1])
            mask = r < self.max_radar_range

            if mask.sum() == 0:
                continue

            pts_e = pts_e[mask]
            vel_e = vel_e[mask]
            chunk = np.concatenate([pts_e, vel_e[:, :2]], axis=1)  # (N,5)
            all_pts.append(chunk)

        if not all_pts:
            return np.zeros((0, 5), dtype=np.float32)
        return np.vstack(all_pts).astype(np.float32)

    def _load_gt_ego(self, sample, R_e2w, t_e2w):
        """
        GT 박스를 ego frame으로 변환.
        Returns:
            gt_boxes:  (M, 7) [x_ego, y_ego, z_ego, w, l, h, yaw_ego] float32
            gt_labels: (M,)  int64
        """
        boxes_list = []
        labels_list = []

        # world→ego 역변환
        R_w2e = R_e2w.T
        t_w2e = -R_w2e @ t_e2w

        for ann_token in sample["anns"]:
            ann = self.nusc.get("sample_annotation", ann_token)
            cat = ann["category_name"]

            # 클래스 매핑 (prefix 매칭)
            cls_idx = None
            for prefix, idx in CATEGORY_MAP.items():
                if cat.startswith(prefix):
                    cls_idx = idx
                    break
            if cls_idx is None:
                continue

            # world → ego
            pos_w = np.array(ann["translation"], dtype=np.float64)
            pos_e = se3_transform(R_w2e, t_w2e, pos_w[None])[0]

            # yaw in ego frame
            R_box_w = quat_to_rot(ann["rotation"])
            R_box_e = R_w2e @ R_box_w
            yaw_e = math.atan2(R_box_e[1, 0], R_box_e[0, 0])

            # wlh
            w, l, h = ann["size"]   # nuScenes: [width, length, height]

            boxes_list.append([pos_e[0], pos_e[1], pos_e[2], w, l, h, yaw_e])
            labels_list.append(cls_idx)

        if not boxes_list:
            return (np.zeros((0, 7), dtype=np.float32),
                    np.zeros((0,), dtype=np.int64))

        return (np.array(boxes_list, dtype=np.float32),
                np.array(labels_list, dtype=np.int64))
