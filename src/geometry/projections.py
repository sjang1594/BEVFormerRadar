"""
BEV ↔ 카메라 투영 기하학

project_bev_to_image():
    BEV 그리드의 3D 레퍼런스 포인트를 6개 카메라 이미지 평면으로 투영.
    반환: UV 좌표 + 유효 마스크

PointPainting/src/02_point_painting.py의 lidar_to_camera_transform 패턴 참조.
"""

import torch
import torch.nn.functional as F
from typing import Tuple


def build_bev_ref_points(cfg: dict, device: torch.device) -> torch.Tensor:
    """
    BEV 그리드의 3D 레퍼런스 포인트 생성 (ego frame).

    config의 x/y range, cell_size, z_anchors를 사용.

    Returns:
        ref_pts: (H_bev, W_bev, n_z, 3)  [x_ego, y_ego, z_ego]
    """
    bev = cfg["bev"]
    H, W = bev["h"], bev["w"]
    x_min, x_max = bev["x_min"], bev["x_max"]
    y_min, y_max = bev["y_min"], bev["y_max"]
    z_anchors = bev["z_anchors"]

    # BEV 셀 중심 좌표 (ego frame)
    # x: W방향 (left→right), y: H방향 (back→front)
    xs = torch.linspace(x_min + bev["cell_size"]/2,
                        x_max - bev["cell_size"]/2, W, device=device)
    ys = torch.linspace(y_min + bev["cell_size"]/2,
                        y_max - bev["cell_size"]/2, H, device=device)
    zs = torch.tensor(z_anchors, dtype=torch.float32, device=device)

    # grid: (H, W, n_z, 3)
    yy, xx, zz = torch.meshgrid(ys, xs, zs, indexing="ij")
    ref_pts = torch.stack([xx, yy, zz], dim=-1)   # (H, W, n_z, 3)

    return ref_pts


@torch.no_grad()
def project_bev_to_image(
    ref_pts_ego: torch.Tensor,   # (N_bev, n_z, 3)  ego frame
    cam_intrinsics: torch.Tensor, # (B, 6, 3, 3)
    cam_extrinsics: torch.Tensor, # (B, 6, 4, 4)  cam→ego
    img_h: int,
    img_w: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    3D 포인트(ego frame)를 카메라 이미지 평면에 투영.

    수식:
        p_cam = R_e2c @ p_ego + t_e2c   (cam←ego 역변환)
        [u, v, 1]^T = K @ p_cam / p_cam[2]

    Args:
        ref_pts_ego:    (N_bev, n_z, 3)  ego frame 3D 포인트
        cam_intrinsics: (B, 6, 3, 3)
        cam_extrinsics: (B, 6, 4, 4)  cam→ego SE3
    Returns:
        uv_norm:  (B, 6, N_bev*n_z, 2)  grid_sample용 [-1,1] 정규화 UV
        valid:    (B, 6, N_bev*n_z)      bool, 이미지 범위 내 + 카메라 앞
    """
    B, n_cam = cam_intrinsics.shape[:2]
    N_bev, n_z, _ = ref_pts_ego.shape
    N = N_bev * n_z
    device = ref_pts_ego.device

    # ref_pts: (1, 1, N, 3) → broadcast
    pts = ref_pts_ego.reshape(1, 1, N, 3)   # (1,1,N,3)

    # ego→cam 역변환: cam_extrinsics는 cam→ego
    # R_e2c = R_c2e^T,  t_e2c = -R_e2c @ t_c2e
    R_c2e = cam_extrinsics[:, :, :3, :3]    # (B,6,3,3)
    t_c2e = cam_extrinsics[:, :, :3,  3]    # (B,6,3)
    R_e2c = R_c2e.transpose(-1, -2)         # (B,6,3,3)
    t_e2c = -(R_e2c @ t_c2e.unsqueeze(-1)).squeeze(-1)  # (B,6,3)

    # p_cam: (B,6,N,3)
    # p_cam[b,c,n] = R_e2c[b,c] @ pts[n] + t_e2c[b,c]
    p_cam = (R_e2c.unsqueeze(2) @ pts.unsqueeze(-1)).squeeze(-1) + t_e2c.unsqueeze(2)
    # (B,6,1,3,3) @ (1,1,N,3,1) → (B,6,N,3)

    z_cam = p_cam[..., 2]                   # (B,6,N)

    # 투영: u = fx*x/z + cx, v = fy*y/z + cy
    K = cam_intrinsics   # (B,6,3,3)
    fx = K[:, :, 0, 0].unsqueeze(-1)  # (B,6,1)
    fy = K[:, :, 1, 1].unsqueeze(-1)
    cx = K[:, :, 0, 2].unsqueeze(-1)
    cy = K[:, :, 1, 2].unsqueeze(-1)

    u = fx * p_cam[..., 0] / (z_cam + 1e-6) + cx   # (B,6,N)
    v = fy * p_cam[..., 1] / (z_cam + 1e-6) + cy   # (B,6,N)

    # 유효 마스크: z>0 and u∈[0,W-1] and v∈[0,H-1]
    valid = (
        (z_cam > 0.1) &
        (u >= 0) & (u <= img_w - 1) &
        (v >= 0) & (v <= img_h - 1)
    )   # (B,6,N) bool

    # PyTorch grid_sample을 위한 [-1,1] 정규화
    u_norm = (u / (img_w - 1)) * 2 - 1   # (B,6,N)
    v_norm = (v / (img_h - 1)) * 2 - 1   # (B,6,N)

    uv_norm = torch.stack([u_norm, v_norm], dim=-1)   # (B,6,N,2)

    return uv_norm, valid
