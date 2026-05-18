"""
Mixed Training — Real(nuScenes) + Synthetic(3DGS) 데이터 혼합 학습.

Usage:
    python train_mixed.py --synth-dir PATH --ratio 1.0
    python train_mixed.py --synth-dir PATH --ratio 3.0
    python train_mixed.py  # real only (baseline)

비교 실험:
  Exp A: --ratio 0   → real only (baseline)
  Exp B: --ratio 1   → 1:1
  Exp C: --ratio 3   → 1:3
"""

import io, sys
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import os, yaml, math, argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.data.dataset import BEVFormerDataset, collate_fn
from src.data.transforms import build_train_transform, build_val_transform
from src.data.synthetic_dataset import SyntheticDataset
from src.data.mixed_dataset import MixedDataset
from src.data.radar_aug_dataset import RadarAugDataset
from src.model.bevformer import BEVFormer
from src.model.detection_head import DetectionHead
from src.loss.detection_loss import detection_loss


def load_cfg():
    with open("config.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def cosine_lr(optimizer, epoch, total_epochs, base_lr):
    lr = base_lr * 0.5 * (1 + math.cos(math.pi * epoch / total_epochs))
    for pg in optimizer.param_groups:
        pg["lr"] = lr
    return lr


def train_epoch(model, head, loader, optimizer, cfg, device, epoch):
    model.train(); head.train()
    prev_bev = prev_e2w = None
    total_loss = 0.0; n = 0

    for batch in loader:
        is_first = batch["prev_token"][0] == ""
        if is_first:
            prev_bev = prev_e2w = None

        imgs   = batch["imgs"].to(device)
        cam_K  = batch["cam_intrinsics"].to(device)
        cam_E  = batch["cam_extrinsics"].to(device)
        e2w    = batch["ego_to_world"].to(device)
        radar  = batch["radar_pts_ego"]

        bev = model(imgs, cam_K, cam_E, e2w,
                    prev_bev=prev_bev, prev_ego_to_world=prev_e2w,
                    radar_pts_list=radar)
        preds  = head(bev)
        losses = detection_loss(preds, batch, cfg)

        optimizer.zero_grad()
        losses["total"].backward()
        nn.utils.clip_grad_norm_(
            list(model.parameters()) + list(head.parameters()),
            cfg["train"]["grad_clip_norm"])
        optimizer.step()

        prev_bev = bev.detach()
        prev_e2w = e2w.detach()
        total_loss += losses["total"].item(); n += 1

        if n % 50 == 0:
            print(f"  [ep{epoch} {n}/{len(loader)}] "
                  f"total={losses['total'].item():.4f} "
                  f"hm={losses['heatmap'].item():.4f}")

    return total_loss / max(n, 1)


@torch.no_grad()
def val_epoch(model, head, loader, cfg, device):
    model.eval(); head.eval()
    prev_bev = prev_e2w = None
    total = 0.0; n = 0

    for batch in loader:
        if batch["prev_token"][0] == "":
            prev_bev = prev_e2w = None

        imgs  = batch["imgs"].to(device)
        cam_K = batch["cam_intrinsics"].to(device)
        cam_E = batch["cam_extrinsics"].to(device)
        e2w   = batch["ego_to_world"].to(device)

        bev    = model(imgs, cam_K, cam_E, e2w, prev_bev=prev_bev,
                       prev_ego_to_world=prev_e2w,
                       radar_pts_list=batch["radar_pts_ego"])
        preds  = head(bev)
        losses = detection_loss(preds, batch, cfg)

        prev_bev = bev.detach(); prev_e2w = e2w.detach()
        total += losses["total"].item(); n += 1

    return total / max(n, 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--synth-dir", default=None,
                        help="합성 데이터 .npz 디렉토리 (없으면 real only)")
    parser.add_argument("--radar-aug-dir", default=None,
                        help="radar_aug .npz 디렉토리 (real cam + aug radar, Exp D)")
    parser.add_argument("--ratio", type=float, default=1.0,
                        help="synthetic/real 비율 (0=real only, 1=1:1, 3=1:3)")
    parser.add_argument("--real-fraction", type=float, default=1.0,
                        help="real 데이터 사용 비율 (0~1, scene 단위 subset)")
    parser.add_argument("--ckpt-root", default="checkpoints",
                        help="체크포인트 저장 루트 디렉토리")
    args = parser.parse_args()

    cfg    = load_cfg()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── 데이터셋 ──────────────────────────────────────────────────────────────
    all_scenes  = cfg["data"]["train_scenes"]
    n_scenes    = max(1, round(len(all_scenes) * args.real_fraction))
    use_scenes  = all_scenes[:n_scenes]
    frac_label  = f"s{n_scenes}"   # e.g. s3 = 3 scenes out of 6

    real_train = BEVFormerDataset(cfg, use_scenes,
                                  transform=build_train_transform(cfg))
    val_ds     = BEVFormerDataset(cfg, cfg["data"]["val_scenes"],
                                  transform=build_val_transform(cfg))

    if args.radar_aug_dir and os.path.isdir(args.radar_aug_dir):
        # Exp D: real camera + radar augmentation (3DGS 카메라 없음)
        aug_ds    = RadarAugDataset(args.radar_aug_dir, cfg, augment=True)
        train_ds  = MixedDataset(real_train, aug_ds, ratio=args.ratio)
        exp_label = f"real+radar_aug_r{args.ratio:.0f}_{frac_label}"
    elif args.synth_dir and args.ratio > 0 and os.path.isdir(args.synth_dir):
        synth_ds  = SyntheticDataset(args.synth_dir, cfg, augment=True)
        train_ds  = MixedDataset(real_train, synth_ds, ratio=args.ratio)
        exp_label = f"real+synth_r{args.ratio:.0f}_{frac_label}"
    else:
        train_ds  = real_train
        exp_label = f"real_only_{frac_label}"

    # real_fraction=1.0 (기본값)이면 scene suffix 제거 (하위 호환)
    if args.real_fraction >= 1.0:
        exp_label = exp_label.replace("_s6", "")

    train_loader = DataLoader(train_ds, batch_size=cfg["train"]["batch_size"],
                              shuffle=False, collate_fn=collate_fn,
                              num_workers=cfg["train"]["num_workers"])
    val_loader   = DataLoader(val_ds, batch_size=1, shuffle=False,
                              collate_fn=collate_fn,
                              num_workers=cfg["train"]["num_workers"])

    print(f"Experiment : {exp_label}")
    print(f"Train size : {len(train_ds)}  |  Val size: {len(val_ds)}")

    # ── 모델 ──────────────────────────────────────────────────────────────────
    model = BEVFormer(cfg).to(device)
    head  = DetectionHead(cfg).to(device)
    params = [p for p in list(model.parameters()) + list(head.parameters())
              if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=cfg["train"]["lr"],
                                   weight_decay=cfg["train"]["weight_decay"])

    # ── 학습 ──────────────────────────────────────────────────────────────────
    ckpt_dir = f"{args.ckpt_root}/{exp_label}"
    os.makedirs(ckpt_dir, exist_ok=True)
    best_val  = float("inf")
    epochs    = cfg["train"]["epochs"]
    base_lr   = cfg["train"]["lr"]

    for epoch in range(1, epochs + 1):
        lr = cosine_lr(optimizer, epoch - 1, epochs, base_lr)
        print(f"\n[Epoch {epoch}/{epochs}]  lr={lr:.2e}  exp={exp_label}")

        train_loss = train_epoch(model, head, train_loader, optimizer,
                                  cfg, device, epoch)
        val_loss   = val_epoch(model, head, val_loader, cfg, device)
        print(f"  Train={train_loss:.4f}  Val={val_loss:.4f}")

        if val_loss < best_val:
            best_val = val_loss
            torch.save({"model": model.state_dict(), "head": head.state_dict(),
                        "epoch": epoch, "val_loss": val_loss,
                        "exp": exp_label},
                       f"{ckpt_dir}/best.pt")
            print(f"  [Best] {val_loss:.4f}")

        if epoch % cfg["train"]["checkpoint_interval"] == 0:
            torch.save({"model": model.state_dict(), "head": head.state_dict(),
                        "epoch": epoch},
                       f"{ckpt_dir}/epoch_{epoch:03d}.pt")

    print(f"\nDone. Best val loss: {best_val:.4f}  [{exp_label}]")


if __name__ == "__main__":
    main()
