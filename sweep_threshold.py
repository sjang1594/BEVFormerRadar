"""
score_threshold 스윕 — 단일 체크포인트 기준.

forward pass 1회로 raw detection을 캐싱, 각 threshold별 MOTA를 계산.

Usage:
    python sweep_threshold.py
    python sweep_threshold.py --ckpt checkpoints/real+synth_r3/epoch_015.pt
    python sweep_threshold.py --thresholds 1e-5 1e-4 1e-3 1e-2 0.1
"""

import io, sys
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import os, yaml, json, argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.data.dataset import BEVFormerDataset, collate_fn
from src.data.transforms import build_val_transform
from src.model.bevformer import BEVFormer
from src.model.detection_head import DetectionHead
from src.evaluation.nms import decode_predictions, center_nms
from src.evaluation.mot_evaluator import MOTEvaluator


def load_cfg():
    with open("config.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


@torch.no_grad()
def collect_raw(ckpt_path, cfg, device):
    """forward pass 1회 — 모든 프레임의 raw (boxes, scores, gt_boxes) 캐싱."""
    model = BEVFormer(cfg).to(device).eval()
    head  = DetectionHead(cfg).to(device).eval()
    ckpt  = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    head.load_state_dict(ckpt["head"])
    print(f"  Loaded: ep={ckpt.get('epoch')}  val_loss={ckpt.get('val_loss', -1):.4f}")

    val_ds = BEVFormerDataset(cfg, cfg["data"]["val_scenes"],
                              transform=build_val_transform(cfg))
    loader = DataLoader(val_ds, batch_size=1, shuffle=False,
                        collate_fn=collate_fn,
                        num_workers=cfg["train"]["num_workers"])

    frames = []   # [(boxes_np, scores_np, gt_boxes_np)]
    prev_bev = prev_e2w = None

    for batch in loader:
        if batch["prev_token"][0] == "":
            prev_bev = prev_e2w = None

        imgs  = batch["imgs"].to(device)
        cam_K = batch["cam_intrinsics"].to(device)
        cam_E = batch["cam_extrinsics"].to(device)
        e2w   = batch["ego_to_world"].to(device)
        radar = batch["radar_pts_ego"]

        bev   = model(imgs, cam_K, cam_E, e2w, prev_bev=prev_bev,
                      prev_ego_to_world=prev_e2w, radar_pts_list=radar)
        preds = head(bev)
        prev_bev = bev.detach(); prev_e2w = e2w.detach()

        # 최대한 낮은 threshold로 모든 후보 수집
        dets = decode_predictions(preds, cfg, 1e-7)[0]
        frames.append((
            dets["boxes"].numpy(),
            dets["scores"].numpy(),
            batch["gt_boxes"][0].numpy(),
        ))

    return frames


def evaluate_at_threshold(frames, cfg, thr):
    ev    = MOTEvaluator(threshold=cfg["eval"]["match_threshold"])
    nms_r = cfg["eval"]["nms_radius"]

    for boxes_np, scores_np, gt_boxes in frames:
        mask = scores_np >= thr
        b = boxes_np[mask]
        s = scores_np[mask]
        if len(b) > 0:
            keep = center_nms(b, s, nms_r)
            b    = b[keep]

        ev.update(b[:, :2].tolist() if len(b) else [],
                  list(range(len(b))),
                  gt_boxes[:, :2].tolist() if len(gt_boxes) else [],
                  list(range(len(gt_boxes))))

    return ev.summary()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", default="checkpoints/real+synth_r3/epoch_015.pt")
    parser.add_argument("--thresholds", nargs="+", type=float,
                        default=[1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 0.1, 0.2])
    args = parser.parse_args()

    cfg    = load_cfg()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")
    print(f"Ckpt   : {args.ckpt}\n")

    # ── forward pass 1회 ──────────────────────────────────────────────────────
    print("Forward pass (collecting raw detections)...")
    frames = collect_raw(args.ckpt, cfg, device)
    print(f"  Frames: {len(frames)}")

    all_scores = np.concatenate([s for _, s, _ in frames])
    print(f"  Raw detections (thr=1e-7): {len(all_scores)}")
    if len(all_scores):
        for p in [50, 75, 90, 95, 99, 99.9]:
            print(f"    p{p:4.1f}: {np.percentile(all_scores, p):.2e}")

    # ── threshold sweep ────────────────────────────────────────────────────────
    results = []
    print(f"\n{'thr':>10} {'MOTA':>8} {'MOTP':>8} {'FP/f':>8} {'IDSW':>6} {'det/f':>8}")
    print("-" * 58)

    for thr in args.thresholds:
        r = evaluate_at_threshold(frames, cfg, thr)
        avg_det = np.mean([np.sum(s >= thr) for _, s, _ in frames])
        results.append({"thr": thr, **r, "avg_det_per_frame": float(avg_det)})
        print(f"  {thr:>8.1e}  {r['MOTA']:>8.3f}  {r['MOTP']:>8.3f}  "
              f"{r['FP/frame']:>8.1f}  {r['IDSW']:>6}  {avg_det:>8.1f}")

    best = max(results, key=lambda x: x["MOTA"])
    print(f"\nBest: thr={best['thr']:.1e}  MOTA={best['MOTA']:.3f}  "
          f"FP/f={best['FP/frame']:.1f}  IDSW={best['IDSW']}")

    # ── 저장 & 플롯 ────────────────────────────────────────────────────────────
    os.makedirs("results", exist_ok=True)
    with open("results/threshold_sweep.json", "w") as f:
        json.dump(results, f, indent=2)

    thrs  = [r["thr"]               for r in results]
    motas = [r["MOTA"]              for r in results]
    fps   = [r["FP/frame"]          for r in results]
    idsws = [r["IDSW"]              for r in results]
    dets  = [r["avg_det_per_frame"] for r in results]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(f"Threshold Sweep — {os.path.basename(args.ckpt)}", fontsize=11)

    for ax, vals, title in zip(axes.flat,
        [motas, fps, idsws, dets],
        ["MOTA (↑)", "FP/frame (↓)", "IDSW (↓)", "det/frame"]):
        ax.semilogx(thrs, vals, marker="o", linewidth=1.5)
        ax.axvline(best["thr"], color="red", linestyle="--", alpha=0.7,
                   label=f"best thr={best['thr']:.1e}")
        ax.set_xlabel("score_threshold (log)")
        ax.set_title(title, fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("results/threshold_sweep.png", dpi=120, bbox_inches="tight")
    plt.close()
    print("Plot saved: results/threshold_sweep.png")


if __name__ == "__main__":
    main()
