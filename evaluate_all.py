"""
Phase 5 — 전체 실험 비교 평가.

학습된 모든 체크포인트를 평가하고 MOTA/MOTP/FP 비교표 + 시각화 생성.

Usage:
    python evaluate_all.py
    python evaluate_all.py --ckpt-root checkpoints
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
def evaluate_ckpt(ckpt_path: str, cfg: dict, device: torch.device,
                  use_radar: bool = True, score_thr: float = None) -> dict:
    """단일 체크포인트 평가. MOTA/MOTP/FP/IDSW + val_loss 반환."""
    model = BEVFormer(cfg).to(device).eval()
    head  = DetectionHead(cfg).to(device).eval()

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    head.load_state_dict(ckpt["head"])

    val_ds = BEVFormerDataset(cfg, cfg["data"]["val_scenes"],
                              transform=build_val_transform(cfg))
    loader = DataLoader(val_ds, batch_size=1, shuffle=False,
                        collate_fn=collate_fn,
                        num_workers=cfg["train"]["num_workers"])

    ev         = MOTEvaluator(threshold=cfg["eval"]["match_threshold"])
    nms_r      = cfg["eval"]["nms_radius"]
    if score_thr is None:
        score_thr = cfg["eval"]["score_threshold"]
    prev_bev   = prev_e2w = None

    for batch in loader:
        if batch["prev_token"][0] == "":
            prev_bev = prev_e2w = None

        imgs   = batch["imgs"].to(device)
        cam_K  = batch["cam_intrinsics"].to(device)
        cam_E  = batch["cam_extrinsics"].to(device)
        e2w    = batch["ego_to_world"].to(device)
        radar  = batch["radar_pts_ego"] if use_radar else None

        bev   = model(imgs, cam_K, cam_E, e2w, prev_bev=prev_bev,
                      prev_ego_to_world=prev_e2w, radar_pts_list=radar)
        preds = head(bev)

        prev_bev = bev.detach(); prev_e2w = e2w.detach()

        dets     = decode_predictions(preds, cfg, score_thr)[0]
        boxes_np = dets["boxes"].numpy()
        if len(boxes_np) > 0:
            keep     = center_nms(boxes_np, dets["scores"].numpy(), nms_r)
            boxes_np = boxes_np[keep]

        gt_boxes = batch["gt_boxes"][0].numpy()
        ev.update(boxes_np[:, :2].tolist() if len(boxes_np) else [],
                  list(range(len(boxes_np))),
                  gt_boxes[:, :2].tolist() if len(gt_boxes) else [],
                  list(range(len(gt_boxes))))

    result = ev.summary()
    result["val_loss"] = float(ckpt.get("val_loss", -1))
    result["epoch"]    = int(ckpt.get("epoch", -1))
    result["exp"]      = str(ckpt.get("exp", os.path.basename(os.path.dirname(ckpt_path))))
    return result


def find_checkpoints(ckpt_root: str) -> list:
    """체크포인트 디렉토리 탐색. [(label, path)] 반환."""
    entries = []

    # 구버전 단일 best.pt
    legacy = os.path.join(ckpt_root, "best.pt")
    if os.path.isfile(legacy):
        entries.append(("real_only (v0)", legacy))

    # 실험별 디렉토리
    for d in sorted(os.listdir(ckpt_root)):
        best = os.path.join(ckpt_root, d, "best.pt")
        if os.path.isfile(best):
            entries.append((d, best))

    return entries


def plot_comparison(results: dict, save_path: str):
    """실험 비교 바 차트 생성."""
    PMBM = {"MOTA": -0.177, "MOTP": 1.20, "FP/frame": 0.8}

    labels = list(results.keys())
    motas  = [results[k]["MOTA"]     for k in labels]
    motps  = [results[k]["MOTP"]     for k in labels]
    fps    = [results[k]["FP/frame"] for k in labels]

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    fig.suptitle("Experiment Comparison: Real vs Real+Synthetic", fontsize=12)

    colors = plt.cm.Set2(np.linspace(0, 1, len(labels)))

    for ax, values, title, baseline, better in zip(
        axes,
        [motas, motps, fps],
        ["MOTA (higher better)", "MOTP [m] (lower better)", "FP/frame (lower better)"],
        [PMBM["MOTA"], PMBM["MOTP"], PMBM["FP/frame"]],
        [True, False, False],
    ):
        bars = ax.bar(labels, values, color=colors, alpha=0.8, edgecolor="black")
        ax.axhline(baseline, color="red", linestyle="--", linewidth=1.5,
                   label=f"PMBM baseline ({baseline})")
        ax.set_title(title, fontsize=10)
        ax.set_ylabel("")
        ax.tick_params(axis="x", rotation=25, labelsize=8)
        ax.legend(fontsize=8)

        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"Plot saved: {save_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt-root", default="checkpoints")
    parser.add_argument("--score-thr", type=float, default=None,
                        help="score threshold override (default: config.yaml 값)")
    args = parser.parse_args()

    cfg    = load_cfg()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    score_thr = args.score_thr if args.score_thr is not None else cfg["eval"]["score_threshold"]
    print(f"Device: {device}  |  score_threshold: {score_thr}\n")

    ckpts = find_checkpoints(args.ckpt_root)
    if not ckpts:
        print("No checkpoints found.")
        return

    print(f"Found {len(ckpts)} checkpoint(s):")
    for label, path in ckpts:
        print(f"  [{label}]  {path}")

    # ── 평가 ──────────────────────────────────────────────────────────────────
    all_results = {}
    for label, path in ckpts:
        print(f"\nEvaluating [{label}]...")
        try:
            r = evaluate_ckpt(path, cfg, device, use_radar=True, score_thr=score_thr)
            all_results[label] = r
            print(f"  MOTA={r['MOTA']:.3f}  MOTP={r['MOTP']:.3f}  "
                  f"FP/f={r['FP/frame']:.1f}  ep={r['epoch']}  val_loss={r['val_loss']:.4f}")
        except Exception as e:
            print(f"  ERROR: {e}")

    # ── 비교 표 ───────────────────────────────────────────────────────────────
    PMBM = {"MOTA": -0.177, "MOTP": 1.20, "FP/frame": 0.8, "IDSW": 3}

    print("\n" + "=" * 70)
    print(f"{'Method':<30} {'MOTA':>7} {'MOTP':>7} {'FP/f':>7} {'IDSW':>6} {'ep':>4}")
    print("-" * 70)
    print(f"{'PMBM Phase29 (radar GT)':<30} "
          f"{PMBM['MOTA']:>7.3f} {PMBM['MOTP']:>7.2f} "
          f"{PMBM['FP/frame']:>7.1f} {PMBM['IDSW']:>6}")
    for label, r in all_results.items():
        print(f"{label:<30} {r['MOTA']:>7.3f} {r['MOTP']:>7.2f} "
              f"{r['FP/frame']:>7.1f} {r['IDSW']:>6} {r['epoch']:>4}")
    print("=" * 70)

    # ── 저장 ──────────────────────────────────────────────────────────────────
    os.makedirs("results", exist_ok=True)
    out_json = "results/all_experiments.json"
    with open(out_json, "w") as f:
        json.dump({"PMBM_baseline": PMBM, "experiments": all_results}, f, indent=2)
    print(f"\nSaved: {out_json}")

    if len(all_results) >= 1:
        plot_comparison(all_results, "results/experiment_comparison.png")


if __name__ == "__main__":
    main()
