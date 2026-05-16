"""
epoch별 MOTA curve 분석.

모든 epoch_*.pt 체크포인트를 순회하며 MOTA/FP/IDSW를 평가하고 curve 플롯.
ep5 스냅샷 표도 출력.

Usage:
    python evaluate_epochs.py
    python evaluate_epochs.py --score-thr 0.3
    python evaluate_epochs.py --exps real_only real+synth_r1 real+synth_r3
    python evaluate_epochs.py --score-thr 0.3 --exps real_only
"""

import io, sys
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import os, yaml, json, argparse, glob
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
def evaluate_ckpt(ckpt_path, cfg, device, score_thr):
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

    ev    = MOTEvaluator(threshold=cfg["eval"]["match_threshold"])
    nms_r = cfg["eval"]["nms_radius"]
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
    result["epoch"]    = int(ckpt.get("epoch", -1))
    result["val_loss"] = float(ckpt.get("val_loss", -1))
    return result


def find_epoch_ckpts(ckpt_root, exp_names):
    """각 실험의 epoch_*.pt 목록 반환. {exp: [path, ...]} 오름차순."""
    result = {}
    for exp in exp_names:
        exp_dir = os.path.join(ckpt_root, exp)
        if not os.path.isdir(exp_dir):
            print(f"  [skip] {exp_dir} not found")
            continue
        paths = sorted(glob.glob(os.path.join(exp_dir, "epoch_*.pt")))
        if paths:
            result[exp] = paths
        else:
            print(f"  [skip] no epoch_*.pt in {exp_dir}")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt-root", default="checkpoints")
    parser.add_argument("--score-thr", type=float, default=None,
                        help="score threshold override (default: config.yaml 값)")
    parser.add_argument("--exps", nargs="+",
                        default=["real_only", "real+synth_r1", "real+synth_r3"])
    args = parser.parse_args()

    cfg    = load_cfg()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    score_thr = args.score_thr if args.score_thr is not None else cfg["eval"]["score_threshold"]

    print(f"Device    : {device}")
    print(f"score_thr : {score_thr}")
    print(f"Exps      : {args.exps}\n")

    ckpt_map = find_epoch_ckpts(args.ckpt_root, args.exps)
    if not ckpt_map:
        print("체크포인트 없음.")
        return

    # ── 평가 ──────────────────────────────────────────────────────────────────
    all_results = {}   # {exp: [result_dict, ...]}
    for exp, paths in ckpt_map.items():
        print(f"[{exp}]  {len(paths)} epoch checkpoints")
        results = []
        for path in paths:
            ep_num = int(os.path.basename(path).replace("epoch_", "").replace(".pt", ""))
            print(f"  ep{ep_num:03d} ...", end=" ", flush=True)
            r = evaluate_ckpt(path, cfg, device, score_thr)
            r["epoch_num"] = ep_num
            results.append(r)
            print(f"MOTA={r['MOTA']:.3f}  FP/f={r['FP/frame']:.1f}  IDSW={r['IDSW']}")
        all_results[exp] = results

    # ── ep5 스냅샷 표 ──────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"Epoch-5 스냅샷  (score_thr={score_thr})")
    print(f"{'Exp':<25} {'MOTA':>7} {'MOTP':>7} {'FP/f':>7} {'IDSW':>6}")
    print(f"{'-'*70}")
    for exp, results in all_results.items():
        ep5 = next((r for r in results if r["epoch_num"] == 5), None)
        if ep5:
            print(f"{exp:<25} {ep5['MOTA']:>7.3f} {ep5['MOTP']:>7.2f} "
                  f"{ep5['FP/frame']:>7.1f} {ep5['IDSW']:>6}")

    # ── best MOTA 표 ───────────────────────────────────────────────────────────
    print(f"\nBest-MOTA 체크포인트  (score_thr={score_thr})")
    print(f"{'Exp':<25} {'MOTA':>7} {'MOTP':>7} {'FP/f':>7} {'IDSW':>6} {'ep':>4}")
    print(f"{'-'*70}")
    for exp, results in all_results.items():
        best = max(results, key=lambda r: r["MOTA"])
        print(f"{exp:<25} {best['MOTA']:>7.3f} {best['MOTP']:>7.2f} "
              f"{best['FP/frame']:>7.1f} {best['IDSW']:>6} {best['epoch_num']:>4}")
    print(f"{'='*70}")

    # ── 저장 ──────────────────────────────────────────────────────────────────
    os.makedirs("results", exist_ok=True)
    tag = f"thr{score_thr:.4f}"
    out_json = f"results/epoch_sweep_{tag}.json"
    with open(out_json, "w") as f:
        json.dump({"score_thr": score_thr, "experiments": all_results}, f, indent=2)
    print(f"\nSaved: {out_json}")

    # ── MOTA curve 플롯 ────────────────────────────────────────────────────────
    metrics = [
        ("MOTA",     "MOTA (↑)"),
        ("FP/frame", "FP/frame (↓)"),
        ("IDSW",     "IDSW (↓)"),
    ]
    colors = {
        "real_only":      "tab:blue",
        "real+synth_r1":  "tab:orange",
        "real+synth_r3":  "tab:green",
    }

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(f"MOTA curve per epoch  (score_thr={score_thr})", fontsize=12)

    for ax, (metric, title) in zip(axes, metrics):
        for exp, results in all_results.items():
            epochs = [r["epoch_num"] for r in results]
            vals   = [r[metric]      for r in results]
            ax.plot(epochs, vals, marker="o", label=exp,
                    color=colors.get(exp), linewidth=1.5)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Epoch")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = f"results/mota_curve_{tag}.png"
    plt.savefig(plot_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"Plot saved: {plot_path}")


if __name__ == "__main__":
    main()
