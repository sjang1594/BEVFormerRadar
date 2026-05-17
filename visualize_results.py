"""
실험 결과 시각화 — 로컬 실행용 (데이터셋 불필요)
python visualize_results.py
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import numpy as np

# ── 색상 ──────────────────────────────────────────────────────────────────────
C_PMBM   = "#2ecc71"
C_REAL   = "#3498db"
C_SYNTH  = "#e74c3c"
C_GRAY   = "#95a5a6"
C_ARROW  = "#2c3e50"

# ── 데이터 ────────────────────────────────────────────────────────────────────
methods  = ["PMBM\n(upper bound)", "Exp A\nreal-only", "Exp B\nreal+synth 1:1"]
colors   = [C_PMBM, C_REAL, C_SYNTH]
motas    = [-0.177, -0.048, -0.054]
fps      = [0.8,    2.8,    2.2  ]
idsws    = [3,      13,     3    ]

thr_vals  = [1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 0.1, 0.2]
mota_real = [-10.289, -10.288, -10.288, -10.287, -10.284, -10.263,
             -10.180, -2.468, -0.048, 0.000]
mota_r1   = [-8.504, -8.502, -8.500, -8.496, -8.492, -8.476,
             -8.467, -8.259, -0.054, -0.004]

fig = plt.figure(figsize=(16, 12))
fig.patch.set_facecolor("#f8f9fa")
gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.38)

# ── 1. 파이프라인 다이어그램 ───────────────────────────────────────────────────
ax0 = fig.add_subplot(gs[0, :])
ax0.set_xlim(0, 10); ax0.set_ylim(0, 3.2)
ax0.axis("off")
ax0.set_facecolor("#f8f9fa")
ax0.set_title("Synthetic Data Pipeline — What Actually Improves IDSW",
              fontsize=13, fontweight="bold", pad=8)

def box(ax, x, y, w, h, color, text, fontsize=9):
    ax.add_patch(mpatches.FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.05",
        facecolor=color, edgecolor="white", linewidth=1.5, zorder=3))
    ax.text(x + w/2, y + h/2, text, ha="center", va="center",
            fontsize=fontsize, fontweight="bold", color="white",
            zorder=4, wrap=True)

def arrow(ax, x1, y1, x2, y2):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="->", color=C_ARROW,
                                lw=1.8), zorder=5)

# 노드
box(ax0, 0.2, 1.8, 2.0, 1.0, "#7f8c8d", "nuScenes\nCamera + LiDAR\n(real data)", 8)
box(ax0, 0.2, 0.3, 2.0, 1.0, "#7f8c8d", "nuScenes\nRadar GT Boxes\n+ Velocities", 8)

box(ax0, 3.0, 1.8, 2.0, 1.0, "#8e44ad", "3DGS\nScene Fitting\n(PSNR 30.94 dB)", 8)
box(ax0, 3.0, 0.3, 2.0, 1.0, "#d35400", "dynamic_compositor\nreal Doppler\n(GT velocity)", 8)

box(ax0, 6.0, 1.8, 1.8, 1.0, "#8e44ad", "Novel View\nCamera Images\n(domain gap ❌)", 8)
box(ax0, 6.0, 0.3, 1.8, 1.0, "#27ae60", "Synthetic Radar\nw/ correct Doppler\n(physics ✅)", 8)

box(ax0, 8.3, 1.0, 1.5, 1.0, "#2980b9",
    "BEVFormer\nTraining\n(Exp B)", 8)

# 화살표
arrow(ax0, 2.2, 2.3, 3.0, 2.3)
arrow(ax0, 2.2, 0.8, 3.0, 0.8)
arrow(ax0, 5.0, 2.3, 6.0, 2.3)
arrow(ax0, 5.0, 0.8, 6.0, 0.8)
arrow(ax0, 7.8, 2.3, 8.3, 1.7)
arrow(ax0, 7.8, 0.8, 8.3, 1.3)

# 라벨
ax0.text(6.9, 1.62, "❌ FP↑", color="#c0392b", fontsize=8, ha="center", va="center",
         fontweight="bold")
ax0.text(6.9, 0.18, "✅ IDSW↓", color="#27ae60", fontsize=8, ha="center", va="center",
         fontweight="bold")

# ── 2. MOTA 비교 ──────────────────────────────────────────────────────────────
ax1 = fig.add_subplot(gs[1, 0])
ax1.set_facecolor("white")
bars = ax1.bar(methods, motas, color=colors, edgecolor="white",
               linewidth=1.5, width=0.5)
ax1.axhline(0, color=C_GRAY, linewidth=0.8, linestyle="--")
ax1.set_title("MOTA  (↑ higher is better)", fontsize=10, fontweight="bold")
ax1.set_ylabel("MOTA")
for bar, val in zip(bars, motas):
    ax1.text(bar.get_x() + bar.get_width()/2,
             bar.get_height() - 0.005,
             f"{val:.3f}", ha="center", va="top",
             fontsize=9, fontweight="bold", color="white")
ax1.set_ylim(-0.25, 0.05)
ax1.tick_params(axis="x", labelsize=8)

# ── 3. IDSW 비교 ─────────────────────────────────────────────────────────────
ax2 = fig.add_subplot(gs[1, 1])
ax2.set_facecolor("white")
bars2 = ax2.bar(methods, idsws, color=colors, edgecolor="white",
                linewidth=1.5, width=0.5)
ax2.set_title("IDSW  (↓ lower is better)", fontsize=10, fontweight="bold")
ax2.set_ylabel("ID Switches")
for bar, val in zip(bars2, idsws):
    ax2.text(bar.get_x() + bar.get_width()/2,
             bar.get_height() + 0.3,
             str(val), ha="center", va="bottom",
             fontsize=11, fontweight="bold", color=C_ARROW)
# -77% 화살표
ax2.annotate("", xy=(2, 3), xytext=(1, 13),
             arrowprops=dict(arrowstyle="->", color=C_SYNTH, lw=2))
ax2.text(1.6, 8, "−77%", color=C_SYNTH, fontsize=11,
         fontweight="bold", ha="center")
ax2.set_ylim(0, 18)
ax2.tick_params(axis="x", labelsize=8)

# ── 4. Threshold sweep ────────────────────────────────────────────────────────
ax3 = fig.add_subplot(gs[1, 2])
ax3.set_facecolor("white")
ax3.semilogx(thr_vals, mota_real, marker="o", color=C_REAL,
             linewidth=2, label="real-only", markersize=5)
ax3.semilogx(thr_vals, mota_r1,  marker="s", color=C_SYNTH,
             linewidth=2, label="real+synth r1", markersize=5)
ax3.axhline(-0.177, color=C_PMBM, linestyle="--",
            linewidth=1.5, label="PMBM upper bound")
ax3.axvline(0.1, color=C_GRAY, linestyle=":", linewidth=1.5,
            label="optimal thr=0.1")
ax3.set_title("MOTA vs Score Threshold", fontsize=10, fontweight="bold")
ax3.set_xlabel("score_threshold (log scale)")
ax3.set_ylabel("MOTA")
ax3.legend(fontsize=7.5)
ax3.grid(True, alpha=0.3)
ax3.set_ylim(-12, 0.2)

# ── 전체 제목 ─────────────────────────────────────────────────────────────────
fig.suptitle(
    "NeuralSensorSim + BEVFormerRadar — Experiment Results\n"
    "Synthetic Doppler-correct radar reduces IDSW by 77% (13→3), matching PMBM level",
    fontsize=12, fontweight="bold", y=0.98)

plt.savefig("results/experiment_summary.png", dpi=150,
            bbox_inches="tight", facecolor=fig.get_facecolor())
plt.show()
print("Saved: results/experiment_summary.png")
