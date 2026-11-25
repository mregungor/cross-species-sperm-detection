"""Generate manuscript Figure 1 (learning curve) and Figure 2 (variance evolution).

Reads from the canonical CSV (github/data/multi_seed_results.csv) so that the
figures always reflect the latest multi-seed data without an intermediate
statistics build step.
"""
from __future__ import annotations
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
CSV = ROOT / "data" / "multi_seed_results.csv"
OUT = ROOT / "figures"
OUT.mkdir(parents=True, exist_ok=True)

MODELS = ["YOLOv8m", "YOLO11m", "YOLO26m", "YOLO26n"]
COLORS = {"YOLOv8m": "#9467bd", "YOLO11m": "#1f77b4", "YOLO26m": "#d62728", "YOLO26n": "#2ca02c"}
MARKERS = {"YOLOv8m": "D", "YOLO11m": "o", "YOLO26m": "s", "YOLO26n": "^"}
FRAMES = [0, 20, 50, 100]
OP_BY_FRAMES = {0: "bull_zs", 20: "bull_fs20", 50: "bull_fs50", 100: "bull_fs100"}

df = pd.read_csv(CSV)


def stat(model: str, op: str, metric: str):
    s = df[(df.model == model) & (df.operating_point == op)][metric].dropna()
    if len(s) == 0:
        return None, None
    return s.mean(), (s.std(ddof=1) if len(s) > 1 else 0.0)


# ==========================================================================
# Figure 1: learning curve (mAP50 + mAP50-95)
# ==========================================================================
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharex=True)

for ax, metric, metric_label in [(axes[0], "mAP50", r"$\mathrm{mAP}_{50}$"),
                                  (axes[1], "mAP50_95", r"$\mathrm{mAP}_{50\text{-}95}$")]:
    for m in MODELS:
        ys, yerr = [], []
        for n in FRAMES:
            mean, sd = stat(m, OP_BY_FRAMES[n], metric)
            if mean is None:
                ys.append(np.nan); yerr.append(0)
            else:
                ys.append(mean); yerr.append(sd or 0)
        ax.errorbar(FRAMES, ys, yerr=yerr, marker=MARKERS[m], color=COLORS[m],
                    label=m, markersize=8, capsize=4, linewidth=2,
                    linestyle="-", alpha=0.9)
    ax.set_xlabel("Target-domain fine-tuning frames", fontsize=11)
    ax.set_ylabel(f"Bull {metric_label}", fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(FRAMES)
    ax.legend(loc="lower right", fontsize=10)

fig.tight_layout()
fig.savefig(OUT / "fig1_learning_curve.pdf", bbox_inches="tight", dpi=150)
fig.savefig(OUT / "fig1_learning_curve.png", bbox_inches="tight", dpi=150)
print(f"[saved] {OUT / 'fig1_learning_curve.pdf'}")
plt.close(fig)


# ==========================================================================
# Figure 2: variance evolution (per-model SD across regimes)
# ==========================================================================
fig, ax = plt.subplots(figsize=(9, 5))

regime_labels = ["Source\nvalidation\n(mAP50-95)", "Zero-shot\nbull\n(mAP50)",
                 "FS-20\nbull\n(mAP50)", "FS-50\nbull\n(mAP50)",
                 "FS-100\nbull\n(mAP50)"]
regime_specs = [("source_val", "mAP50_95"), ("bull_zs", "mAP50"),
                ("bull_fs20", "mAP50"), ("bull_fs50", "mAP50"),
                ("bull_fs100", "mAP50")]
x_pos = np.arange(len(regime_labels))

bar_w = 0.2
for i, m in enumerate(MODELS):
    sds = []
    for op, metric in regime_specs:
        _, sd = stat(m, op, metric)
        sds.append(sd if sd is not None else 0)
    offsets = (i - 1.5) * bar_w
    bars = ax.bar(x_pos + offsets, sds, bar_w, label=m,
                  color=COLORS[m], alpha=0.85, edgecolor="black", linewidth=0.5)
    for b, v in zip(bars, sds):
        if v > 1e-4:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.003,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=7)
        else:
            ax.text(b.get_x() + b.get_width() / 2, 0.002,
                    "—", ha="center", va="bottom", fontsize=8, color="gray")

ax.set_xticks(x_pos)
ax.set_xticklabels(regime_labels, fontsize=9)
ax.set_ylabel("Inter-seed standard deviation", fontsize=11)
ax.legend(loc="upper right", fontsize=10)
ax.grid(True, axis="y", alpha=0.3)
ax.set_axisbelow(True)
ax.set_ylim(0, 0.25)

fig.tight_layout()
fig.savefig(OUT / "fig2_variance_evolution.pdf", bbox_inches="tight", dpi=150)
fig.savefig(OUT / "fig2_variance_evolution.png", bbox_inches="tight", dpi=150)
print(f"[saved] {OUT / 'fig2_variance_evolution.pdf'}")
plt.close(fig)
print("Figures done.")
