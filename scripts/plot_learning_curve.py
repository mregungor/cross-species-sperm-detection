"""Plot the bull few-shot learning curve (paper Figure 1) and variance evolution (paper Figure 2).

Reads data/three_seed_results.csv and writes figures/fig1_learning_curve.pdf and
figures/fig2_variance_evolution.pdf.
"""
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
CSV = ROOT / "data" / "three_seed_results.csv"
FIG = ROOT / "figures"
FIG.mkdir(exist_ok=True)

MODELS = ["YOLOv8m", "YOLO11m", "YOLO26m", "YOLO26n"]
OP_TO_FRAMES = {"bull_zs": 0, "bull_fs20": 20, "bull_fs50": 50, "bull_fs100": 100}


def _agg(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    sub = df[df.operating_point.isin(OP_TO_FRAMES)].copy()
    sub["frames"] = sub.operating_point.map(OP_TO_FRAMES)
    return sub.groupby(["model", "frames"])[metric].agg(["mean", "std"]).reset_index()


def plot_learning_curve(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharex=True)
    for metric, ax in zip(("mAP50", "mAP50_95"), axes):
        agg = _agg(df, metric)
        for model in MODELS:
            m = agg[agg.model == model].sort_values("frames")
            ax.errorbar(m.frames, m["mean"], yerr=m["std"], marker="o", capsize=3, label=model)
        ax.set_xlabel("Target frames"); ax.set_ylabel(metric); ax.grid(alpha=0.3); ax.legend()
    fig.tight_layout(); fig.savefig(FIG / "fig1_learning_curve.pdf"); plt.close(fig)


def plot_variance_evolution(df: pd.DataFrame) -> None:
    regimes = [("source_val", "mAP50_95", "source"),
               ("bull_zs", "mAP50", "zs"),
               ("bull_fs20", "mAP50", "fs-20"),
               ("bull_fs50", "mAP50", "fs-50"),
               ("bull_fs100", "mAP50", "fs-100")]
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    for model in MODELS:
        sds = []
        for op, metric, _ in regimes:
            vals = df[(df.model == model) & (df.operating_point == op)][metric].dropna()
            sds.append(vals.std(ddof=1) if len(vals) >= 2 else None)
        ax.plot([r[2] for r in regimes], sds, marker="s", label=model)
    ax.set_ylabel("Inter-seed SD"); ax.set_yscale("log"); ax.grid(alpha=0.3, which="both"); ax.legend()
    fig.tight_layout(); fig.savefig(FIG / "fig2_variance_evolution.pdf"); plt.close(fig)


def main() -> None:
    df = pd.read_csv(CSV)
    plot_learning_curve(df)
    plot_variance_evolution(df)
    print(f"Wrote {FIG/'fig1_learning_curve.pdf'} and {FIG/'fig2_variance_evolution.pdf'}")


if __name__ == "__main__":
    main()
