"""Transfer-variance characterization: source vs zero-shot SD (manuscript Table 6, Supplementary Table S4).

For each model, compares the five-seed source-domain mAP50-95 with the five-seed bull zero-shot mAP50
of the same checkpoints. Because both are measured on the same five runs, the primary test is the
Pitman--Morgan test for paired variances, which also gives the 95% interval of the variance ratio.
Levene's test (classical mean-centered statistic, center="mean", not scipy's default Brown--Forsythe)
is reported alongside; it is robust to non-normality but treats the two samples as independent.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

CSV = Path(__file__).resolve().parent.parent / "data" / "multi_seed_results.csv"
MODELS = ["YOLOv8m", "YOLO11m", "YOLO26m", "YOLO26n"]


def pitman_morgan(x, y):
    """Paired test of Var(x) = Var(y); returns p, 95% interval of Var(x)/Var(y), and corr(x, y)."""
    n = len(x)
    r = stats.pearsonr(x + y, x - y)[0]
    p = 2 * stats.t.sf(abs(r) * np.sqrt((n - 2) / (1 - r * r)), n - 2)
    rho = stats.pearsonr(x, y)[0]
    k = 1 + 2 * (1 - rho * rho) * stats.t.ppf(0.975, n - 2) ** 2 / (n - 2)
    ratio = x.var(ddof=1) / y.var(ddof=1)
    return p, ratio * (k - np.sqrt(k * k - 1)), ratio * (k + np.sqrt(k * k - 1)), rho


def main() -> None:
    df = pd.read_csv(CSV, dtype={"seed": str})
    vals = lambda m, op, k: df[(df.model == m) & (df.operating_point == op)].sort_values("seed")[k].to_numpy()
    for title, src_metric, zs_metric in (("Table 6: source mAP50-95 vs bull mAP50", "mAP50_95", "mAP50"),
                                         ("Supplementary Table S4: mAP50-95 in both domains", "mAP50_95", "mAP50_95")):
        rows = []
        for m in MODELS:
            src, zs, fs20 = vals(m, "source_val", src_metric), vals(m, "bull_zs", zs_metric), vals(m, "bull_fs20", zs_metric)
            p_pm, lo, hi, rho = pitman_morgan(zs, src)
            rows.append({
                "model": m,
                "src_SD": round(src.std(ddof=1), 4),
                "zs_SD": round(zs.std(ddof=1), 4),
                "fs20_SD": round(fs20.std(ddof=1), 4),
                "Var(zs)/Var(src)": round(zs.var(ddof=1) / src.var(ddof=1), 1),
                "CI95_low": round(lo, 1),
                "Var(zs)/Var(fs20)": round(zs.var(ddof=1) / fs20.var(ddof=1), 1),
                "PitmanMorgan_p": round(p_pm, 4),
                "Levene_p": round(stats.levene(src, zs, center="mean").pvalue, 4),
            })
        print(f"\n{title}")
        print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
