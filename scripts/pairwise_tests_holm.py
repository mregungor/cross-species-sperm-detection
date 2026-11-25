"""Pairwise architecture comparisons with Holm--Bonferroni correction (manuscript Table 8, Supplementary Table S3).

Primary test (Table 8): Welch's unpaired t-test on the five per-seed scores of each architecture.
Runs of different architectures that share a seed number start from different weights and show no
consistent correlation, so pairing by seed number removes no shared variance. The paired t-test matched
by seed number is printed alongside for comparison (Supplementary Table S3), with the Pearson
correlation between the two architectures across seed numbers.

Family size = 24 per metric (6 model pairs x 4 operating points).
"""
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

CSV = Path(__file__).resolve().parent.parent / "data" / "multi_seed_results.csv"
MODELS = ["YOLOv8m", "YOLO11m", "YOLO26m", "YOLO26n"]
OPS = ["bull_zs", "bull_fs20", "bull_fs50", "bull_fs100"]


def holm_correct(pvals):
    n = len(pvals)
    order = np.argsort(pvals)
    adj = np.empty(n)
    running_max = 0.0
    for rank, idx in enumerate(order):
        running_max = max(running_max, min(1.0, pvals[idx] * (n - rank)))
        adj[idx] = running_max
    return adj


def collect(df, metric):
    rows = []
    for op in OPS:
        for a, b in combinations(MODELS, 2):
            sa = df[(df.model == a) & (df.operating_point == op)].sort_values("seed")[metric].to_numpy()
            sb = df[(df.model == b) & (df.operating_point == op)].sort_values("seed")[metric].to_numpy()
            rows.append({
                "op": op, "pair": f"{a}-{b}", "mean_diff": round(float((sa - sb).mean()), 4),
                "p_welch": stats.ttest_ind(sa, sb, equal_var=False).pvalue,
                "p_paired": stats.ttest_rel(sa, sb).pvalue,
                "r_seed": round(float(stats.pearsonr(sa, sb)[0]), 2),
            })
    out = pd.DataFrame(rows)
    out["p_Holm_welch"] = holm_correct(out.p_welch.to_numpy()).round(4)
    out["p_Holm_paired"] = holm_correct(out.p_paired.to_numpy()).round(4)
    out["p_welch"] = out.p_welch.round(4)
    out["p_paired"] = out.p_paired.round(4)
    return out


def main():
    df = pd.read_csv(CSV, dtype={"seed": str})
    for metric in ("mAP50", "mAP50_95"):
        print(f"\n=== {metric} ===")
        print(collect(df, metric).to_string(index=False))


if __name__ == "__main__":
    main()
