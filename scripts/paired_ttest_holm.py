"""Paired t-tests on three-seed scalar metric scores with Holm--Bonferroni correction.

Reproduces Table 6. Family size = 24 per metric (6 model pairs x 4 operating points).
"""
from itertools import combinations
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

CSV = Path(__file__).resolve().parent.parent / "data" / "three_seed_results.csv"
MODELS = ["YOLOv8m", "YOLO11m", "YOLO26m", "YOLO26n"]
OPS = ["bull_zs", "bull_fs20", "bull_fs50", "bull_fs100"]


def holm_correct(pvals: list[float]) -> list[float]:
    n = len(pvals)
    order = np.argsort(pvals)
    adj = np.empty(n)
    running_max = 0.0
    for rank, idx in enumerate(order):
        adj_p = min(1.0, pvals[idx] * (n - rank))
        running_max = max(running_max, adj_p)
        adj[idx] = running_max
    return adj.tolist()


def collect(df: pd.DataFrame, metric: str) -> list[dict]:
    raw = []
    for op in OPS:
        for a, b in combinations(MODELS, 2):
            sa = df[(df.model == a) & (df.operating_point == op)].sort_values("seed")[metric].dropna().values
            sb = df[(df.model == b) & (df.operating_point == op)].sort_values("seed")[metric].dropna().values
            n = min(len(sa), len(sb))
            if n < 2:
                continue
            sa, sb = sa[:n], sb[:n]
            diff = sa - sb
            t, p = stats.ttest_rel(sa, sb)
            raw.append({"op": op, "pair": f"{a}-{b}", "mean_diff": round(float(diff.mean()), 4), "p": float(p)})
    if raw:
        adj = holm_correct([r["p"] for r in raw])
        for r, ap in zip(raw, adj):
            r["p_Holm"] = round(ap, 4)
            r["p"] = round(r["p"], 4)
    return raw


def main() -> None:
    df = pd.read_csv(CSV)
    for metric in ("mAP50", "mAP50_95"):
        print(f"\n=== {metric} ===")
        print(pd.DataFrame(collect(df, metric)).to_string(index=False))


if __name__ == "__main__":
    main()
