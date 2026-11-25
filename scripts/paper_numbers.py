"""Print every number quoted in the Paper 1 manuscript from the canonical CSV.

Single source of truth for the CMPB text: tables, ranges, ratios, intervals, paired tests.
Usage: python scripts/paper_numbers.py
"""
from __future__ import annotations

from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

D = Path(__file__).resolve().parents[1] / "data"
df = pd.read_csv(D / "multi_seed_results.csv", dtype={"seed": str})
re_ = pd.read_csv(D / "bull_eval_maxdet300_vs_1000.csv", dtype={"seed": str})
re_["model"] = re_["model"].str.lower()
MODELS = ["YOLOv8m", "YOLO11m", "YOLO26m", "YOLO26n"]
OPS = ["bull_zs", "bull_fs20", "bull_fs50", "bull_fs100"]
FR = {"bull_zs": 0, "bull_fs20": 20, "bull_fs50": 50, "bull_fs100": 100}


def vals(model, op, metric):
    return df[(df.model == model) & (df.operating_point == op)].sort_values("seed")[metric].to_numpy()


def ms(x):
    return f"{x.mean():.3f} ± {x.std(ddof=1):.3f}"


print("=== Table 2 (source, unchanged) ===")
for m in MODELS:
    print(m, *(ms(vals(m, "source_val", k)) for k in ("mAP50", "mAP50_95", "precision", "recall")))

print("\n=== COCO-only, class-agnostic (Table 3) ===")
import json as _json
coco_ag = _json.load(open(D / "coco_only_class_agnostic.json"))
print({k: round(v["mAP50"], 4) for k, v in coco_ag.items()})

print("\n=== Table 3/4 learning curve ===")
for m in MODELS:
    for k in ("mAP50", "mAP50_95"):
        print(m, k, *(ms(vals(m, op, k)) for op in OPS))
for op in OPS[1:]:
    for k in ("mAP50", "mAP50_95"):
        best = max(MODELS, key=lambda m: vals(m, op, k).mean())
        print("leader", op, k, best)
zs_means = {m: vals(m, "bull_zs", "mAP50").mean() for m in MODELS}
print("BF contribution:", {m: round(zs_means[m] - coco_ag[m.lower()]["mAP50"], 3) for m in MODELS})
print("zs per-seed min/max:", {m: (round(vals(m, "bull_zs", "mAP50").min(), 3), round(vals(m, "bull_zs", "mAP50").max(), 3)) for m in MODELS})
for op in OPS:
    mm = [vals(m, op, "mAP50").mean() for m in MODELS]
    print(f"range {op} mAP50 {min(mm):.3f}-{max(mm):.3f}")

print("\n=== slopes and AUC (mAP50 means) ===")
for m in MODELS:
    y = [vals(m, op, "mAP50").mean() for op in OPS]
    x = [0, 20, 50, 100]
    auc = sum((x[i + 1] - x[i]) * (y[i] + y[i + 1]) / 2 for i in range(3))  # trapezoid; no NumPy-version dependency
    print(m, f"slope zs->fs20 {100 * (y[1] - y[0]):+.1f} pt", f"AUC {auc:.1f}")

print("\n=== Table 5 precision/recall ===")
for m in MODELS:
    for k in ("precision", "recall"):
        print(m, k, *(ms(vals(m, op, k)) for op in OPS))
for op in OPS:
    p = [vals(m, op, "precision").mean() for m in MODELS]; r = [vals(m, op, "recall").mean() for m in MODELS]
    print(f"{op}: precision {min(p):.3f}-{max(p):.3f}  recall {min(r):.3f}-{max(r):.3f}")

def pitman_morgan(x, y):
    """Pitman--Morgan test of Var(x) = Var(y) for paired measurements (the same five runs evaluated
    twice), with the 95% interval for Var(x)/Var(y). Returns p, lower, upper, corr(x, y)."""
    n = len(x)
    r = stats.pearsonr(x + y, x - y)[0]
    p = 2 * stats.t.sf(abs(r) * np.sqrt((n - 2) / (1 - r * r)), n - 2)
    rho = stats.pearsonr(x, y)[0]
    k = 1 + 2 * (1 - rho * rho) * stats.t.ppf(0.975, n - 2) ** 2 / (n - 2)
    ratio = x.var(ddof=1) / y.var(ddof=1)
    return p, ratio * (k - np.sqrt(k * k - 1)), ratio * (k + np.sqrt(k * k - 1)), rho


print("\n=== Table 6 variance (Pitman--Morgan paired test + Levene) ===")
for m in MODELS:
    s = vals(m, "source_val", "mAP50_95"); s50 = vals(m, "source_val", "mAP50")
    z = vals(m, "bull_zs", "mAP50"); f20 = vals(m, "bull_fs20", "mAP50")
    r = z.var(ddof=1) / s.var(ddof=1)
    print(m, f"src {s.std(ddof=1):.4f} zs {z.std(ddof=1):.4f} fs20 {f20.std(ddof=1):.4f}",
          f"zs/src {r:.0f}x  PM p {pitman_morgan(z, s)[0]:.4f} 95% CI [{pitman_morgan(z, s)[1]:.1f}, {pitman_morgan(z, s)[2]:.0f}]x"
          f" r(src,zs) {pitman_morgan(z, s)[3]:+.2f}  zs/fs20 {z.var(ddof=1) / f20.var(ddof=1):.0f}x",
          f"like-for-like mAP50 {z.var(ddof=1) / s50.var(ddof=1):.0f}x (Levene p {stats.levene(s50, z, center='mean').pvalue:.3f})",
          f"Levene p {stats.levene(s, z, center='mean').pvalue:.3f}")
    # Levene over every 3-seed subset of the five seeds
    ps = []
    seeds = sorted(df[df.operating_point == "source_val"].seed.unique(), key=int)
    for sub in combinations(seeds, 3):
        ss = df[(df.model == m) & (df.operating_point == "source_val") & df.seed.isin(sub)]["mAP50_95"]
        zz = df[(df.model == m) & (df.operating_point == "bull_zs") & df.seed.isin(sub)]["mAP50"]
        ps.append(stats.levene(ss, zz, center="mean").pvalue)
    print(f"   3-seed Levene p range {min(ps):.3f}-{max(ps):.3f}, significant in {sum(p < 0.05 for p in ps)}/10")
for op in ("bull_fs20", "bull_fs50", "bull_fs100"):
    print(op, "SD mAP50:", {m: round(vals(m, op, "mAP50").std(ddof=1), 4) for m in MODELS})
print("\n=== Supplementary Table S4 (mAP50-95 both domains) ===")
for m in MODELS:
    s95, z95 = vals(m, "source_val", "mAP50_95"), vals(m, "bull_zs", "mAP50_95")
    print(m, f"ratio {z95.var(ddof=1) / s95.var(ddof=1):.1f}x  PM p {pitman_morgan(z95, s95)[0]:.3f}",
          f"Levene p {stats.levene(s95, z95, center='mean').pvalue:.3f}",
          f"CV ratio zs/src {(z95.std(ddof=1) / z95.mean()) / (s95.std(ddof=1) / s95.mean()):.1f}")
lo, hi = np.sqrt(4 / stats.chi2.ppf(0.975, 4)), np.sqrt(4 / stats.chi2.ppf(0.025, 4))
z26 = vals("YOLO26m", "bull_zs", "mAP50").std(ddof=1)
print(f"chi2 SD factor [{lo:.2f}, {hi:.2f}]  YOLO26m zs SD CI [{z26 * lo:.3f}, {z26 * hi:.3f}]")

print("\n=== Table 7 subset ===")
for m in MODELS:
    init = vals(m, "bull_fs20", "mAP50")
    orig = df[(df.model == m) & (df.operating_point == "bull_fs20") & (df.seed == "42")]["mAP50"].tolist()
    draws = np.array(orig + df[(df.model == m) & (df.operating_point == "bull_fs20_subset")]["mAP50"].tolist())
    zs = vals(m, "bull_zs", "mAP50")
    print(m, f"init {init.std(ddof=1):.4f} subset {draws.std(ddof=1):.4f} ratio {draws.std(ddof=1) / init.std(ddof=1):.1f}x zs {zs.std(ddof=1):.4f} zs/subset {zs.std(ddof=1) / draws.std(ddof=1):.1f}x")

print("\n=== Table 8: Welch t-tests (primary) and paired t-tests (Supplementary Table S3), Holm over 24 ===")
res = {}
for k in ("mAP50", "mAP50_95"):
    rows = []
    for op in OPS:
        for a, b in combinations(MODELS, 2):
            xa, xb = vals(a, op, k), vals(b, op, k)
            rows.append([op, a, b, (xa - xb).mean(), stats.ttest_ind(xa, xb, equal_var=False).pvalue, stats.ttest_rel(xa, xb).pvalue,
                         stats.pearsonr(xa, xb)[0]])
    def _holm(p):
        p = np.array(p); order = np.argsort(p); adj = np.empty(len(p)); run = 0
        for rank, i in enumerate(order):
            run = max(run, min(1, (len(p) - rank) * p[i])); adj[i] = run
        return adj
    w, pr = _holm([r[4] for r in rows]), _holm([r[5] for r in rows])
    res[k] = [r[:4] + [wi, pi, r[6]] for r, wi, pi in zip(rows, w, pr)]
short = {"YOLOv8m": "v8m", "YOLO11m": "11m", "YOLO26m": "26m", "YOLO26n": "26n"}
for i in range(24):
    a, b = res["mAP50"][i], res["mAP50_95"][i]
    fmt = lambda v: "<0.001" if v < 0.001 else f"{v:.3f}"
    print(f"{a[0]:10s} {short[a[1]]}-{short[a[2]]}  {a[3]:+.3f} / {b[3]:+.3f}  Welch {fmt(a[4])} {fmt(b[4])}"
          + ("  *50" if a[4] < 0.05 else "") + ("  *5095" if b[4] < 0.05 else "")
          + f"   | paired {fmt(a[5])} {fmt(b[5])}  r(mAP50-95) {b[6]:+.2f}")

print("\n=== imgsz=1280 ablation zero-shot (max_det=1000) ===")
hr = re_[(re_.operating_point == "bull_zs_img1280") & (re_.max_det == 1000)]
print(hr[["model", "seed", "mAP50"]].to_string(index=False))
for m in ("yolo11m", "yolov8m"):
    s42 = re_[(re_.operating_point == "bull_zs") & (re_.max_det == 1000) & (re_.model == m) & (re_.seed == "42")]["mAP50"].iloc[0]
    print(m, f"seed42 @640 {s42:.3f}")


print("\n=== Baselines (bull test) ===")
import json as _json
cls = _json.load(open(D / "classical_baseline_bull.json"))
print("classical config", cls["selected_config"], "test", {k: round(v, 3) for k, v in cls["test"].items() if isinstance(v, float)})
coco_ag = _json.load(open(D / "coco_only_class_agnostic.json"))
print("COCO class-agnostic mAP50", {k: round(v["mAP50"], 4) for k, v in coco_ag.items()})
for m in MODELS:
    for op in ("bull_zs", "bull_fs20"):
        p, r = vals(m, op, "precision"), vals(m, op, "recall")
        f1 = 2 * p * r / (p + r)
        print(m, op, f"F1 {f1.mean():.3f} ± {f1.std(ddof=1):.3f}")

print("\n=== Subset vs initialization variance, one-sided F test (df 3,4) ===")
for m in MODELS:
    init = vals(m, "bull_fs20", "mAP50")
    sub = np.array(df[(df.model == m) & (df.operating_point == "bull_fs20") & (df.seed == "42")]["mAP50"].tolist()
                   + df[(df.model == m) & (df.operating_point == "bull_fs20_subset")]["mAP50"].tolist())
    F_ = sub.var(ddof=1) / init.var(ddof=1)
    print(m, f"F {F_:.1f} p {1 - stats.f.cdf(F_, 3, 4):.3f}")

print("\n=== fs-100 mAP50 spread (full precision) ===")
mm = [vals(m, "bull_fs100", "mAP50").mean() for m in ("YOLOv8m", "YOLO11m", "YOLO26m")]
print(f"{max(mm) - min(mm):.4f}")

print("\n=== Welch vs paired, headline fs-20 YOLO26m vs YOLO11m mAP50-95 ===")
a, b = vals("YOLO26m", "bull_fs20", "mAP50_95"), vals("YOLO11m", "bull_fs20", "mAP50_95")
print(f"paired p {stats.ttest_rel(a, b).pvalue:.2e}  Welch p {stats.ttest_ind(a, b, equal_var=False).pvalue:.2e}")
