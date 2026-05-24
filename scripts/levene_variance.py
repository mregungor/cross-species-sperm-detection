"""Transfer-variance characterization: source vs zero-shot SD with Levene's test.

Reproduces Table 5 from data/three_seed_results.csv. For each model, compares the
three-seed source-domain mAP50-95 distribution against the three-seed bull
zero-shot mAP50 distribution under the null of equal variance.

Note: we use the classical mean-centered Levene statistic (center="mean")
rather than scipy's default Brown--Forsythe (center="median"), matching the
test reported in the paper. With n=3 the two centering choices can disagree
materially on the resulting p-value.
"""
from pathlib import Path
import pandas as pd
from scipy import stats

CSV = Path(__file__).resolve().parent.parent / "data" / "three_seed_results.csv"


def main() -> None:
    df = pd.read_csv(CSV)
    rows = []
    for model in df["model"].unique():
        src = df[(df.model == model) & (df.operating_point == "source_val")]["mAP50_95"].dropna()
        zs = df[(df.model == model) & (df.operating_point == "bull_zs")]["mAP50"].dropna()
        fs20 = df[(df.model == model) & (df.operating_point == "bull_fs20")]["mAP50"].dropna()
        if len(src) < 2 or len(zs) < 2:
            continue
        src_sd, zs_sd = src.std(ddof=1), zs.std(ddof=1)
        fs20_sd = fs20.std(ddof=1) if len(fs20) >= 2 else float("nan")
        ratio_zs_src = (zs_sd ** 2) / (src_sd ** 2) if src_sd > 0 else float("inf")
        ratio_zs_fs20 = (zs_sd ** 2) / (fs20_sd ** 2) if fs20_sd > 0 else float("inf")
        _, pval = stats.levene(src, zs, center="mean")
        rows.append({
            "model": model,
            "src_SD": round(src_sd, 4),
            "zs_SD": round(zs_sd, 4),
            "fs20_SD": round(fs20_sd, 4),
            "Var(zs)/Var(src)": round(ratio_zs_src, 1),
            "Var(zs)/Var(fs20)": round(ratio_zs_fs20, 1),
            "Levene_p": round(pval, 4),
        })
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
