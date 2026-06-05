"""Subset-selection variance (paper Section 5.9, Table tab:subset).

Compares, per architecture, the fs-20 subset-selection SD (four 20-frame draws at
a fixed initialization seed: the original seed-42 subset plus three alternatives
labelled by seeds 101/202/303 under operating_point ``bull_fs20_subset'') against
the initialization SD (five-seed fs-20 SD at the fixed original subset). The
zero-shot SD is shown for reference. All on bull validation mAP50.
"""
from pathlib import Path
import pandas as pd
import numpy as np

CSV = Path(__file__).resolve().parent.parent / "data" / "three_seed_results.csv"


def main() -> None:
    df = pd.read_csv(CSV)
    rows = []
    for model in sorted(df["model"].unique()):
        init_sd = df[(df.model == model) & (df.operating_point == "bull_fs20")]["mAP50"].std(ddof=1)
        zs_sd = df[(df.model == model) & (df.operating_point == "bull_zs")]["mAP50"].std(ddof=1)
        orig42 = df[(df.model == model) & (df.operating_point == "bull_fs20") & (df.seed == 42)]["mAP50"]
        alts = df[(df.model == model) & (df.operating_point == "bull_fs20_subset")]["mAP50"].tolist()
        if orig42.empty or len(alts) < 2:
            continue
        draws = [orig42.iloc[0]] + alts
        subset_sd = float(np.std(draws, ddof=1))
        rows.append({
            "model": model,
            "init_SD": round(init_sd, 4),
            "subset_SD": round(subset_sd, 4),
            "subset/init": round(subset_sd / init_sd, 1) if init_sd > 0 else float("inf"),
            "zs_SD": round(zs_sd, 4),
        })
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
