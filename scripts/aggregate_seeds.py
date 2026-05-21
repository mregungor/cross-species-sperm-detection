"""Aggregate three-seed metric scores into mean +/- SD per (model, operating point).

Reproduces Table 1 (source-domain baselines) and Table 3 (few-shot learning curve)
from data/three_seed_results.csv.
"""
from pathlib import Path
import pandas as pd

CSV = Path(__file__).resolve().parent.parent / "data" / "three_seed_results.csv"


def main() -> None:
    df = pd.read_csv(CSV)
    metrics = ["mAP50", "mAP50_95", "precision", "recall"]
    agg = (
        df.groupby(["operating_point", "model"])[metrics]
        .agg(["mean", "std", "count"])
        .round(4)
    )
    print(agg)


if __name__ == "__main__":
    main()
