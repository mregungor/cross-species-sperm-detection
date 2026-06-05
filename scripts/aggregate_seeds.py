"""Aggregate multi-seed metric scores into mean +/- SD per (model, operating point).

Reproduces Table 2 (source-domain baselines) and Table 4 (few-shot learning curve)
from data/multi_seed_results.csv.
"""
from pathlib import Path
import pandas as pd

CSV = Path(__file__).resolve().parent.parent / "data" / "multi_seed_results.csv"


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
