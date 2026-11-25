"""Aggregate multi-seed metric scores into mean +/- SD per (model, operating point).

Reproduces Table 2 (source-domain baselines), Table 4 (few-shot learning curve), and
Table 5 (bull precision and recall)
from data/multi_seed_results.csv.
"""
from pathlib import Path
import pandas as pd

CSV = Path(__file__).resolve().parent.parent / "data" / "multi_seed_results.csv"


def main() -> None:
    df = pd.read_csv(CSV)
    df["F1"] = 2 * df.precision * df.recall / (df.precision + df.recall)  # per seed, then averaged
    metrics = ["mAP50", "mAP50_95", "precision", "recall", "F1"]
    agg = (
        df.groupby(["operating_point", "model"])[metrics]
        .agg(["mean", "std", "count"])
        .round(4)
    )
    print(agg)


if __name__ == "__main__":
    main()
