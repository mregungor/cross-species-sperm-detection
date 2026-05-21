# Cross-Species Transfer Learning for Brightfield Spermatozoa Detection — Analysis and Statistical Pipeline

Repository: <https://github.com/mregungor/cross-species-sperm-detection>

This repository accompanies the paper:

> Güngör, E. *Cross-Species Transfer Learning for Brightfield Spermatozoa Detection: Characterising Zero-Shot Variance and Few-Shot Stabilisation on Bull Microscopy.* (Full venue and year will be added upon publication.)

It releases the **evaluation and statistical-analysis layer** of the study: the canonical per-seed scalar metric records together with the scripts that aggregate them into the manuscript's headline tables and figures (multi-seed means and SDs, Levene's test, paired *t*-tests with Holm--Bonferroni correction, and the learning-curve and variance-evolution figures). The training and data-preparation code is part of an ongoing research programme and is available from the corresponding author upon reasonable request; this repository is therefore not a full model-training reference implementation, but rather the reproducibility layer that lets a reader regenerate every numerical claim in the paper from the released CSV.

## What this repository contains

- `data/three_seed_results.csv` — scalar metric scores (mAP50, mAP50-95, precision, recall) for each combination of model × operating point × seed. This is the canonical, paper-of-record source for every numerical claim in the manuscript; all tables and figures listed below are derived from it.
- `scripts/aggregate_seeds.py` — produces Table 1 (source-domain baselines) and Table 3 (few-shot learning curve, both metrics) from the CSV.
- `scripts/levene_variance.py` — produces Table 4 (transfer-variance characterisation with the classical mean-centred Levene's test, matching the paper).
- `scripts/paired_ttest_holm.py` — produces Table 5 (paired *t*-tests with Holm--Bonferroni correction, family size 12 per metric). Sample sizes follow the multi-seed design: *n* = 3 at zero-shot and fs-20, *n* = 2 at fs-50 and fs-100.
- `scripts/plot_learning_curve.py` — produces Figure 1 (learning curve) and Figure 2 (variance evolution). The output files are named `figures/fig3_learning_curve.pdf` and `figures/fig4_variance_evolution.pdf` for historical reasons; the paper numbering is 1 and 2.

## Datasets

All datasets used in the paper are publicly available:

- **SVIA** — Chen et al., 2022.
- **EVISAN** — <https://zenodo.org/records/4303768>
- **VISEM-Tracking** — Thambawita et al., 2023.
- **DeepSperm** — Hidayatullah et al., 2021.

## Requirements

- Python 3.10+
- `pip install -r requirements.txt`

## Reproducing the paper's tables and figures

```bash
python scripts/aggregate_seeds.py       # -> Table 1 (source baselines) + Table 3 (few-shot curve)
python scripts/levene_variance.py       # -> Table 4 (variance characterisation)
python scripts/paired_ttest_holm.py     # -> Table 5 (paired t-tests)
python scripts/plot_learning_curve.py   # -> Figure 1 (learning curve) + Figure 2 (variance evolution)
```

The scripts read the canonical `data/three_seed_results.csv` and emit aggregated statistics or figures; no external data download is required to reproduce the analysis layer. The CSV itself is the per-seed scalar output of the training runs described in the paper; the raw training and data-preparation code is part of an ongoing research programme and is available from the corresponding author upon reasonable request.

## Citation

If you use this code or the accompanying analysis, please cite the paper above.

## License

MIT License — see `LICENSE`.
