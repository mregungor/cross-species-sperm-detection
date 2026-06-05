# Cross-Species Transfer Learning for Brightfield Spermatozoa Detection — Analysis and Statistical Pipeline

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
<!-- [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.XXXXXXX.svg)](https://doi.org/10.5281/zenodo.XXXXXXX) -->
<!-- Uncomment the DOI badge above and replace XXXXXXX after the first Zenodo release is minted. -->

Repository: <https://github.com/mregungor/cross-species-sperm-detection>

This repository accompanies the paper:

> Güngör, E. *Cross-Species Transfer Learning for Brightfield Spermatozoa Detection: Characterizing Zero-Shot Variance and Few-Shot Stabilization on Bull Microscopy.* (Full venue and year will be added upon publication.)

It releases the **evaluation and statistical-analysis layer** of the study: the canonical per-seed scalar metric records together with the scripts that aggregate them into the manuscript's headline tables and figures (multi-seed means and SDs, Levene's test, paired *t*-tests with Holm--Bonferroni correction, and the learning-curve and variance-evolution figures). The training and data-preparation code is part of an ongoing research program and is available from the corresponding author upon reasonable request. This repository is therefore not a full model-training reference implementation; it is the reproducibility layer that lets a reader regenerate every numerical claim in the paper from the released per-seed metric records.

## Repository structure

```
cross-species-sperm-detection/
├── data/
│   └── multi_seed_results.csv     # Canonical per-seed metric records (4 architectures, 88 rows; five source-training seeds)
├── scripts/
│   ├── aggregate_seeds.py         # Tables 2 and 4 (source baselines, few-shot curve)
│   ├── levene_variance.py         # Table 5 (variance characterization)
│   ├── paired_ttest_holm.py       # Table 6 (paired t-tests with Holm correction)
│   ├── subset_variance.py         # Subset-selection variance (Section 5.9 table)
│   └── plot_learning_curve.py     # Figures 1 and 2 (learning curve, variance evolution)
├── provenance_example/
│   ├── RunReport.json             # Example per-run provenance record (redacted paths)
│   └── split_manifest.json        # Example per-run split manifest with SHA-256 hashes
├── figures/                       # Generated outputs (fig1_*, fig2_*)
├── requirements.txt
├── LICENSE
└── README.md
```

## What this repository contains

- `data/multi_seed_results.csv` — scalar metric scores (mAP50, mAP50-95, precision, recall) for each combination of model × operating point × seed. This is the canonical, paper-of-record source for every numerical claim in the manuscript; all tables and figures listed below are derived from it.
- `scripts/aggregate_seeds.py` — produces Table 2 (source-domain baselines) and Table 4 (few-shot learning curve, both metrics) from the CSV.
- `scripts/levene_variance.py` — produces Table 5 (transfer-variance characterization with the classical mean-centered Levene's test, matching the paper).
- `scripts/paired_ttest_holm.py` — produces Table 6 (paired *t*-tests with Holm--Bonferroni correction, family size 24 per metric: 6 model pairs × 4 operating points). Sample sizes follow the multi-seed design: *n* = 5 at zero-shot and fs-20, *n* = 2 at fs-50 and fs-100.
- `scripts/subset_variance.py` — produces the subset-selection variance table (manuscript Section 5.9): per architecture, the fs-20 subset-selection SD (four 20-frame draws at fixed initialization) versus the five-seed initialization SD, with the zero-shot SD for reference.
- `scripts/plot_learning_curve.py` — produces Figure 1 (learning curve, `figures/fig1_learning_curve.pdf`) and Figure 2 (variance evolution, `figures/fig2_variance_evolution.pdf`).
- `provenance_example/` — two illustrative artifacts emitted per training run, as described in the manuscript (Sections 4.1–4.2): `split_manifest.json` (per-split SHA-256 hashes, frame counts, and the random seed) and `RunReport.json` (training hyperparameters, deterministic flag, seed, Python/CUDA/GPU environment identification, and final aggregate metrics). Absolute machine paths and the hostname are redacted to placeholders; hashes, seed, hyperparameters, and metrics are the real recorded values. These are shown as a concrete example of the per-run provenance format; the complete set of records, the full environment snapshots, and the training code are available from the corresponding author on reasonable academic request.

## Datasets

All datasets used in the paper are publicly available:

- **SVIA** — Chen et al., 2022. <https://github.com/Demozsj/Detection-Sperm>
- **EVISAN** — Yan, 2020. <https://zenodo.org/records/4303768>
- **VISEM-Tracking** — Thambawita et al., 2023. <https://datasets.simula.no/visem-tracking/>
- **DeepSperm** — Hidayatullah et al., 2021. Bull spermatozoa detection benchmark (available from the original authors' repository or on request).

Note: this repository does **not** require dataset access to reproduce the statistical analysis. All numerical claims in the paper are derived from the released per-seed metric records in `data/multi_seed_results.csv`. Re-running the training pipeline from raw data is out of scope for this release.

## Requirements

- Python 3.10 or later
- Core dependencies:
  - `numpy >= 1.24`
  - `pandas >= 2.0`
  - `scipy >= 1.11` (provides `scipy.stats.levene` and `scipy.stats.ttest_rel` used in the paper's Section 3.4)
  - `matplotlib >= 3.7` (for figure generation)

Install all dependencies with:

```bash
pip install -r requirements.txt
```

## Reproducing the paper's tables and figures

```bash
python scripts/aggregate_seeds.py       # -> Table 2 (source baselines) + Table 4 (few-shot curve)
python scripts/levene_variance.py       # -> Table 5 (variance characterization)
python scripts/paired_ttest_holm.py     # -> Table 6 (paired t-tests)
python scripts/subset_variance.py       # -> subset-selection variance table (Section 5.9)
python scripts/plot_learning_curve.py   # -> Figure 1 (learning curve) + Figure 2 (variance evolution)
```

The scripts read the canonical `data/multi_seed_results.csv` and emit aggregated statistics or figures; no external data download is required. The CSV is the per-seed scalar output of the training runs described in the paper.

## Expected outputs

Running `python scripts/levene_variance.py` should produce output matching **Table 5** of the manuscript:

```
  model  src_SD  zs_SD  fs20_SD  Var(zs)/Var(src)  Var(zs)/Var(fs20)  Levene_p
YOLO11m  0.0058 0.1761   0.0053             907.4             1112.0    0.0033
YOLO26m  0.0091 0.0549   0.0007              36.7             5922.4    0.0901
YOLO26n  0.0085 0.1527   0.0094             321.7              261.5    0.0890
YOLOv8m  0.0074 0.1361   0.0096             334.3              200.5    0.0624
```

Running `python scripts/paired_ttest_holm.py` prints one table per metric (`=== mAP50 ===` and `=== mAP50_95 ===`); among the rows, the manuscript's headline pairwise comparison (YOLO26m vs YOLO11m at the 20-frame operating point) appears in the `mAP50_95` table as:

```
        op            pair  mean_diff      p  p_Holm
 bull_fs20 YOLO11m-YOLO26m    -0.0735 0.0001  0.0032
```

If any reproduced value differs materially, please open an issue with your Python and SciPy versions.

## Citation

If you use this code, the released metric records, or build on the statistical analysis methodology, please cite both the manuscript and the archived release:

### Manuscript (TO BE UPDATED)

```bibtex
@article{gungor2026crossspecies,
  title   = {Cross-Species Transfer Learning for Brightfield Spermatozoa Detection:
             Characterizing Zero-Shot Variance and Few-Shot Stabilization on Bull Microscopy},
  author  = {G{\"u}ng{\"o}r, Emre},
  journal = {Manuscript under review},
  year    = {2026}
}
```

### Software / data archive (TO BE UPDATED)

```bibtex
@software{gungor2026repo,
  author  = {G{\"u}ng{\"o}r, Emre},
  title   = {cross-species-sperm-detection: Analysis and statistical pipeline},
  year    = {2026},
  version = {v1.0.0},
  doi     = {10.5281/zenodo.XXXXXXX},
  url     = {https://github.com/mregungor/cross-species-sperm-detection}
}
```

## Questions and feedback

For methodological questions about the paper, please contact the corresponding author (see paper). For issues with the released code or CSV (e.g., reproduction discrepancies, dependency conflicts), please open a GitHub issue.

## License

MIT License — see `LICENSE`.
