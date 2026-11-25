# Cross-Species Transfer Learning for Brightfield Spermatozoa Detection — Evaluation and Statistical Analysis Code

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20785515.svg)](https://doi.org/10.5281/zenodo.20785515)

Repository: <https://github.com/mregungor/cross-species-sperm-detection>

This repository accompanies the paper:

> Güngör, E. *Cross-Species Transfer Learning for Brightfield Spermatozoa Detection: Characterizing Zero-Shot Variance and Few-Shot Stabilization on Bull Microscopy.* (Full venue and year will be added upon publication.)

It releases two layers of the study:

1. **Statistical analysis** (`scripts/`, `data/`): the canonical per-seed metric records and the scripts that turn them into every multi-seed table, statistical test, figure, and number in the manuscript. These run without any image data.
2. **Evaluation** (`evaluation/`, `data/bull_splits/`): the DeepSperm split lists and the scripts that produced the bull evaluation records — the re-evaluation of every checkpoint at 300 and 1000 detections per image, the deduplicated-test check, the tuned classical baseline, and the class-agnostic COCO-only baseline. These need the public DeepSperm data and the trained checkpoints.

The training and data-preparation code and the trained weights are part of an ongoing research program and are available from the corresponding author upon reasonable request.

## Repository structure

```
cross-species-sperm-detection/
├── data/
│   ├── multi_seed_results.csv           # Canonical per-seed metric records (4 architectures, 112 rows; five seeds)
│   ├── bull_eval_maxdet300_vs_1000.csv  # Every bull checkpoint evaluated at max_det=300 and 1000 (Table 9)
│   ├── coco_only_class_agnostic.json    # COCO-only baseline, class-agnostic (Table 3)
│   ├── classical_baseline_bull.json     # Tuned classical baseline on the bull test split (Table 5, Supplementary S2)
│   └── bull_splits/                     # DeepSperm frame lists: 220 test, 20/50/100 training, 20 validation, 3 alternative 20-frame subsets
├── scripts/
│   ├── aggregate_seeds.py               # Tables 2, 4, 5 (means and SDs incl. F1)
│   ├── variance_tests.py                # Table 6 and Supplementary Table S4 (Pitman--Morgan and Levene tests)
│   ├── subset_variance.py               # Table 7 (subset-selection variance)
│   ├── pairwise_tests_holm.py           # Table 8 (Welch t-tests) and Supplementary Table S3 (paired t-tests)
│   ├── plot_figures.py                  # Figures 1 and 2 as in the manuscript
│   └── paper_numbers.py                 # Every multi-seed number quoted in the manuscript text
├── evaluation/
│   ├── bull_splits.py                   # Loads data/bull_splits/ lists and checks them against the manifest SHA-256
│   ├── reeval_bull_maxdet.py            # Evaluates all checkpoints on the test split at 300 and 1000 detections
│   ├── eval_bull_dedup.py               # Re-evaluation without test frames that have a pixel-identical copy in training
│   ├── classical_baseline_bull.py       # Grid-searched thresholding baseline (1,110 configurations)
│   └── eval_coco_class_agnostic.py      # COCO-only baseline with single_cls=True
├── provenance_example/
│   ├── RunReport.json                   # Example source-training provenance record (redacted paths)
│   ├── split_manifest.json              # Human source-domain train/val/test manifest (SHA-256 hashes)
│   └── split_manifest_bull.json         # Bull (DeepSperm) split manifest: 100-frame pool / 20-frame val / 220-frame test
├── figures/                             # Generated outputs (fig1_*, fig2_*)
├── requirements.txt
├── LICENSE
└── README.md
```

## Data

- `data/multi_seed_results.csv` — mAP50, mAP50-95, precision, and recall for each model × operating point × seed; the paper-of-record source for every numerical claim. Source-domain rows (`source_val`) are on the human validation split. All bull rows (`bull_zs`, `bull_fs20/50/100`, `bull_fs20_subset`) are on the 220-frame held-out DeepSperm test split, evaluated with at most 1000 detections per image (the densest test frame has 568 annotated cells) and at batch size 8, so that no batch exceeded the non-maximum-suppression time limit (see *Evaluation settings*). Every fine-tuning seed `s` starts from the source checkpoint trained with seed `s`; the subset-selection rows use source and fine-tuning seed 42.
- `data/bull_eval_maxdet300_vs_1000.csv` — every bull checkpoint (zero-shot, few-shot, subset-selection, COCO-only, imgsz-1280) evaluated at `max_det=300` and `max_det=1000`, batch size 8 (2 for the imgsz-1280 checkpoints). The `max_det=1000` rows equal the bull rows of the canonical CSV. Its `bull_coco_only` rows use the class-0 ("person") mapping, kept for comparability with the training-time logs.
- `data/coco_only_class_agnostic.json` — the COCO-only baseline of Table 3 (`single_cls=True`, so any COCO class counts as a detection).
- `data/classical_baseline_bull.json` — the selected classical configuration and its precision, recall, and F1 on the test split.
- `data/bull_splits/*.txt` — the DeepSperm frames of each split, as paths relative to the project root (`data/raw/deepsperm_bull/...`). `evaluation/bull_splits.py` checks every list against the SHA-256 recorded in `provenance_example/split_manifest_bull.json`.

## Datasets

All datasets used in the paper are publicly available:

- **SVIA** — Chen et al., 2022. <https://github.com/Demozsj/Detection-Sperm>
- **EVISAN** — Yan, 2020. <https://zenodo.org/records/4303768>
- **VISEM-Tracking** — Thambawita et al., 2023. <https://datasets.simula.no/visem-tracking/>
- **DeepSperm** — Hidayatullah et al., 2021. Bull spermatozoa detection benchmark (available from the original authors' repository).

## Requirements

- Python 3.10 or later
- Statistical analysis: `numpy >= 1.24`, `pandas >= 2.0`, `scipy >= 1.11`, `matplotlib >= 3.7`
- Evaluation scripts additionally: `ultralytics` (8.4 was used), `opencv-python`, and a CUDA GPU

```bash
pip install -r requirements.txt
```

## Reproducing the paper's tables and figures

```bash
python scripts/aggregate_seeds.py       # -> Tables 2, 4, 5 (source baselines, few-shot curve, precision/recall/F1)
python scripts/variance_tests.py        # -> Table 6 and Supplementary Table S4 (variance tests)
python scripts/subset_variance.py       # -> Table 7 (subset-selection variance)
python scripts/pairwise_tests_holm.py   # -> Table 8 (Welch) and Supplementary Table S3 (paired)
python scripts/plot_figures.py          # -> Figure 1 (learning curve) + Figure 2 (variance evolution)
python scripts/paper_numbers.py         # -> every multi-seed number quoted in the text
```

The evaluation scripts in `evaluation/` read the frame lists in `data/bull_splits/` and expect a project root holding the raw DeepSperm data in `data/raw/deepsperm_bull/` (as distributed by the original authors), the COCO weights (`yolov8m.pt`, ...), and the trained checkpoints and outputs under `paper1_detection_segmentation/results/`. Set the environment variable `SPERM_PROJECT_ROOT` to that root before running them, e.g. `SPERM_PROJECT_ROOT=/path/to/project python evaluation/classical_baseline_bull.py`.

## Expected outputs

`python scripts/variance_tests.py` reproduces **Table 6** (first block):

```
  model  src_SD  zs_SD  fs20_SD  Var(zs)/Var(src)  CI95_low  Var(zs)/Var(fs20)  PitmanMorgan_p  Levene_p
YOLOv8m  0.0074 0.1616   0.0152             471.1      36.7              112.8          0.0002    0.0465
YOLO11m  0.0058 0.2171   0.0088            1378.9      94.5              611.8          0.0001    0.0047
YOLO26m  0.0091 0.0875   0.0113              93.2       9.0               59.9          0.0019    0.0665
YOLO26n  0.0085 0.1919   0.0415             508.3      56.8               21.4          0.0001    0.0683
```

Source and zero-shot values come from the same five checkpoints, so the Pitman--Morgan test for paired variances is the primary test; Levene's test treats the two samples as independent and is shown alongside.

`python scripts/pairwise_tests_holm.py` prints one table per metric; the YOLO26m vs YOLO11m comparison at 20 frames appears in the `mAP50_95` table as:

```
        op            pair  mean_diff  p_welch  p_paired  r_seed  p_Holm_welch  p_Holm_paired
 bull_fs20 YOLO11m-YOLO26m    -0.0682   0.0001    0.0026   -0.15        0.0023         0.0524
```

Runs of different architectures that share a seed number start from different weights and show no consistent correlation (`r_seed`), so the manuscript uses the Welch test; the paired test is reported for comparison.

If any reproduced value differs materially, please open an issue with your Python and SciPy versions.

## Evaluation settings

DeepSperm frames are dense: 97 of the 220 test frames contain more than 300 annotated cells (maximum 568). Two library defaults distort detection metrics on such frames without stopping the evaluation:

1. **Per-image detection cap.** Ultralytics returns at most `max_det=300` detections per image and truncates silently, which bounds recall at 0.737 on this split. All bull rows use `max_det=1000`. Under the default, 100-frame mAP50 would read 0.69–0.72 instead of 0.93–0.96 (manuscript Table 9).
2. **NMS time limit.** When post-processing of a batch exceeds the time limit, the remaining boxes are dropped with only a log warning. This happened in YOLOv8m/YOLO11m evaluations at batch size 16; all bull rows were evaluated at batch size 8 with no such warning. The NMS-free YOLO26 head is unaffected.

In Ultralytics 8.4, `val()` resets the detection limit of the YOLO26 end-to-end head on every call, but `predict()` sets it only at the first call on a model object; a later `predict()` with a larger `max_det` silently keeps the old limit. Use a fresh `YOLO(...)` instance when changing `max_det` for inference.

## Repeated images in DeepSperm

DeepSperm contains 39 groups of pixel-identical frames; with the frame-level split (`provenance_example/split_manifest_bull.json`), 22 of the 220 test frames have an identical copy in the 100-frame fine-tuning pool or the validation split. Excluding them changes the fine-tuned mAP50 by -0.001 to +0.015 (manuscript Supplementary Section S7, `evaluation/eval_bull_dedup.py`), so all reported numbers use the full 220-frame split.

## Changelog

- **v1.1.0** — Bull rows re-evaluated with `max_det=1000` and without NMS time-limit losses (all 92 bull rows changed; the 20 `source_val` rows are unchanged). Twelve fine-tuning runs (YOLO11m, YOLO26m and YOLO26n: fs-20 seeds 123 and 2024, fs-50 and fs-100 seed 2024) that in v1.0.0 had started from the seed-42 source checkpoint were re-trained from their own source checkpoint. Architecture comparisons now use Welch t-tests (paired tests reported alongside), and the source-versus-zero-shot variance comparison uses the Pitman--Morgan paired test (Levene alongside). Added the DeepSperm split lists (`data/bull_splits/`), F1, the class-agnostic COCO-only baseline, the tuned classical baseline, the evaluation scripts, and `paper_numbers.py`; the figure script now produces the manuscript figures. Table and section numbers follow the CMPB manuscript.
- **v1.0.0** — Initial release (bull rows evaluated at the Ultralytics default `max_det=300`; superseded).

## Citation

If you use this code, the released metric records, or build on the statistical analysis methodology, please cite both the manuscript and the archived release:

### Manuscript

```bibtex
@article{gungor2026crossspecies,
  title   = {Cross-Species Transfer Learning for Brightfield Spermatozoa Detection:
             Characterizing Zero-Shot Variance and Few-Shot Stabilization on Bull Microscopy},
  author  = {G{\"u}ng{\"o}r, Emre},
  journal = {Manuscript under review},
  year    = {2026}
}
```

### Software / data archive

```bibtex
@software{gungor2026repo,
  author  = {G{\"u}ng{\"o}r, Emre},
  title   = {cross-species-sperm-detection: Evaluation and statistical analysis code},
  year    = {2026},
  version = {v1.1.0},
  doi     = {10.5281/zenodo.20785515},  % concept DOI: always resolves to the latest version
  url     = {https://github.com/mregungor/cross-species-sperm-detection}
}
```

## Questions and feedback

For methodological questions about the paper, please contact the corresponding author (see paper). For issues with the released code or CSV (e.g., reproduction discrepancies, dependency conflicts), please open a GitHub issue.

## License

MIT License — see `LICENSE`.
