"""Robustness check: evaluate bull checkpoints on the test frames that have no pixel-identical copy in the
fine-tuning pool or validation split.

DeepSperm contains 39 groups of pixel-identical images; with the frame-level split, 22 of the 220 test frames
have an identical copy in the 100-frame pool or the 20-frame validation split. This script drops them
(results/bull_maxdet_reeval/test_dedup.txt, 198 frames) and re-evaluates every zero-shot and few-shot
checkpoint at max_det=1000, batch 8.

Usage: python evaluation/eval_bull_dedup.py  (with SPERM_PROJECT_ROOT set)
Output: paper1_detection_segmentation/results/bull_maxdet_reeval/dedup_results.csv
"""
from __future__ import annotations

import csv
import hashlib
import os
import re
from pathlib import Path

import cv2

from bull_splits import load_split, project_root

os.environ.setdefault("YOLO_OFFLINE", "1")
ROOT = project_root()
P1 = ROOT / "paper1_detection_segmentation"
RUNS = P1 / "results" / "runs"
OUT = P1 / "results" / "bull_maxdet_reeval"


def build_dedup() -> Path:
    read = lambda k: [str(p) for p in load_split(k, ROOT)]
    h = lambda p: hashlib.md5(cv2.imread(p).tobytes()).hexdigest()
    seen = {h(p) for k in ("fewshot_train_100", "fewshot_val") for p in read(k)}
    keep = [p for p in read("fewshot_test") if h(p) not in seen]
    (OUT / "test_dedup.txt").write_text("\n".join(keep), encoding="utf-8")
    y = OUT / "test_dedup.yaml"
    y.write_text(f"path: {OUT.as_posix()}\ntrain: test_dedup.txt\nval: test_dedup.txt\nnames:\n  0: sperm\n", encoding="utf-8")
    print("dedup test frames:", len(keep), flush=True)
    return y


def main():
    from ultralytics import YOLO

    data = build_dedup()
    pat_base = re.compile(r"^baseline_(\w+?)_bf_seed(\d+)$")
    pat_fs = re.compile(r"^fewshot_(\w+?)_bf_to_bull_(\d+)frame_seed(\d+)$")
    rows = []
    for d in sorted(RUNS.iterdir()):
        w = d / "weights" / "best.pt"
        if not w.exists():
            continue
        if m := pat_base.match(d.name):
            key = (m[1], "bull_zs", m[2])
        elif m := pat_fs.match(d.name):
            key = (m[1], f"bull_fs{m[2]}", m[3])
        else:
            continue
        r = YOLO(str(w)).val(data=str(data), split="val", imgsz=640, batch=8, max_det=1000, workers=0,
                             plots=False, verbose=False, project=str(OUT / "_val"), name="tmp", exist_ok=True)
        rows.append([*key, r.box.map50, r.box.map, r.box.mp, r.box.mr])
        print(*key, f"{r.box.map50:.4f}", flush=True)
    with (OUT / "dedup_results.csv").open("w", newline="", encoding="utf-8") as f:
        wr = csv.writer(f)
        wr.writerow(["model", "operating_point", "seed", "mAP50", "mAP50_95", "precision", "recall"])
        wr.writerows(rows)


if __name__ == "__main__":
    main()
