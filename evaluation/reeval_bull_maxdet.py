"""Re-evaluate every bull checkpoint on the 220-frame held-out test split at max_det=300 and 1000.

Why: DeepSperm frames carry up to 568 annotated cells (median ~414 in two of three subsets), but
Ultralytics returns at most max_det=300 boxes per image by default. The v1.0.0 bull metrics
were computed at the default, which caps recall at 0.737 on this split; all manuscript bull metrics use
max_det=1000.

The test split is read from data/bull_splits/fewshot_test.txt and verified against the SHA-256
recorded in provenance_example/split_manifest_bull.json.

Usage:
    python evaluation/reeval_bull_maxdet.py  (with SPERM_PROJECT_ROOT set)
Output:
    paper1_detection_segmentation/results/bull_maxdet_reeval/reeval_results.csv
"""
from __future__ import annotations

import csv
import os
import re
from pathlib import Path

from bull_splits import load_split, project_root

os.environ.setdefault("YOLO_OFFLINE", "1")

ROOT = project_root()
P1 = ROOT / "paper1_detection_segmentation"
RUNS = P1 / "results" / "runs"
OUT = P1 / "results" / "bull_maxdet_reeval"
MAX_DETS = (300, 1000)


def build_test_split() -> Path:
    test = load_split("fewshot_test", ROOT)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "test220.txt").write_text("\n".join(str(p) for p in test), encoding="utf-8")
    yaml = OUT / "test220.yaml"
    yaml.write_text(f"path: {OUT.as_posix()}\ntrain: test220.txt\nval: test220.txt\nnames:\n  0: sperm\n",
                    encoding="utf-8")
    return yaml


def checkpoints():
    pat_base = re.compile(r"^baseline_(\w+?)_bf_seed(\d+)$")
    pat_fs = re.compile(r"^fewshot_(\w+?)_bf_to_bull_(\d+)frame_seed(\d+)$")
    pat_sub = re.compile(r"^fewshot_(\w+?)_bf_to_bull_20frame_sub(\d+)_srcseed(\d+)$")
    pat_hr = re.compile(r"^ablation_(\w+?)_bf_img1280_batch8_seed(\d+)$")
    for d in sorted(RUNS.iterdir()):
        w = d / "weights" / "best.pt"
        if not w.exists():
            continue
        if m := pat_base.match(d.name):
            yield m[1], "bull_zs", m[2], w, 640
        elif m := pat_fs.match(d.name):
            yield m[1], f"bull_fs{m[2]}", m[3], w, 640
        elif m := pat_sub.match(d.name):
            yield m[1], "bull_fs20_subset", m[2], w, 640
        elif m := pat_hr.match(d.name):
            yield m[1], "bull_zs_img1280", m[2], w, 1280
    for name in ("yolov8m", "yolo11m", "yolo26n", "yolo26m"):
        yield name, "bull_coco_only", "-", ROOT / f"{name}.pt", 640


def main():
    from ultralytics import YOLO

    data = build_test_split()
    out_csv = OUT / "reeval_results.csv"
    done = set()
    if out_csv.exists():
        with out_csv.open(encoding="utf-8") as f:
            done = {(r["model"], r["operating_point"], r["seed"], r["max_det"]) for r in csv.DictReader(f)}
    new = not out_csv.exists()
    with out_csv.open("a", newline="", encoding="utf-8") as f:
        wr = csv.writer(f)
        if new:
            wr.writerow(["model", "operating_point", "seed", "max_det", "mAP50", "mAP50_95", "precision", "recall"])
        for model, op, seed, w, imgsz in checkpoints():
            for md in MAX_DETS:
                if (model, op, seed, str(md)) in done:
                    continue
                # A fresh YOLO object per setting, so no head attribute carries over between settings (see README).
                m = YOLO(str(w)).val(data=str(data), split="val", imgsz=imgsz, batch=8 if imgsz == 640 else 2, max_det=md,
                                     workers=0, plots=False, verbose=False, save_json=False,
                                     project=str(OUT / "_val"), name="tmp", exist_ok=True)
                wr.writerow([model, op, seed, md, m.box.map50, m.box.map, m.box.mp, m.box.mr])
                f.flush()
                print(f"{model:8s} {op:18s} {seed:>5s} md={md:<5d} mAP50={m.box.map50:.4f} R={m.box.mr:.4f}",
                      flush=True)


if __name__ == "__main__":
    main()
