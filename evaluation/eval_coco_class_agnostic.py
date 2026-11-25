"""COCO-only baseline on the bull test split, evaluated class-agnostically (manuscript Table 3).

The 80-class COCO weights are evaluated on single-class data with single_cls=True, so any COCO class
counts as a spermatozoon; without it only class-0 ("person") predictions would match.

Usage: python evaluation/eval_coco_class_agnostic.py  (with SPERM_PROJECT_ROOT set)
Output: paper1_detection_segmentation/results/bull_maxdet_reeval/coco_class_agnostic.json
"""
from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("YOLO_OFFLINE", "1")
from bull_splits import project_root

ROOT = project_root()
OUT = ROOT / "paper1_detection_segmentation" / "results" / "bull_maxdet_reeval"
DATA = OUT / "test220.yaml"  # written by reeval_bull_maxdet.py


def main():
    from ultralytics import YOLO

    res = {}
    for name in ("yolov8m", "yolo11m", "yolo26n", "yolo26m"):
        m = YOLO(str(ROOT / f"{name}.pt")).val(data=str(DATA), split="val", imgsz=640, batch=8, max_det=1000,
                                                single_cls=True, workers=0, plots=False, verbose=False,
                                                project=str(OUT / "_val"), name="tmp", exist_ok=True)
        res[name] = dict(mAP50=m.box.map50, mAP50_95=m.box.map, precision=m.box.mp, recall=m.box.mr)
        print(name, round(m.box.map50, 4), flush=True)
    (OUT / "coco_class_agnostic.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
