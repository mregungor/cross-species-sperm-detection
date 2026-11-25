"""Tuned classical baseline on the DeepSperm bull splits (threshold + connected components).

The earlier source-domain baseline (classical_baseline.py) used a single untuned configuration with
bright-object polarity, which suits phase contrast but not brightfield, where spermatozoa are dark.
Here both polarities, global Otsu and local adaptive thresholding, and the component-size filter are
grid-searched on the 20-frame bull validation split (maximum F1 at IoU 0.5), and the selected
configuration is applied once to the 220-frame held-out test split.

Usage: python evaluation/classical_baseline_bull.py  (with SPERM_PROJECT_ROOT set)
Output: paper1_detection_segmentation/results/classical_baseline_bull/result.json
"""
from __future__ import annotations

import itertools
import json
import os
from pathlib import Path

import cv2
import numpy as np

from bull_splits import load_split, project_root

ROOT = project_root()
RAW = ROOT / "data" / "raw" / "deepsperm_bull"
OUT = ROOT / "paper1_detection_segmentation" / "results" / "classical_baseline_bull"


def splits():
    return load_split("fewshot_val", ROOT), load_split("fewshot_test", ROOT)


def gt_boxes(img_path, w, h):
    out = []
    for line in img_path.with_suffix(".txt").read_text().splitlines():
        p = line.split()
        if len(p) == 5:
            x, y, bw, bh = map(float, p[1:])
            out.append([(x - bw / 2) * w, (y - bh / 2) * h, (x + bw / 2) * w, (y + bh / 2) * h])
    return np.array(out).reshape(-1, 4)


def detect(gray, method, dark, block, c, min_area, max_area):
    if method == "otsu":
        flag = cv2.THRESH_BINARY_INV if dark else cv2.THRESH_BINARY
        _, thr = cv2.threshold(gray, 0, 255, flag + cv2.THRESH_OTSU)
    else:
        flag = cv2.THRESH_BINARY_INV if dark else cv2.THRESH_BINARY
        thr = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, flag, block, c)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    thr = cv2.morphologyEx(cv2.morphologyEx(thr, cv2.MORPH_OPEN, k), cv2.MORPH_CLOSE, k)
    n, _, st, _ = cv2.connectedComponentsWithStats(thr, connectivity=8)
    boxes = [[x, y, x + w, y + h] for x, y, w, h, a in st[1:]
             if min_area <= a <= max_area and 0.3 <= w / max(h, 1) <= 3.0]
    return np.array(boxes, dtype=float).reshape(-1, 4)


def iou_matrix(a, b):
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    x1 = np.maximum(a[:, None, 0], b[None, :, 0]); y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2]); y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area = lambda z: (z[:, 2] - z[:, 0]) * (z[:, 3] - z[:, 1])
    return inter / (area(a)[:, None] + area(b)[None, :] - inter + 1e-9)


def score(images, cfg):
    tp = fp = fn = 0
    for gray, gt in images:
        pr = detect(gray, *cfg)
        m = iou_matrix(pr, gt)
        used = set()
        hit = 0
        # greedy one-to-one matching at IoU >= 0.5
        for i in np.argsort(-m.max(axis=1)) if len(pr) and len(gt) else []:
            j = int(np.argmax(np.where([k not in used for k in range(len(gt))], m[i], -1)))
            if m[i, j] >= 0.5 and j not in used:
                used.add(j); hit += 1
        tp += hit; fp += len(pr) - hit; fn += len(gt) - hit
    p = tp / max(tp + fp, 1); r = tp / max(tp + fn, 1)
    return dict(TP=tp, FP=fp, FN=fn, precision=p, recall=r, F1=2 * p * r / max(p + r, 1e-9))


def load(paths):
    out = []
    for p in paths:
        img = cv2.imread(str(p)); h, w = img.shape[:2]
        out.append((cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), gt_boxes(p, w, h)))
    return out


def main():
    val_paths, test_paths = splits()
    val, test = load(val_paths), load(test_paths)
    grid = []
    for dark in (True, False):
        for lo, hi in itertools.product((5, 10, 20, 40, 80), (300, 800, 2000)):
            grid.append(("otsu", dark, 0, 0, lo, hi))
            for block, c in itertools.product((15, 31, 51, 75, 101, 151), (2, 5, 10, 15, 20, 30)):
                grid.append(("adaptive", dark, block, c, lo, hi))
    results = [(score(val, cfg), cfg) for cfg in grid]
    best_val, best = max(results, key=lambda r: r[0]["F1"])
    test_res = score(test, best)
    keys = ("method", "dark_objects", "block", "C", "min_area", "max_area")
    out = dict(selected_config=dict(zip(keys, best)), val=best_val, test=test_res, n_configs=len(grid))
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "result.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
