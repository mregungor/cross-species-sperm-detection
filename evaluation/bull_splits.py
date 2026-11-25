"""Released DeepSperm bull split lists (data/bull_splits/), checked against the manifest SHA-256.

Each list holds project-relative image paths. The manifest hashes were computed on the original list
files, written on Windows with CRLF line endings and absolute paths under F:\\SpermMotion_train, so the
check rebuilds that form; it does not depend on the operating system or on directory-listing order.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SPLITS = REPO / "data" / "bull_splits"
MANIFEST = REPO / "provenance_example" / "split_manifest_bull.json"
ORIG_ROOT = "F:\\SpermMotion_train"


def project_root() -> Path:
    """Root holding data/raw/deepsperm_bull/, the COCO weights and paper1_detection_segmentation/."""
    if "SPERM_PROJECT_ROOT" not in os.environ:
        raise SystemExit("Set SPERM_PROJECT_ROOT to the project root (see README, Evaluation scripts).")
    return Path(os.environ["SPERM_PROJECT_ROOT"])


def load_split(name: str, root: Path) -> list[Path]:
    rel = [ln for ln in (SPLITS / f"{name}.txt").read_text(encoding="utf-8").splitlines() if ln.strip()]
    orig = "\r\n".join(ORIG_ROOT + "\\" + r.replace("/", "\\") for r in rel)
    digests = {hashlib.sha256(s.encode()).hexdigest() for s in (orig, orig + "\r\n")}
    expected = json.loads(MANIFEST.read_text(encoding="utf-8"))["splits"][name]["sha256"]
    assert expected in digests, f"{name}: list does not match the manifest hash"
    return [root / r for r in rel]
