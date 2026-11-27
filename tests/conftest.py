"""Expose standalone stage scripts to their local tests."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
for stage in ("02_annotation", "03_yolo_segmentation", "04_roi_label_mapping", "05_coral_ordinal"):
    sys.path.insert(0, str(ROOT / stage))
