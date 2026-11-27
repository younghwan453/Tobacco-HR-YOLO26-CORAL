"""Evaluate a one-class segmentation checkpoint on a private split copy."""

import argparse
import json
from pathlib import Path
from yolo_runtime import stage_dataset


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--weights", type=Path, required=True)
    p.add_argument("--split", choices=("valid", "test"), default="test")
    p.add_argument("--imgsz", type=int)
    p.add_argument("--batch", type=int)
    p.add_argument("--device", default="0")
    a = p.parse_args()
    if not a.weights.is_file():
        p.error("Supply a trusted local checkpoint")
    cfg = stage_dataset(a.dataset, a.output, (a.split,))
    from ultralytics import YOLO
    overrides = {k: v for k, v in {"imgsz": a.imgsz, "batch": a.batch}.items() if v is not None}
    result = YOLO(str(a.weights.resolve())).val(
        data=str(cfg), split="val" if a.split == "valid" else "test", **overrides,
        device=a.device, workers=0, plots=False,
        project=str(a.output.resolve()), name="evaluation",
    )
    print(json.dumps(result.results_dict, indent=2))
    with (a.output / "metrics.json").open("x", encoding="utf-8") as handle:
        json.dump(result.results_dict, handle, indent=2)


if __name__ == "__main__":
    main()
