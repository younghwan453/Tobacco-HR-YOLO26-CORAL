"""Build disposable dataset views so Ultralytics writes only in the output tree."""

import os
from pathlib import Path
import shutil
import yaml


def stage_dataset(source, output, splits):
    source = source.resolve()
    output = output.resolve()
    if output.is_relative_to(source):
        raise ValueError("Output must be outside the source dataset")
    layout = {"train": "train/images", "val": "valid/images", "test": "test/images"}
    paths = {split: source / split for split in splits}
    for split_root in paths.values():
        for kind in ("images", "labels"):
            if not (split_root / kind).is_dir():
                raise FileNotFoundError(split_root / kind)
    output.mkdir(parents=True)
    work = output / "dataset"
    for split, split_root in paths.items():
        for kind in ("images", "labels"):
            shutil.copytree(split_root / kind, work / split / kind, ignore=shutil.ignore_patterns("*.cache"))
    # Validation requires train and val keys even when evaluating test only.
    if len(splits) == 1:
        chosen = f"{splits[0]}/images"
        layout.update(train=chosen, val=chosen)
    cfg = work / "data.yaml"
    cfg.write_text(yaml.safe_dump(dict(path=str(work), **layout, names={0: "HR_region"})), encoding="utf-8")
    os.environ.setdefault("YOLO_CONFIG_DIR", str(output / "ultralytics"))
    return cfg
