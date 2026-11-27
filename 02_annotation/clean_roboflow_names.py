"""Preview or copy a Roboflow export with its historical filename cleanup."""

import argparse
from pathlib import Path
import re
import shutil

SPLITS = ("train", "valid", "test")
FOLDERS = ("images", "labels")


def make_new_name(filename):
    # 예: SHA_2_1_3_jpg.rf.abc123.jpg
    pattern = (
        r"^(?:corrected_)?(.+?)_jpg\.rf\.[^.]+"
        r"\.(jpg|jpeg|png|txt)$"
    )

    match = re.fullmatch(
        pattern, filename, flags=re.IGNORECASE
    )

    if not match:
        return None

    name, ext = match.groups()

    # 하이픈을 언더스코어로 변경
    name = name.replace("-", "_")

    return f"{name}.{ext.lower()}"


def plan(source):
    for split in SPLITS:
        entries = {}
        for folder in FOLDERS:
            destinations = {}
            files = (source / split / folder).iterdir()
            for path in sorted(files):
                if not path.is_file():
                    continue
                name = make_new_name(path.name) or path.name
                target = name.casefold()
                if target in destinations:
                    raise ValueError(f"Filename collision in {split}/{folder}: {name}")
                destinations[target] = (path, name)
            entries[folder] = list(destinations.values())
        images = {Path(name).stem.casefold() for _, name in entries["images"]
                  if Path(name).suffix.lower() in (".jpg", ".jpeg", ".png", ".webp", ".bmp")}
        labels = {Path(name).stem.casefold() for _, name in entries["labels"]
                  if Path(name).suffix.lower() == ".txt"}
        if labels - images:
            raise ValueError(f"{split}: {len(labels - images)} labels do not match any image after cleanup")
        for folder, items in entries.items():
            for path, name in items:
                yield path, Path(split) / folder / name


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("source", type=Path)
    p.add_argument("--output", type=Path, help="Copy to a new dataset directory. Omit to preview changes.")
    a = p.parse_args()
    files = list(plan(a.source))
    for src, dest in files:
        if src.name != dest.name:
            print(f"{src.name} -> {dest}")
    if a.output is None:
        return
    if a.output.resolve().is_relative_to(a.source.resolve()):
        p.error("Output must be outside the source dataset")
    a.output.mkdir(parents=True)
    for split in SPLITS:
        for folder in FOLDERS:
            (a.output / split / folder).mkdir(parents=True)
    for src, dest in files:
        shutil.copy2(src, a.output / dest)
    if (a.source / "data.yaml").is_file():
        import yaml
        cfg = yaml.safe_load((a.source / "data.yaml").read_text(encoding="utf-8-sig"))
        cfg.update(path=".", train="train/images", val="valid/images", test="test/images")
        # Retain class metadata without the exporter workspace identifiers.
        cfg.pop("roboflow", None)
        (a.output / "data.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")


if __name__ == "__main__":
    main()
