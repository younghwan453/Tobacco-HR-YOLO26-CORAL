"""Audit a YOLO dataset or convert its five HR polygon IDs to HR_region."""

import argparse
from collections import Counter, defaultdict
from hashlib import sha256
import json
from pathlib import Path
import re
import shutil

SPLITS = ("train", "valid", "test")
IMAGE_ENDINGS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp", ".bmp"}


def convert_text(text, allowed=range(5)):
    """Preserve all characters other than the class token on each line."""
    result = []
    for index, line in enumerate(text.splitlines(keepends=True), 1):
        parts = line.split()
        if not parts:
            result.append(line)
            continue
        cls, *coords = parts
        if not cls.isascii() or not cls.isdecimal() or int(cls) not in allowed:
            raise ValueError(f"Line {index}: invalid class")
        if len(coords) < 6 or len(coords) % 2 or not all(0 <= float(c) <= 1 for c in coords):
            raise ValueError(f"Line {index}: invalid normalized polygon")
        result.append(re.sub(r"^([^\S\r\n]*)[0-9]+", r"\g<1>0", line, count=1))
    return "".join(result)


def inspect_dataset(root, allowed=range(5)):
    pairs = []
    names, hashes = defaultdict(set), defaultdict(set)
    counts = {}
    for split in SPLITS:
        images = defaultdict(list)
        labels = defaultdict(list)
        for path in (root / split / "images").iterdir():
            if path.is_file() and path.suffix.lower() in IMAGE_ENDINGS:
                images[path.stem.casefold()].append(path)
        for path in (root / split / "labels").glob("*.txt"):
            if path.is_file():
                labels[path.stem.casefold()].append(path)
        classes = Counter()
        missing_labels = 0
        orphan_labels = len(labels.keys() - images.keys())
        duplicate_stems = 0
        empty_labels = 0
        for stem, paths in sorted(images.items()):
            if len(paths) != 1 or len(labels[stem]) > 1:
                duplicate_stems += 1
                continue
            image = paths[0]
            names[stem].add(split)
            with image.open("rb") as handle:
                digest = sha256()
                for block in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(block)
                hashes[digest.hexdigest()].add(split)
            label = next(iter(labels[stem]), None)
            converted = None
            if label:
                raw = label.read_bytes().decode("utf-8-sig")
                try:
                    count = Counter(int(line.split()[0]) for line in raw.splitlines() if line.strip())
                    converted = convert_text(raw, allowed)
                except ValueError as exc:
                    raise ValueError(f"{label}: {exc}") from exc
                classes.update(count)
                empty_labels += not count
            else:
                missing_labels += 1
            pairs.append((split, image, label, converted))
        counts[split] = dict(images=sum(map(len, images.values())), polygons=sum(classes.values()),
                             classes=dict(sorted(classes.items())), missing_labels=missing_labels,
                             orphan_labels=orphan_labels, duplicate_stems=duplicate_stems,
                             empty_labels=empty_labels)
    audit = dict(splits=counts, repeated_stems=sum(len(v) > 1 for v in names.values()),
                 repeated_image_hashes=sum(len(v) > 1 for v in hashes.values()))
    return pairs, audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--output", type=Path, help="New directory. Omit for a read-only audit.")
    parser.add_argument("--single-class", action="store_true", help="Audit only class ID 0.")
    args = parser.parse_args()
    source = args.source.resolve()
    pairs, audit = inspect_dataset(source, range(1) if args.single_class else range(5))
    print(json.dumps(audit, indent=2))
    if not args.output:
        return
    out = args.output.resolve()
    if out.is_relative_to(source):
        parser.error("Output must be outside the source tree")
    if any(row["orphan_labels"] or row["duplicate_stems"] for row in audit["splits"].values()):
        parser.error("Resolve orphan labels or duplicate stems before conversion")
    if audit["repeated_stems"] or audit["repeated_image_hashes"]:
        parser.error("Resolve cross-split duplicates before conversion")
    out.mkdir(parents=True)
    for split in SPLITS:
        (out / split / "images").mkdir(parents=True)
        (out / split / "labels").mkdir()
    for split, image, label, converted in pairs:
        shutil.copy2(image, out / split / "images" / image.name)
        if label:
            (out / split / "labels" / label.name).write_bytes(converted.encode("utf-8"))
    (out / "data.yaml").write_text(
        "train: train/images\nval: valid/images\ntest: test/images\nnames:\n  0: HR_region\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
