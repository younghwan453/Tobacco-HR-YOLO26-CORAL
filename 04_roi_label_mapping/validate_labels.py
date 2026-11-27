"""Audit bounding-box overlap between one-class polygons and three-rater site scores."""

import argparse
from collections import Counter
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from generate_roi import load_overrides, make_rois
from load_rater_csv import SITE_ORDER, median_hr, read_ratings_table
from match_roi_polygon import assign_roi, load_yolo_polygons, polygon_union_to_contours
from normalize_image_ids import standard_id

SPLITS = ("train", "valid", "test")
EXTENSIONS = {".jpg", ".jpeg", ".png"}


def read_image(path):
    image = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Unreadable image: {path}")
    return image


def audit(dataset, ratings, overrides, output=None, previews=False):
    counters = Counter()
    sites, polygons = [], []
    seen = {}
    for split in SPLITS:
        for path in sorted((dataset / split / "images").iterdir()):
            if not path.is_file() or path.suffix.lower() not in EXTENSIONS:
                continue
            image_id = standard_id(path.stem)
            counters["images"] += 1
            if not image_id or image_id not in ratings:
                counters["images_without_rating_id"] += 1
                continue
            if image_id in seen:
                raise ValueError(f"Duplicate normalized image ID: {image_id} in {seen[image_id]} and {split}")
            seen[image_id] = split
            image = read_image(path)
            rois, _ = make_rois(image, image_id, overrides=overrides)
            label = dataset / split / "labels" / (path.stem + ".txt")
            if not label.exists():
                counters["missing_label_files"] += 1
            outlines = load_yolo_polygons(label, image.shape)
            grades = {site: median_hr(ratings[image_id][site]) for site in SITE_ORDER}
            proposals = Counter()
            assigned = {site: [] for site in SITE_ORDER}
            canvas = image.copy() if previews else None
            for index, outline in enumerate(outlines):
                site, overlap, ambiguous = assign_roi(outline, rois, image.shape)
                grade = None
                if site is None:
                    status = "outside_roi"
                else:
                    proposals[site] += 1
                    if ambiguous:
                        status = "ambiguous"
                    elif grades[site] is None:
                        status = "no_consensus"
                    else:
                        status = "linked"
                        grade = grades[site]
                        assigned[site].append(outline)
                counters[status] += 1
                polygons.append(dict(split=split, image_id=image_id, polygon_index=index,
                                     proposed_site=site, overlap=overlap, status=status, grade=grade))
                if canvas is not None:
                    color = (55, 155, 45) if status == "linked" else (40, 70, 220)
                    cv2.polylines(canvas, [outline], True, color, 3)
            for site, (x0, y0, x1, y1) in rois.items():
                grade = grades[site]
                contours, _ = polygon_union_to_contours(assigned[site], image.shape)
                sites.append(dict(split=split, image_id=image_id, site=site, rater_scores=ratings[image_id][site],
                                  median_grade=grade, x0=x0, y0=y0, x1=x1, y1=y1,
                                  proposed_polygons=proposals[site], linked_polygons=len(assigned[site]),
                                  linked_components=len(contours)))
                if canvas is not None:
                    cv2.rectangle(canvas, (x0, y0), (x1, y1), (0, 175, 255), 2)
                    cv2.putText(canvas, f"{site} HR{grade if grade is not None else 'NA'}",
                                (x0, max(18, y0)), cv2.FONT_HERSHEY_SIMPLEX, .6, (35, 35, 225), 2)
            if canvas is not None:
                folder = output / "previews" / split
                folder.mkdir(parents=True, exist_ok=True)
                ok, encoded = cv2.imencode(".jpg", canvas)
                if not ok:
                    raise OSError(f"Cannot encode overlay for {image_id}")
                encoded.tofile(str(folder / (image_id + ".jpg")))
    site_table = pd.DataFrame(sites, columns=["split", "image_id", "site", "rater_scores",
                             "median_grade", "x0", "y0", "x1", "y1", "proposed_polygons",
                             "linked_polygons", "linked_components"])
    polygon_table = pd.DataFrame(polygons, columns=["split", "image_id", "polygon_index",
                                "proposed_site", "overlap", "status", "grade"])
    summary = dict(rated_images=len(ratings), matched_images=len(seen),
                   rated_images_without_matches=len(ratings.keys() - seen.keys()), sites=len(sites),
                   consensus_sites=int(site_table.median_grade.notna().sum()),
                   sites_without_consensus=int(site_table.median_grade.isna().sum()),
                   sites_without_proposed_polygons=int((site_table.proposed_polygons == 0).sum()),
                   sites_with_multiple_proposals=int((site_table.proposed_polygons > 1).sum()),
                   total_polygons=len(polygons),
                   **{key: counters[key] for key in ("outside_roi", "ambiguous", "no_consensus",
                                                     "linked", "images_without_rating_id",
                                                     "missing_label_files")})
    train = polygon_table[(polygon_table.split == "train") & (polygon_table.status == "linked")]
    summary["linked_train_polygons_by_grade"] = {
        grade: int((train.grade == grade).sum()) for grade in range(5)
    }
    return site_table, polygon_table, summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", required=True, type=Path)
    p.add_argument("--ratings", required=True, type=Path)
    p.add_argument("--overrides", type=Path)
    p.add_argument("--output", type=Path)
    p.add_argument("--previews", action="store_true")
    a = p.parse_args()
    if a.previews and a.output is None:
        p.error("--previews needs --output")
    ratings = read_ratings_table(a.ratings)
    overrides = load_overrides(a.overrides)
    if a.output:
        if a.output.resolve().is_relative_to(a.dataset.resolve()):
            p.error("Output must be outside the input dataset")
        a.output.mkdir(parents=True)
    sites, polygons, summary = audit(a.dataset, ratings, overrides, a.output, a.previews)
    encoded = json.dumps(summary, indent=2)
    print(encoded)
    if a.output:
        sites.to_csv(a.output / "site_audit.csv", index=False)
        polygons.to_csv(a.output / "polygon_audit.csv", index=False)
        (a.output / "summary.json").write_text(encoded + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
