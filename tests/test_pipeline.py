import csv
from pathlib import Path

import nbformat
import numpy as np
import pandas as pd
import pytest
import yaml

from evaluate_coral import evaluate
from load_rater_csv import median_hr, read_ratings_table
from match_roi_polygon import assign_roi
from clean_roboflow_names import make_new_name, plan
from normalize_image_ids import canonical_id
from prepare_single_class import convert_text, inspect_dataset
from yolo_runtime import stage_dataset


def test_names_and_rater_consensus(tmp_path):
    assert canonical_id("crop_SHA_1-1-1.jpg") == "SHA_1_1_1"
    assert canonical_id("SHA_1_1_1") == "SHA_1_1_1"
    assert canonical_id("no_valid_id") is None
    path = tmp_path / "ratings.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        csv.writer(handle).writerow(["SHA_1-1-1"] + ["0,0,1", "2,2,3", "0,2,2", "4,4,4", "0,0,0", "3,4,4"])
    scores = read_ratings_table(path)["SHA_1_1_1"]
    assert [median_hr(scores[site]) for site in ("EV", "1", "2", "3", "4", "SARM1")] == [0, 2, None, 4, 0, 4]


def test_class_remapping_preserves_coordinate_tokens():
    value = "  4\t0 0   1 0 1 1\r\n\r\n2 0.3 0 0 1 0 0\n"
    assert convert_text(value) == value.replace("4\t", "0\t").replace("\n2 ", "\n0 ")
    with pytest.raises(ValueError):
        convert_text("2 0 0 nan 0 1 1")
    with pytest.raises(ValueError):
        convert_text("5 0 0 1 0 1 1")


def test_split_audit_and_output_write_isolation(tmp_path):
    root = tmp_path / "source"
    for split in ("train", "valid", "test"):
        (root / split / "images").mkdir(parents=True)
        (root / split / "labels").mkdir()
        (root / split / "images" / "a.png").write_bytes(b"same")
        (root / split / "labels" / "a.txt").write_text("0 0 0 1 0 1 1\n")
    _, audit = inspect_dataset(root, range(1))
    assert audit["repeated_stems"] == audit["repeated_image_hashes"] == 1
    assert audit["splits"]["test"]["missing_labels"] == 0
    with pytest.raises(ValueError):
        stage_dataset(root, root / "bad", ("test",))
    target = tmp_path / "working"
    cfg = stage_dataset(root, target, ("test",))
    assert cfg.parent == target / "dataset"
    assert yaml.safe_load(cfg.read_text())["test"] == "test/images"
    assert not (target / "dataset" / "train").exists()
    with pytest.raises(FileExistsError):
        stage_dataset(root, target, ("test",))


def test_overlap_flags_ambiguous_and_outside_masks():
    poly = np.array([[1, 1], [3, 1], [3, 3], [1, 3]], dtype=np.int32)
    assert assign_roi(poly, {"1": (0, 0, 4, 4), "2": (4, 4, 6, 6)}, (6, 6)) == ("1", 1.0, False)
    assert assign_roi(poly, {"1": (4, 4, 6, 6)}, (6, 6)) == (None, 0.0, False)
    assert assign_roi(poly, {"1": (0, 0, 4, 4), "2": (0, 0, 4, 4)}, (6, 6))[2]


def test_roboflow_suffix_cleanup(tmp_path):
    assert make_new_name("corrected_SHA_1-2-3_jpg.rf.abc123.JPG") == "SHA_1_2_3.jpg"
    assert make_new_name("SHA_1-2-3_jpg.rf.abc123.txt") == "SHA_1_2_3.txt"
    assert make_new_name("SHA_1_2_3.jpg") is None
    for split in ("train", "valid", "test"):
        for folder in ("images", "labels"):
            (tmp_path / split / folder).mkdir(parents=True)
        (tmp_path / split / "images" / "SHA_1-2-3_jpg.rf.abc.jpg").touch()
        (tmp_path / split / "labels" / "SHA_1-2-3_jpg.rf.abc.txt").touch()
    assert len(list(plan(tmp_path))) == 6
    (tmp_path / "train" / "images" / "SHA_1_2_3.jpg").touch()
    with pytest.raises(ValueError, match="collision"):
        list(plan(tmp_path))


def test_coral_expected_score_and_label_validation():
    frame = pd.DataFrame({
        "hr_grade": [1, 4], "hr_pred": [1, 4], "hr_expected": [1.5, 4.0],
        "prob_HR0": [0.0, 0.0], "prob_HR1": [0.5, 0.0],
        "prob_HR2": [0.5, 0.0], "prob_HR3": [0.0, 0.0], "prob_HR4": [0.0, 1.0],
    })
    metrics = evaluate(frame, "gt")
    assert metrics["accuracy"] == 1 and metrics["mae_expected"] == 0.25
    assert len(metrics["confusion_matrix"]) == 5
    with pytest.raises(ValueError):
        evaluate(frame.assign(hr_expected=[0, 4]), "gt")
    with pytest.raises(ValueError):
        evaluate(frame.assign(hr_pred=[5, 4]), "gt")


def test_public_notebooks_are_valid_and_unexecuted():
    root = Path(__file__).resolve().parents[1]
    for folder in ("01_preprocessing", "04_roi_label_mapping", "05_coral_ordinal"):
        for path in (root / folder).glob("*.ipynb"):
            nb = nbformat.read(path, as_version=4)
            nbformat.validate(nb)
            for cell in nb.cells:
                if cell.cell_type == "code":
                    assert not cell.outputs and cell.execution_count is None
                    compile(cell.source, str(path), "exec")
