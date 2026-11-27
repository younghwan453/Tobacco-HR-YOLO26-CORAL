"""Report ordinal metrics from stored GT predictions or an existing checkpoint."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, cohen_kappa_score, confusion_matrix, mean_absolute_error,
    precision_recall_fscore_support,
)

GRADES = list(range(5))


def evaluate(frame, schema="gt"):
    if schema != "gt":
        raise ValueError("Expected Roboflow GT polygon predictions")
    if frame.empty:
        raise ValueError("No evaluable predictions")
    y, p, expected = (pd.to_numeric(frame[c]).to_numpy(float)
                      for c in ("hr_grade", "hr_pred", "hr_expected"))
    if not np.isin(y, GRADES).all() or not np.isin(p, GRADES).all():
        raise ValueError("True and predicted labels must be integer grades 0–4")
    if not (np.isfinite(expected) & (expected >= 0) & (expected <= 4)).all():
        raise ValueError("Expected HR must be finite and within 0–4")
    prob = frame[[f"prob_HR{k}" for k in GRADES]].to_numpy(float)
    if not (np.isfinite(prob) & (prob >= -1e-6) & (prob <= 1 + 1e-6)).all():
        raise ValueError("Invalid class probabilities")
    if not np.allclose(prob.sum(axis=1), 1, atol=2e-4, rtol=0):
        raise ValueError("Class probabilities do not sum to one")
    if not np.allclose(prob @ GRADES, expected, atol=5e-4, rtol=0):
        raise ValueError("Expected HR differs from probability-weighted grade")
    precision, recall, f1, support = precision_recall_fscore_support(
        y, p, labels=GRADES, zero_division=0,
    )
    qwk = cohen_kappa_score(y, p, labels=GRADES, weights="quadratic")
    return dict(n=len(frame), accuracy=accuracy_score(y, p), mae_grade=mean_absolute_error(y, p),
                mae_expected=mean_absolute_error(y, expected),
                mean_expected_hr=float(expected.mean()),
                qwk=float(qwk) if np.isfinite(qwk) else None,
                confusion_matrix=confusion_matrix(y, p, labels=GRADES).tolist(),
                per_class=[dict(grade=k, precision=precision[k], recall=recall[k],
                                f1=f1[k], support=int(support[k])) for k in GRADES])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Prediction CSV, or feature CSV with --checkpoint")
    parser.add_argument("--schema", default="gt", choices=("gt",))
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    frame = pd.read_csv(args.input)
    if args.checkpoint:
        import torch
        from coral_model import load_checkpoint, predict_coral_from_features
        device = torch.device(args.device)
        model, scaler = load_checkpoint(args.checkpoint, device)
        frame = predict_coral_from_features(frame, model, scaler, device)
    encoded = json.dumps(evaluate(frame, args.schema), indent=2, allow_nan=False)
    if args.output:
        with args.output.open("x", encoding="utf-8") as handle:
            handle.write(encoded + "\n")
    print(encoded)


if __name__ == "__main__":
    main()
