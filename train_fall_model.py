"""
SentinelX - train the fall classifier on the data from collect_fall_data.py.

Model: Random Forest on 54 pose features (keypoint positions, visibility,
box aspect ratio, torso angle). Small, fast on CPU, and needs only a few
thousand rows.

Evaluation is done clip-wise (StratifiedGroupKFold): frames from one recording
clip are never split between train and test, so the scores are not inflated
by near-duplicate frames. The rule-based posture check is scored on the same
folds as a baseline, which is useful for your report.

Run:
    python train_fall_model.py                     # reads fall_data.csv, writes fall_model.joblib
    python train_fall_model.py --data a.csv --data b.csv
"""

import argparse
import csv
import time
from collections import Counter

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

from fall_features import FEATURE_NAMES, LABELS, rule_fall_score


def parse_args():
    p = argparse.ArgumentParser(description="Train the SentinelX fall classifier")
    p.add_argument("--data", action="append", help="CSV file(s) from collect_fall_data.py (default fall_data.csv)")
    p.add_argument("--out", default="fall_model.joblib")
    p.add_argument("--report", default="training_report.txt")
    p.add_argument("--trees", type=int, default=200)
    return p.parse_args()


def load(paths):
    X, y, groups = [], [], []
    for path in paths:
        with open(path, newline="") as f:
            reader = csv.reader(f)
            header = next(reader)
            if header[2:] != FEATURE_NAMES:
                raise SystemExit(f"{path}: columns don't match fall_features.py - recollect the data")
            for row in reader:
                if len(row) != len(header):
                    continue
                y.append(LABELS.index(row[0]))
                groups.append(f"{path}:{row[1]}")
                X.append([float(v) for v in row[2:]])
    return np.array(X, dtype=np.float32), np.array(y), np.array(groups)


def make_bundle(model, n_rows):
    import sklearn
    return {"model": model, "feature_names": FEATURE_NAMES, "labels": LABELS,
            "sklearn": sklearn.__version__, "trained": time.strftime("%Y-%m-%d %H:%M"), "rows": int(n_rows)}


def quick_fit(paths, out, trees=200):
    """Train on all data without evaluation (used by sentinelx.py to rebuild the model)."""
    X, y, _ = load(paths)
    clf = RandomForestClassifier(n_estimators=trees, class_weight="balanced", random_state=42, n_jobs=-1)
    clf.fit(X, y)
    bundle = make_bundle(clf, len(y))
    joblib.dump(bundle, out)
    return bundle


def main():
    args = parse_args()
    paths = args.data or ["fall_data.csv"]
    X, y, groups = load(paths)

    counts = Counter(y.tolist())
    if counts[0] == 0 or counts[1] == 0:
        raise SystemExit(f"Need both classes. Have normal={counts[0]}, fall={counts[1]}.")

    clips = {c: len(set(groups[y == c])) for c in (0, 1)}
    n_splits = min(5, clips[0], clips[1])
    lines = [
        f"SentinelX fall classifier - {time.strftime('%Y-%m-%d %H:%M')}",
        f"Data: {', '.join(paths)}",
        f"Rows: {len(y)}  (normal={counts[0]}, fall={counts[1]})",
        f"Clips: normal={clips[0]}, fall={clips[1]}",
    ]

    if n_splits >= 2:
        cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)
        splits = list(cv.split(X, y, groups))
        lines.append(f"Evaluation: {n_splits}-fold cross-validation, grouped by clip")
    else:
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        splits = list(cv.split(X, y))
        lines.append("Evaluation: 5-fold random split (WARNING: fewer than 2 clips per class, "
                     "scores will look better than real life - record more clips)")

    pred_model = np.zeros_like(y)
    pred_rule = np.zeros_like(y)
    for train_idx, test_idx in splits:
        clf = RandomForestClassifier(n_estimators=args.trees, class_weight="balanced", random_state=42, n_jobs=-1)
        clf.fit(X[train_idx], y[train_idx])
        pred_model[test_idx] = clf.predict(X[test_idx])
        pred_rule[test_idx] = [1 if rule_fall_score(f) >= 0.5 else 0 for f in X[test_idx]]

    for name, pred in (("Random Forest (trained)", pred_model), ("Rule-based baseline", pred_rule)):
        lines += [
            "",
            f"=== {name} ===",
            f"Accuracy: {accuracy_score(y, pred):.3f}   Fall F1: {f1_score(y, pred):.3f}",
            "Confusion matrix (rows = true, cols = predicted) [normal, fall]:",
            str(confusion_matrix(y, pred, labels=[0, 1])),
            classification_report(y, pred, target_names=LABELS, digits=3, zero_division=0),
        ]

    # final model on all data
    final = RandomForestClassifier(n_estimators=args.trees, class_weight="balanced", random_state=42, n_jobs=-1)
    final.fit(X, y)
    top = sorted(zip(final.feature_importances_, FEATURE_NAMES), reverse=True)[:10]
    lines += ["", "Top 10 features:"] + [f"  {n:<16} {v:.3f}" for v, n in top]

    joblib.dump(make_bundle(final, len(y)), args.out)
    lines += ["", f"Saved model to {args.out}"]

    report = "\n".join(lines)
    print(report)
    with open(args.report, "w") as f:
        f.write(report + "\n")
    print(f"Report written to {args.report}")


if __name__ == "__main__":
    main()
