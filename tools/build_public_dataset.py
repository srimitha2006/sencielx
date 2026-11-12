"""
Builds fall_data.csv from two public image datasets, so the fall model can be
trained without recording your own data.

Datasets (clone them first):
  1) Human-Fall-Detection by bakshtb / Y-B-Class-Projects
       git clone --depth 1 https://github.com/Y-B-Class-Projects/Human-Fall-Detection
     fall_dataset/images/{fall,not-fall}  and  fall_dataset/old/{fall,not-fall}
  2) Simuletic CCTV fall detection sample (CC BY-NC 4.0, credit: Simuletic, simuletic.com)
       git clone --depth 1 https://github.com/Simuletic/cctv-fall-detection-dataset
     YOLO boxes with classes person / fallen_person

Every image is run through YOLO11n-pose, the person is turned into the same
54 features that sentinelx.py uses, and a mirrored (left-right flipped) copy
is added as augmentation. Images that come from the same original photo
(the crop in images/ and the full photo in old/, and the flipped copy) share
one group id, so train_fall_model.py never tests on a near-copy of a
training image.

Run (from the project folder):
    python tools/build_public_dataset.py --hfd path/to/Human-Fall-Detection --cctv path/to/cctv-fall-detection-dataset
    python train_fall_model.py
"""

import argparse
import csv
import glob
import os
import re
import sys

import cv2
import numpy as np
from ultralytics import YOLO

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from fall_features import FEATURE_NAMES, extract_people, person_features  # noqa: E402

IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--hfd", help="path to the Human-Fall-Detection repo")
    p.add_argument("--cctv", help="path to the cctv-fall-detection-dataset repo")
    p.add_argument("--out", default="fall_data.csv")
    p.add_argument("--model", default="yolo11n-pose.pt")
    p.add_argument("--conf", type=float, default=0.25)
    return p.parse_args()


def iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def poses(model, img, conf):
    return extract_people(model.predict(img, conf=conf, imgsz=640, verbose=False)[0])


def largest(people):
    return max(people, key=lambda p: (p[0][2] - p[0][0]) * (p[0][3] - p[0][1])) if people else None


def hfd_rows(model, root, conf):
    """Whole-image labels: the main (largest) person gets the folder's label."""
    rows, skipped = [], 0
    for sub in ("images", "old"):
        for folder, label in (("fall", "fall"), ("not-fall", "normal")):
            for path in sorted(glob.glob(os.path.join(root, "fall_dataset", sub, folder, "*"))):
                if not path.lower().endswith(IMG_EXT):
                    continue
                img = cv2.imread(path)
                if img is None:
                    skipped += 1
                    continue
                num = re.match(r"(\d+)", os.path.basename(path))
                group = f"hfd-{folder}-{num.group(1) if num else os.path.basename(path)}"
                for flip in (False, True):
                    im = cv2.flip(img, 1) if flip else img
                    person = largest(poses(model, im, conf))
                    if person is None:
                        skipped += not flip
                        break
                    rows.append((label, group, person_features(*person)))
    print(f"[hfd]  {len(rows)} rows, {skipped} images without a detected person")
    return rows


def cctv_rows(model, root, conf):
    """Box labels: match each labelled box to a detected person (IoU >= 0.4)."""
    rows, unmatched = [], 0
    for path in sorted(glob.glob(os.path.join(root, "images", "*", "*"))):
        if not path.lower().endswith(IMG_EXT):
            continue
        split = os.path.basename(os.path.dirname(path))
        lbl = os.path.join(root, "labels", split, os.path.splitext(os.path.basename(path))[0] + ".txt")
        if not os.path.exists(lbl):
            continue
        img = cv2.imread(path)
        h, w = img.shape[:2]
        boxes = []
        for line in open(lbl):
            parts = line.split()
            if len(parts) != 5:
                continue
            c, cx, cy, bw, bh = int(parts[0]), *map(float, parts[1:])
            boxes.append(("fall" if c == 1 else "normal",
                          [(cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h]))
        for flip in (False, True):
            im = cv2.flip(img, 1) if flip else img
            people = poses(model, im, conf)
            for i, (label, b) in enumerate(boxes):
                bb = [w - b[2], b[1], w - b[0], b[3]] if flip else b
                best = max(people, key=lambda p: iou(p[0], bb), default=None)
                if best is None or iou(best[0], bb) < 0.4:
                    unmatched += not flip
                    continue
                rows.append((label, f"cctv-{os.path.basename(path)}-{i}", person_features(*best)))
    print(f"[cctv] {len(rows)} rows, {unmatched} labelled boxes without a matching pose")
    return rows


def main():
    args = parse_args()
    if not args.hfd and not args.cctv:
        raise SystemExit("give --hfd and/or --cctv")
    model = YOLO(args.model)
    rows = []
    if args.hfd:
        rows += hfd_rows(model, args.hfd, args.conf)
    if args.cctv:
        rows += cctv_rows(model, args.cctv, args.conf)

    with open(args.out, "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["label", "clip"] + FEATURE_NAMES)
        for label, group, feat in rows:
            wr.writerow([label, group] + [f"{v:.5f}" for v in feat])
    n_fall = sum(r[0] == "fall" for r in rows)
    print(f"Wrote {len(rows)} rows to {args.out}  (fall={n_fall}, normal={len(rows) - n_fall})")


if __name__ == "__main__":
    main()
