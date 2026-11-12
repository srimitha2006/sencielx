"""
SentinelX - collect labelled pose data for training the fall classifier.

Each saved row = pose features of the main (largest) person in one frame + its label.
Rows are appended to a CSV (default fall_data.csv), so you can run this many times.

Two ways to label:

1) Live / keyboard mode (default) - works with the webcam or any video file.
   Keys in the window:
     N      start recording as NORMAL (standing, walking, sitting, bending, ...)
     F      start recording as FALL   (lying / collapsed on the floor)
     SPACE  pause recording
     Q      quit
   Every time you press N or F a new "clip" starts. Clips are kept apart during
   evaluation so the accuracy you report is honest (frames from the same clip
   never appear in both training and testing).

2) Whole-video mode - give --label and every frame of the video gets that label.
   Good for clips that contain only one kind of posture, e.g.
     python collect_fall_data.py --source clips/lying1.mp4 --label fall
     python collect_fall_data.py --source clips/walk1.mp4  --label normal

Run:
    python collect_fall_data.py                  # webcam, keyboard labelling
    python collect_fall_data.py --source my.mp4  # label a video with the keyboard
"""

import argparse
import csv
import os
import time

import cv2
from ultralytics import YOLO

from fall_features import FEATURE_NAMES, extract_people, person_features


def parse_args():
    p = argparse.ArgumentParser(description="Collect pose data for fall detection")
    p.add_argument("--source", default="0", help="camera index or video file")
    p.add_argument("--out", default="fall_data.csv", help="CSV file to append rows to")
    p.add_argument("--model", default="yolo11n-pose.pt", help="YOLO pose weights")
    p.add_argument("--label", choices=["normal", "fall"], help="label every frame (whole-video mode)")
    p.add_argument("--every", type=int, default=2, help="save one frame out of every N (frames next to each other are near-identical)")
    p.add_argument("--conf", type=float, default=0.4, help="minimum person confidence")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--no-window", action="store_true", help="don't show video (whole-video mode only)")
    return p.parse_args()


def main():
    args = parse_args()
    if args.no_window and not args.label:
        raise SystemExit("--no-window needs --label (keyboard mode needs the window)")

    model = YOLO(args.model)
    is_cam = args.source.isdigit()
    cap = cv2.VideoCapture(int(args.source), cv2.CAP_DSHOW) if is_cam and os.name == "nt" else \
        cv2.VideoCapture(int(args.source) if is_cam else args.source)
    if not cap.isOpened():
        raise SystemExit(f"Could not open {args.source!r}")

    new_file = not os.path.exists(args.out) or os.path.getsize(args.out) == 0
    f = open(args.out, "a", newline="")
    writer = csv.writer(f)
    if new_file:
        writer.writerow(["label", "clip"] + FEATURE_NAMES)

    session = time.strftime("%Y%m%d-%H%M%S")
    clip_no = 0
    recording = args.label
    clip_id = f"{session}-{os.path.basename(args.source)}-0" if args.label else None
    counts = {"normal": 0, "fall": 0}
    frame_no = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_no += 1

            result = model.predict(frame, conf=args.conf, imgsz=args.imgsz, verbose=False)[0]
            people = extract_people(result)
            view = result.plot() if not args.no_window else None

            if people:
                # the subject = biggest person in the frame (ignores people walking past behind)
                box, kxy, kconf = max(people, key=lambda p: (p[0][2] - p[0][0]) * (p[0][3] - p[0][1]))
                if recording and frame_no % args.every == 0:
                    writer.writerow([recording, clip_id] + [f"{v:.5f}" for v in person_features(box, kxy, kconf)])
                    counts[recording] += 1
                if view is not None:
                    x1, y1, x2, y2 = map(int, box)
                    cv2.rectangle(view, (x1, y1), (x2, y2), (255, 255, 0), 3)

            if view is not None:
                state = f"RECORDING: {recording.upper()}" if recording else "PAUSED"
                color = (0, 0, 255) if recording == "fall" else (0, 200, 0) if recording else (200, 200, 200)
                cv2.putText(view, state, (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
                cv2.putText(view, f"saved  normal={counts['normal']}  fall={counts['fall']}", (15, 65),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                if not args.label:
                    cv2.putText(view, "N=normal  F=fall  SPACE=pause  Q=quit", (15, view.shape[0] - 15),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv2.imshow("SentinelX - data collection", view)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                if not args.label and key in (ord("n"), ord("f"), ord(" ")):
                    recording = {ord("n"): "normal", ord("f"): "fall", ord(" "): None}[key]
                    if recording:
                        clip_no += 1
                        clip_id = f"{session}-{clip_no}"
                    print(f"[collect] {'recording ' + recording if recording else 'paused'}")
    finally:
        f.close()
        cap.release()
        cv2.destroyAllWindows()

    print(f"[collect] Added normal={counts['normal']} fall={counts['fall']} rows to {args.out}")


if __name__ == "__main__":
    main()
