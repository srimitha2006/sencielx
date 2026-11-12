"""
Shared pose-feature code for SentinelX fall detection.

YOLO11-pose gives, for every person, a bounding box and 17 COCO keypoints
(nose, eyes, ears, shoulders, elbows, wrists, hips, knees, ankles).
This module turns one person's box + keypoints into a fixed-length feature
vector. The same function is used when collecting data, when training,
and when running live, so the model always sees the same features.
"""

import math
import os

import numpy as np

KPT_CONF_MIN = 0.3  # keypoints below this confidence are treated as "not visible"

# COCO keypoint indices
L_SHOULDER, R_SHOULDER = 5, 6
L_HIP, R_HIP = 11, 12

KPT_NAMES = [
    "nose", "l_eye", "r_eye", "l_ear", "r_ear",
    "l_shoulder", "r_shoulder", "l_elbow", "r_elbow", "l_wrist", "r_wrist",
    "l_hip", "r_hip", "l_knee", "r_knee", "l_ankle", "r_ankle",
]

FEATURE_NAMES = (
    [f"{k}_{a}" for k in KPT_NAMES for a in ("x", "y")]  # 34: keypoints relative to box centre
    + [f"{k}_vis" for k in KPT_NAMES]                    # 17: 1 if keypoint visible
    + ["aspect_ratio", "torso_angle", "torso_known"]      # 3: body-shape features
)
N_FEATURES = len(FEATURE_NAMES)  # 54

LABELS = ["normal", "fall"]  # class 0 = normal (standing/walking/sitting), class 1 = fallen / lying on floor


def _mid(xy, vis, a, b):
    """Midpoint of two keypoints, or the single visible one, or None."""
    pts = [xy[i] for i in (a, b) if vis[i]]
    if not pts:
        return None
    return np.mean(pts, axis=0)


def torso_angle(kpts_xy, vis):
    """Angle of the shoulder-to-hip line from vertical, in degrees.
    0 = upright, 90 = horizontal (lying down). None if not enough keypoints."""
    sho = _mid(kpts_xy, vis, L_SHOULDER, R_SHOULDER)
    hip = _mid(kpts_xy, vis, L_HIP, R_HIP)
    if sho is None or hip is None:
        return None
    dx, dy = abs(hip[0] - sho[0]), abs(hip[1] - sho[1])
    if dx + dy < 1e-6:
        return None
    return math.degrees(math.atan2(dx, dy))


def person_features(box_xyxy, kpts_xy, kpts_conf):
    """Feature vector (length N_FEATURES) for one person.

    box_xyxy : (4,)    x1, y1, x2, y2 in pixels
    kpts_xy  : (17, 2) keypoint pixel coordinates
    kpts_conf: (17,)   keypoint confidences
    """
    x1, y1, x2, y2 = [float(v) for v in box_xyxy]
    w, h = max(x2 - x1, 1.0), max(y2 - y1, 1.0)
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    scale = max(w, h)

    kpts_xy = np.asarray(kpts_xy, dtype=np.float32)
    vis = np.asarray(kpts_conf, dtype=np.float32) >= KPT_CONF_MIN

    rel = np.zeros((17, 2), dtype=np.float32)
    rel[vis] = (kpts_xy[vis] - [cx, cy]) / scale  # scale-invariant, keeps orientation

    angle = torso_angle(kpts_xy, vis)
    return np.concatenate([
        rel.ravel(),
        vis.astype(np.float32),
        [w / h, (angle if angle is not None else 0.0) / 90.0, 1.0 if angle is not None else 0.0],
    ]).astype(np.float32)


def rule_fall_score(feat):
    """Fallback when no trained model exists: a simple posture rule.
    Returns a 0..1 'fall score' from torso angle and box shape."""
    aspect = float(feat[-3])
    angle = float(feat[-2]) * 90.0
    known = feat[-1] > 0.5
    if known:
        # torso tilted more than ~60 deg from vertical -> lying
        score = min(max((angle - 45.0) / 30.0, 0.0), 1.0)
        # very wide box + clearly tilted torso: lying even if keypoints are noisy.
        # (A wide box with an upright torso is just someone close to the camera.)
        if aspect > 1.6 and angle >= 35.0:
            score = max(score, 0.9)
        return score
    # no torso keypoints: fall back to box shape only
    return min(max((aspect - 1.0) / 0.6, 0.0), 1.0)


def extract_people(result):
    """List of (box_xyxy, kpts_xy, kpts_conf) for every person in a YOLO pose result."""
    people = []
    boxes = result.boxes
    kpts = result.keypoints
    if boxes is None or len(boxes) == 0 or kpts is None:
        return people
    xyxy = boxes.xyxy.cpu().numpy()
    kxy = kpts.xy.cpu().numpy()
    kconf = kpts.conf.cpu().numpy() if kpts.conf is not None else np.ones(kxy.shape[:2], dtype=np.float32)
    for i in range(len(xyxy)):
        people.append((xyxy[i], kxy[i], kconf[i]))
    return people


class FallClassifier:
    """Wraps the trained model (fall_model.joblib) or the rule-based fallback."""

    def __init__(self, path=None, data_path="fall_data.csv"):
        self.model = None
        self.source = "rule-based posture check (no trained model)"
        if not path:
            return
        problem = None
        try:
            import joblib
            import sklearn
            bundle = joblib.load(path)
            if bundle.get("feature_names") != FEATURE_NAMES:
                problem = "it was trained with different features"
            elif bundle.get("sklearn") != sklearn.__version__:
                problem = f"it was saved with scikit-learn {bundle.get('sklearn')}, you have {sklearn.__version__}"
            else:
                self.model = bundle["model"]
                self.source = f"trained model {path} ({bundle.get('rows', '?')} training samples)"
        except FileNotFoundError:
            problem = "file not found"
        except Exception as e:
            problem = str(e)

        # Model missing or saved by another library version: rebuild it from the training data (a few seconds)
        if self.model is None and os.path.exists(data_path):
            print(f"[SentinelX] Fall model needs rebuilding ({problem}). Training from {data_path} ...")
            try:
                from train_fall_model import quick_fit
                bundle = quick_fit([data_path], path)
                self.model = bundle["model"]
                self.source = f"trained model {path} (rebuilt from {data_path}, {bundle['rows']} samples)"
            except Exception as e:
                print(f"[SentinelX] Could not train the fall model: {e}")
        elif self.model is None and problem != "file not found":
            print(f"[SentinelX] Could not use fall model {path!r}: {problem}")

    def scores(self, feats):
        """Fall probability (0..1) for each feature row."""
        if len(feats) == 0:
            return np.zeros(0)
        feats = np.asarray(feats, dtype=np.float32)
        if self.model is None:
            return np.array([rule_fall_score(f) for f in feats])
        return self.model.predict_proba(feats)[:, 1]
