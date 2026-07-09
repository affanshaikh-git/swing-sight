"""
stroke_analyzer.py — Step 2 of the tennis stroke trainer pipeline.

Reads the pose CSV from pose_extract.py, then:
  1. Segments individual strokes (peaks in dominant-wrist speed)
  2. Classifies each stroke: forehand / backhand / serve (rule-based v1)
  3. Computes per-stroke technique metrics
  4. Writes a session report (markdown) + per-stroke CSV

Usage:
    python stroke_analyzer.py session_pose.csv --hand right
    python stroke_analyzer.py session_pose.csv --hand right --label   # interactive labeling for future ML training

Assumes side-on camera. All positions are MediaPipe-normalized coordinates;
speeds are in normalized-units/second — consistent within and across sessions
if camera framing is consistent.
"""

import argparse
import csv
import json
from dataclasses import dataclass, asdict

import numpy as np
from scipy.signal import savgol_filter, find_peaks


# ---------------------------------------------------------------- data loading

def load_pose_csv(path):
    with open(path) as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = [list(map(float, r)) for r in reader]
    data = np.array(rows)
    cols = {name: i for i, name in enumerate(header)}
    return data, cols


def series(data, cols, landmark, axis):
    return data[:, cols[f"{landmark}_{axis}"]]


def smooth(x, fps):
    win = max(5, int(fps * 0.15) | 1)  # ~150ms window, odd
    if len(x) <= win:
        return x
    return savgol_filter(x, win, 3)


# ---------------------------------------------------------------- segmentation

def wrist_speed(data, cols, hand, fps):
    x = smooth(series(data, cols, f"{hand}_wrist", "x"), fps)
    y = smooth(series(data, cols, f"{hand}_wrist", "y"), fps)
    vx, vy = np.gradient(x) * fps, np.gradient(y) * fps
    return np.hypot(vx, vy)


def segment_strokes(speed, fps, min_gap_s=1.0, prominence_frac=0.35):
    """Find stroke events as prominent peaks in wrist speed."""
    if speed.max() <= 0:
        return []
    prom = speed.max() * prominence_frac
    peaks, _ = find_peaks(speed, prominence=prom,
                          distance=int(min_gap_s * fps))
    strokes = []
    for p in peaks:
        # stroke window: back to where speed drops under 15% of the peak
        thresh = speed[p] * 0.15
        s = p
        while s > 0 and speed[s] > thresh and p - s < int(1.5 * fps):
            s -= 1
        e = p
        while e < len(speed) - 1 and speed[e] > thresh and e - p < int(1.5 * fps):
            e += 1
        strokes.append((s, p, e))  # start, impact(≈peak speed), end
    return strokes


# ------------------------------------------------------------- classification

def classify(data, cols, hand, s, p, e):
    """
    Rule-based v1 classifier (side-on camera, y grows downward):
      SERVE   — wrist at impact well above the head
      FOREHAND— dominant wrist starts on dominant side of body midline
      BACKHAND— dominant wrist starts across the body midline
    """
    wrist_y = series(data, cols, f"{hand}_wrist", "y")
    nose_y = series(data, cols, "nose", "y")

    # SERVE: wrist gets well above the head at any point in the stroke window
    if (wrist_y[s:e + 1] < nose_y[s:e + 1] - 0.05).any():
        return "serve"

    wrist_x = series(data, cols, f"{hand}_wrist", "x")
    hip_mid_x = (series(data, cols, "left_hip", "x")
                 + series(data, cols, "right_hip", "x")) / 2
    # extreme wrist offset from body midline over the stroke window:
    # forehands reach farthest on the dominant side, backhands across the body
    offsets = wrist_x[s:e + 1] - hip_mid_x[s:e + 1]
    offset = float(offsets[np.argmax(np.abs(offsets))])

    # which side is "dominant side" in image coords depends on facing
    # direction; use shoulder ordering as facing proxy
    rs = series(data, cols, "right_shoulder", "x")[p]
    ls = series(data, cols, "left_shoulder", "x")[p]
    facing_right = ls < rs  # left shoulder appears left of right shoulder

    dominant_side_positive = (hand == "right") == facing_right
    on_dominant_side = (offset > 0) == dominant_side_positive
    return "forehand" if on_dominant_side else "backhand"


# -------------------------------------------------------------------- metrics

def angle(a, b, c):
    """Angle ABC in degrees from three 2D points."""
    ba, bc = a - b, c - b
    cosang = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-9)
    return float(np.degrees(np.arccos(np.clip(cosang, -1, 1))))


def pt(data, cols, lm, i):
    return np.array([series(data, cols, lm, "x")[i],
                     series(data, cols, lm, "y")[i]])


@dataclass
class Stroke:
    idx: int
    t_start: float
    t_impact: float
    t_end: float
    kind: str
    peak_wrist_speed: float
    prep_time_s: float
    follow_through_s: float
    elbow_angle_impact: float
    x_factor_prep: float          # hip–shoulder separation proxy (deg)
    contact_vs_front_hip: float   # +ahead / -behind, normalized units
    contact_height_frac: float    # 0=feet .. 1=head; serve can exceed 1
    knee_bend_min: float          # serve only: min knee angle in windup


def measure(data, cols, hand, fps, s, p, e, kind, speed):
    t = data[:, cols["time_s"]]

    elbow = angle(pt(data, cols, f"{hand}_shoulder", p),
                  pt(data, cols, f"{hand}_elbow", p),
                  pt(data, cols, f"{hand}_wrist", p))

    # X-factor proxy: |shoulder-line angle − hip-line angle| at prep
    prep = max(s, p - int((p - s) * 0.7))
    def line_angle(l1, l2, i):
        a, b = pt(data, cols, l1, i), pt(data, cols, l2, i)
        return np.degrees(np.arctan2(b[1] - a[1], b[0] - a[0]))
    xf = abs(line_angle("left_shoulder", "right_shoulder", prep)
             - line_angle("left_hip", "right_hip", prep))
    xf = min(xf, 360 - xf)

    # contact point vs front hip (x axis; sign normalized by facing)
    rs = series(data, cols, "right_shoulder", "x")[p]
    ls = series(data, cols, "left_shoulder", "x")[p]
    facing = 1.0 if ls < rs else -1.0
    hips_x = [series(data, cols, "left_hip", "x")[p],
              series(data, cols, "right_hip", "x")[p]]
    front_hip_x = max(hips_x) if facing > 0 else min(hips_x)
    contact_ahead = (series(data, cols, f"{hand}_wrist", "x")[p]
                     - front_hip_x) * facing

    # contact height fraction (0 = ankles, 1 = top of head)
    ankle_y = (series(data, cols, "left_ankle", "y")[p]
               + series(data, cols, "right_ankle", "y")[p]) / 2
    head_y = series(data, cols, "nose", "y")[p] - 0.06  # approx crown
    wrist_y = series(data, cols, f"{hand}_wrist", "y")[p]
    height_frac = (ankle_y - wrist_y) / max(ankle_y - head_y, 1e-6)

    # knee bend (serve): min knee angle during windup
    knee_min = float("nan")
    if kind == "serve":
        angles = []
        for i in range(s, p):
            for side in ("left", "right"):
                angles.append(angle(pt(data, cols, f"{side}_hip", i),
                                    pt(data, cols, f"{side}_knee", i),
                                    pt(data, cols, f"{side}_ankle", i)))
        if angles:
            knee_min = min(angles)

    return Stroke(
        idx=0,
        t_start=round(t[s], 2), t_impact=round(t[p], 2), t_end=round(t[e], 2),
        kind=kind,
        peak_wrist_speed=round(float(speed[p]), 3),
        prep_time_s=round(t[p] - t[s], 2),
        follow_through_s=round(t[e] - t[p], 2),
        elbow_angle_impact=round(elbow, 1),
        x_factor_prep=round(xf, 1),
        contact_vs_front_hip=round(float(contact_ahead), 3),
        contact_height_frac=round(float(height_frac), 2),
        knee_bend_min=round(knee_min, 1) if not np.isnan(knee_min) else knee_min,
    )


# ------------------------------------------------------------------- coaching

TARGETS = {
    "forehand": {
        "x_factor_prep": (25, 55, "Coil more in prep — turn shoulders past hips",
                          "Good separation"),
        "contact_vs_front_hip": (0.02, 0.30, "Contact is late — meet the ball out front",
                                 "Contact point out front — good"),
        "elbow_angle_impact": (120, 170, "Arm too cramped at contact — extend through the ball",
                               "Good extension at contact"),
    },
    "backhand": {
        "x_factor_prep": (20, 50, "More shoulder turn in prep",
                          "Good coil"),
        "contact_vs_front_hip": (0.03, 0.35, "Contact is late — backhands need earlier contact than forehands",
                                 "Early contact — good"),
    },
    "serve": {
        "knee_bend_min": (100, 140, "Deepen knee bend for more leg drive",
                          "Good leg loading"),
        "contact_height_frac": (1.05, 1.5, "Contact too low — full extension, reach up",
                                "Full extension at contact"),
        "elbow_angle_impact": (150, 180, "Arm not fully extended at contact",
                               "Full arm extension"),
    },
}


def coach(stroke: Stroke):
    notes = []
    for metric, (lo, hi, fix, praise) in TARGETS.get(stroke.kind, {}).items():
        v = getattr(stroke, metric)
        if v is None or (isinstance(v, float) and np.isnan(v)):
            continue
        notes.append(praise if lo <= v <= hi else fix)
    return notes


# --------------------------------------------------------------------- report

def run(pose_csv, hand, label_mode=False):
    data, cols = load_pose_csv(pose_csv)
    t = data[:, cols["time_s"]]
    fps = 1.0 / np.median(np.diff(t)) if len(t) > 1 else 30.0

    speed = wrist_speed(data, cols, hand, fps)
    segs = segment_strokes(speed, fps)
    print(f"{len(segs)} strokes detected ({fps:.0f} fps effective)")

    strokes = []
    for i, (s, p, e) in enumerate(segs):
        kind = classify(data, cols, hand, s, p, e)
        if label_mode:
            ans = input(f"Stroke {i} @ {t[p]:.1f}s auto={kind} "
                        f"[f/b/s/skip, enter=accept]: ").strip().lower()
            kind = {"f": "forehand", "b": "backhand", "s": "serve"}.get(ans, kind)
            if ans == "skip":
                continue
        st = measure(data, cols, hand, fps, s, p, e, kind, speed)
        st.idx = i
        strokes.append(st)

    base = pose_csv.rsplit(".", 1)[0]

    # per-stroke CSV (doubles as ML training data once labeled)
    with open(f"{base}_strokes.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(asdict(strokes[0]).keys()) if strokes else ["idx"])
        w.writeheader()
        for st in strokes:
            w.writerow(asdict(st))

    # session report
    lines = [f"# Session Report — {pose_csv}\n"]
    for kind in ("forehand", "backhand", "serve"):
        group = [s for s in strokes if s.kind == kind]
        if not group:
            continue
        lines.append(f"\n## {kind.title()} ({len(group)} strokes)\n")
        avg_speed = np.mean([s.peak_wrist_speed for s in group])
        lines.append(f"- Avg peak wrist speed: {avg_speed:.2f} (consistency "
                     f"CV: {np.std([s.peak_wrist_speed for s in group]) / max(avg_speed, 1e-9):.0%})")
        for st in group:
            notes = coach(st)
            lines.append(f"- t={st.t_impact}s: " + ("; ".join(notes) if notes else "measured"))

    report = "\n".join(lines)
    with open(f"{base}_report.md", "w") as f:
        f.write(report)

    print(report)
    print(f"\nSaved: {base}_strokes.csv, {base}_report.md")
    return strokes


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("pose_csv")
    ap.add_argument("--hand", choices=["left", "right"], default="right",
                    help="dominant (racket) hand")
    ap.add_argument("--label", action="store_true",
                    help="interactively confirm stroke labels (builds ML training data)")
    args = ap.parse_args()
    run(args.pose_csv, args.hand, args.label)
