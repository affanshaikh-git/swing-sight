"""
pose_extract.py — Step 1 of the tennis stroke trainer pipeline.

Extracts pose landmarks from a session video using the MediaPipe Tasks API
(PoseLandmarker, works with mediapipe >= 0.10) and writes them to CSV.

On first run it downloads the pose model (~9MB) automatically.

Usage:
    python pose_extract.py session.mp4 [--out session_pose.csv] [--model full]

Output CSV columns:
    frame, time_s, then x/y/z/visibility for each of the 33 pose landmarks
    (normalized image coordinates; z is relative depth).
"""

import argparse
import os
import sys
import csv
import urllib.request

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

# 33 pose landmarks, standard MediaPipe order
LANDMARK_NAMES = [
    "nose", "left_eye_inner", "left_eye", "left_eye_outer",
    "right_eye_inner", "right_eye", "right_eye_outer",
    "left_ear", "right_ear", "mouth_left", "mouth_right",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_pinky", "right_pinky",
    "left_index", "right_index", "left_thumb", "right_thumb",
    "left_hip", "right_hip", "left_knee", "right_knee",
    "left_ankle", "right_ankle", "left_heel", "right_heel",
    "left_foot_index", "right_foot_index",
]

MODEL_URLS = {
    "lite": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task",
    "full": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task",
    "heavy": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task",
}


def get_model(variant: str) -> str:
    path = f"pose_landmarker_{variant}.task"
    if not os.path.exists(path):
        print(f"Downloading pose model ({variant})...")
        urllib.request.urlretrieve(MODEL_URLS[variant], path)
    return path


def extract(video_path: str, out_csv: str, variant: str = "full") -> None:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        sys.exit(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Video: {video_path} | {fps:.1f} fps | {total} frames")

    options = vision.PoseLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=get_model(variant)),
        running_mode=vision.RunningMode.VIDEO,
        min_pose_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    header = ["frame", "time_s"]
    for name in LANDMARK_NAMES:
        header += [f"{name}_x", f"{name}_y", f"{name}_z", f"{name}_vis"]

    n_written = 0
    with vision.PoseLandmarker.create_from_options(options) as landmarker, \
            open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)

        frame_idx = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            ts_ms = int(frame_idx / fps * 1000)
            result = landmarker.detect_for_video(mp_image, ts_ms)

            if result.pose_landmarks:
                lms = result.pose_landmarks[0]  # first (only) person
                row = [frame_idx, round(frame_idx / fps, 4)]
                for lm in lms:
                    vis = getattr(lm, "visibility", None)
                    row += [round(lm.x, 5), round(lm.y, 5), round(lm.z, 5),
                            round(vis, 3) if vis is not None else 1.0]
                writer.writerow(row)
                n_written += 1

            frame_idx += 1
            if frame_idx % 300 == 0:
                print(f"  {frame_idx}/{total} frames...")

    cap.release()
    print(f"Done. {n_written} frames with pose → {out_csv}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--out", default=None)
    ap.add_argument("--model", choices=["lite", "full", "heavy"], default="full",
                    help="lite=fastest, heavy=most accurate (default: full)")
    args = ap.parse_args()
    out = args.out or args.video.rsplit(".", 1)[0] + "_pose.csv"
    extract(args.video, out, args.model)
