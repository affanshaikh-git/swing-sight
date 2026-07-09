"""
audio_impacts.py — detect ball-impact timestamps from the session video's
audio track (the "pock" is a sharp broadband transient) and optionally refine
the stroke timings in a *_strokes.csv from stroke_analyzer.py.

Audio gives sub-frame impact timing — better than the wrist-speed peak, and
the anchor for syncing racket-IMU data later.

Usage:
    python audio_impacts.py session.mp4
    python audio_impacts.py session.mp4 --strokes session_pose_strokes.csv

Outputs:
    session_impacts.csv                 (all detected impacts: time_s, strength)
    session_pose_strokes.csv updated    (adds t_impact_audio column, if --strokes)
"""

import argparse
import csv
import os
import subprocess
import sys
import tempfile

import numpy as np
from scipy.io import wavfile
from scipy.signal import butter, sosfilt, find_peaks

SR = 16000  # analysis sample rate


def extract_audio(video_path: str) -> tuple[int, np.ndarray]:
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp.close()
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-i", video_path,
           "-ac", "1", "-ar", str(SR), tmp.name]
    subprocess.run(cmd, check=True)
    sr, audio = wavfile.read(tmp.name)
    os.unlink(tmp.name)
    return sr, audio.astype(np.float64)


def detect_impacts(sr: int, audio: np.ndarray,
                   min_gap_s: float = 0.5,
                   sensitivity: float = 6.0):
    """
    Ball impacts are short broadband transients. Band-pass 1–6 kHz (kills
    wind/voice rumble), rectify, smooth to an energy envelope, then pick
    peaks that stand `sensitivity` MADs above the median envelope.
    Returns list of (time_s, strength) where strength is peak/median ratio.
    """
    audio = audio / (np.max(np.abs(audio)) + 1e-9)
    sos = butter(4, [1000, 6000], btype="band", fs=sr, output="sos")
    band = sosfilt(sos, audio)

    env = np.abs(band)
    win = int(sr * 0.01)  # 10ms smoothing
    kernel = np.ones(win) / win
    env = np.convolve(env, kernel, mode="same")

    med = np.median(env)
    mad = np.median(np.abs(env - med)) + 1e-9
    thresh = med + sensitivity * mad

    peaks, props = find_peaks(env, height=thresh, distance=int(min_gap_s * sr))
    out = [(p / sr, float(props["peak_heights"][i] / (med + 1e-9)))
           for i, p in enumerate(peaks)]
    return [(t, s) for t, s in out if s >= 3.0]  # drop weak noise peaks


def refine_strokes(strokes_csv: str, impacts, tolerance_s: float = 0.35):
    """Attach nearest audio impact (within tolerance) to each stroke row."""
    with open(strokes_csv) as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return 0

    times = np.array([t for t, _ in impacts])
    matched = 0
    for row in rows:
        t = float(row["t_impact"])
        if len(times) == 0:
            row["t_impact_audio"] = ""
            continue
        i = int(np.argmin(np.abs(times - t)))
        if abs(times[i] - t) <= tolerance_s:
            row["t_impact_audio"] = round(times[i], 3)
            matched += 1
        else:
            row["t_impact_audio"] = ""

    with open(strokes_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    return matched


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--strokes", default=None,
                    help="*_strokes.csv from stroke_analyzer.py to refine")
    ap.add_argument("--min-gap", type=float, default=0.5,
                    help="min seconds between impacts (default 0.5)")
    ap.add_argument("--sensitivity", type=float, default=6.0,
                    help="lower = more detections (default 6.0)")
    args = ap.parse_args()

    sr, audio = extract_audio(args.video)
    impacts = detect_impacts(sr, audio, args.min_gap, args.sensitivity)
    print(f"{len(impacts)} impacts detected")

    out = args.video.rsplit(".", 1)[0] + "_impacts.csv"
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["time_s", "strength"])
        for t, s in impacts:
            w.writerow([round(t, 3), round(s, 1)])
    print(f"Saved {out}")

    if args.strokes:
        n = refine_strokes(args.strokes, impacts)
        print(f"Matched audio impacts to {n} strokes → t_impact_audio column "
              f"added in {args.strokes}")

    # Sync hint: the 3-tap session-start event shows up as 3 tightly spaced
    # impacts near t=0 — use those to align racket-IMU / watch streams.
    early = [t for t, _ in impacts if t < 15]
    if len(early) >= 3:
        print(f"Possible 3-tap sync event around t={early[0]:.1f}–{early[2]:.1f}s")
