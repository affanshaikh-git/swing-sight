"""
court_heatmap.py — player movement heatmap projected onto a real court.

Camera pixels aren't court meters, so this uses a homography: you supply the
pixel coordinates of 4 known court-line intersections (from any camera angle
— side-on, net post, or elevated), and every ankle position from the pose CSV
is projected into court coordinates before building the heatmap.

Getting the 4 pixel points (one-time per camera setup — reuse if the tripod
spot is taped):
    python court_heatmap.py --grab-frame session.mp4
    → saves frame.png; open it, note the pixel (x, y) of the 4 corners of the
      NEAR singles court half you play in (or full singles court if visible).

Usage:
    python court_heatmap.py session_pose.csv \
        --video-size 1920 1080 \
        --preset near-half-singles \
        --pixels "x1,y1 x2,y2 x3,y3 x4,y4"

Pixel order must match the preset's corner order (printed by --list-presets).

Output: session_pose_heatmap.png
"""

import argparse
import csv
import sys

import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

# Court coordinate system: meters. Origin = center of the net at ground level.
# +x runs toward the player's baseline, +y runs to the player's right→left
# along the net. Player's baseline is x = 11.885.
COURT_L = 11.885          # net to baseline
SINGLES_W = 8.23
DOUBLES_W = 10.97
SERVICE_L = 6.40          # net to service line

# Presets: named corner sets, each corner given in court meters.
# Order matters — supply --pixels in the same order.
PRESETS = {
    # corners of the singles court on the player's side of the net
    "near-half-singles": [
        (0.0, -SINGLES_W / 2),        # net x singles sideline (right)
        (0.0, +SINGLES_W / 2),        # net x singles sideline (left)
        (COURT_L, +SINGLES_W / 2),    # baseline x singles sideline (left)
        (COURT_L, -SINGLES_W / 2),    # baseline x singles sideline (right)
    ],
    # service boxes only (all 4 corners of the near service area)
    "near-service-boxes": [
        (0.0, -SINGLES_W / 2),
        (0.0, +SINGLES_W / 2),
        (SERVICE_L, +SINGLES_W / 2),
        (SERVICE_L, -SINGLES_W / 2),
    ],
    # doubles court near half
    "near-half-doubles": [
        (0.0, -DOUBLES_W / 2),
        (0.0, +DOUBLES_W / 2),
        (COURT_L, +DOUBLES_W / 2),
        (COURT_L, -DOUBLES_W / 2),
    ],
}


def grab_frame(video_path):
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(cap.get(cv2.CAP_PROP_FRAME_COUNT) // 2))
    ok, frame = cap.read()
    cap.release()
    if not ok:
        sys.exit("Could not read video")
    cv2.imwrite("frame.png", frame)
    print("Saved frame.png — note pixel coords of your 4 preset corners")


def load_ankles(pose_csv, w, h):
    """Return Nx2 pixel coords of the ankle midpoint per frame."""
    with open(pose_csv) as f:
        reader = csv.DictReader(f)
        pts = []
        for row in reader:
            lx = float(row["left_ankle_x"]) * w
            ly = float(row["left_ankle_y"]) * h
            rx = float(row["right_ankle_x"]) * w
            ry = float(row["right_ankle_y"]) * h
            pts.append([(lx + rx) / 2, (ly + ry) / 2])
    return np.array(pts, dtype=np.float64)


def draw_court(ax):
    """Near half-court lines in court coordinates (x: 0..baseline+run-off)."""
    line = dict(color="white", lw=2)
    # doubles + singles sidelines
    for wdt in (DOUBLES_W, SINGLES_W):
        ax.plot([0, COURT_L], [-wdt / 2, -wdt / 2], **line)
        ax.plot([0, COURT_L], [+wdt / 2, +wdt / 2], **line)
    ax.plot([COURT_L, COURT_L], [-DOUBLES_W / 2, DOUBLES_W / 2], **line)  # baseline
    ax.plot([0, 0], [-DOUBLES_W / 2, DOUBLES_W / 2], color="#222", lw=4)   # net
    ax.plot([SERVICE_L, SERVICE_L], [-SINGLES_W / 2, SINGLES_W / 2], **line)
    ax.plot([0, SERVICE_L], [0, 0], **line)                                # center service
    ax.plot([COURT_L, COURT_L - 0.2], [0, 0], **line)                      # center mark


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pose_csv", nargs="?")
    ap.add_argument("--grab-frame", metavar="VIDEO", default=None)
    ap.add_argument("--list-presets", action="store_true")
    ap.add_argument("--preset", default="near-half-singles",
                    choices=list(PRESETS.keys()))
    ap.add_argument("--pixels", default=None,
                    help='4 pixel points: "x1,y1 x2,y2 x3,y3 x4,y4"')
    ap.add_argument("--video-size", nargs=2, type=int, default=[1920, 1080],
                    metavar=("W", "H"))
    args = ap.parse_args()

    if args.list_presets:
        for name, corners in PRESETS.items():
            print(f"{name}: corners in order {corners}")
        return
    if args.grab_frame:
        grab_frame(args.grab_frame)
        return
    if not args.pose_csv or not args.pixels:
        sys.exit("Need pose_csv and --pixels (or use --grab-frame / --list-presets)")

    px = np.array([[float(v) for v in p.split(",")]
                   for p in args.pixels.split()], dtype=np.float64)
    court = np.array(PRESETS[args.preset], dtype=np.float64)
    if px.shape != (4, 2):
        sys.exit("--pixels needs exactly 4 x,y points")

    H, _ = cv2.findHomography(px, court)

    w, h = args.video_size
    ankles_px = load_ankles(args.pose_csv, w, h)
    ones = np.ones((len(ankles_px), 1))
    proj = (H @ np.hstack([ankles_px, ones]).T).T
    court_xy = proj[:, :2] / proj[:, 2:3]

    # keep points on/near the near half (allow 4m run-off behind baseline,
    # 2m beyond doubles lines)
    keep = ((court_xy[:, 0] > -0.5) & (court_xy[:, 0] < COURT_L + 4.0)
            & (np.abs(court_xy[:, 1]) < DOUBLES_W / 2 + 2.0))
    pts = court_xy[keep]
    print(f"{len(pts)}/{len(court_xy)} frames on court")

    # 2D histogram at 10cm resolution, gaussian smoothed
    x_edges = np.arange(-0.5, COURT_L + 4.0, 0.1)
    y_edges = np.arange(-DOUBLES_W / 2 - 2.0, DOUBLES_W / 2 + 2.0, 0.1)
    hist, _, _ = np.histogram2d(pts[:, 0], pts[:, 1], bins=[x_edges, y_edges])
    hist = gaussian_filter(hist, sigma=4)

    fig, ax = plt.subplots(figsize=(8, 10))
    ax.set_facecolor("#2f6690")
    ax.imshow(hist, origin="lower", cmap="hot", alpha=0.75,
              extent=[y_edges[0], y_edges[-1], x_edges[0], x_edges[-1]],
              aspect="equal")
    # imshow above plots x vertically: swap draw_court axes accordingly
    ax.clear()
    ax.set_facecolor("#2f6690")
    # transpose so court length runs up the page
    ax.imshow(hist, origin="lower", cmap="hot", alpha=0.8,
              extent=[y_edges[0], y_edges[-1], x_edges[0], x_edges[-1]],
              aspect="equal")

    # court lines with axes: horizontal = y (net direction), vertical = x
    line = dict(color="white", lw=2)
    for wdt in (DOUBLES_W, SINGLES_W):
        ax.plot([-wdt / 2, -wdt / 2], [0, COURT_L], **line)
        ax.plot([+wdt / 2, +wdt / 2], [0, COURT_L], **line)
    ax.plot([-DOUBLES_W / 2, DOUBLES_W / 2], [COURT_L, COURT_L], **line)
    ax.plot([-DOUBLES_W / 2, DOUBLES_W / 2], [0, 0], color="#111", lw=4)
    ax.plot([-SINGLES_W / 2, SINGLES_W / 2], [SERVICE_L, SERVICE_L], **line)
    ax.plot([0, 0], [0, SERVICE_L], **line)
    ax.plot([0, 0], [COURT_L, COURT_L - 0.25], **line)

    ax.set_xlim(y_edges[0], y_edges[-1])
    ax.set_ylim(x_edges[-1], -0.5)   # net at top, baseline+run-off at bottom
    ax.set_xticks([]), ax.set_yticks([])
    ax.set_title("Movement heatmap — near half court (net at top)",
                 fontsize=13, pad=12)

    out = args.pose_csv.rsplit(".", 1)[0] + "_heatmap.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="#f5f2ec")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
