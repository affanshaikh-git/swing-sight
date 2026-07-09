# 🎾 swingsight

Computer vision + sensor fusion for tennis stroke training. Films your session, finds every forehand, backhand, and serve, and tells you what to fix.

## How it works

```
Phone video (60fps) ──▶ MediaPipe pose ──▶ stroke segmentation ──▶ FH/BH/serve
Apple Watch IMU ─────▶ swing tempo      ──▶ classification     ──▶ per-stroke metrics
DIY racket sensor ───▶ impact detection ──▶                    ──▶ coaching notes
```

Phase 1 (this repo, working today) is the vision pipeline. Phases 2–4 add Apple Watch IMU, a DIY ESP32 racket sensor, and real-time audio feedback.

**Docs:** [Full plan](docs/PLAN.md) · [Capture protocol](docs/CAPTURE.md) · [Racket sensor build](docs/HARDWARE.md)

## Quick start

```bash
pip install -r requirements.txt

# 1. Film side-on at 60fps, tripod at baseline, 4–5m from hitting zone
# 2. Extract pose (downloads the ~9MB model on first run)
python pose_extract.py session.mp4

# 3. Analyze
python stroke_analyzer.py session_pose.csv --hand right
```

Outputs:
- `session_pose_strokes.csv` — per-stroke metrics (also your future ML training data)
- `session_pose_report.md` — session report with coaching notes

## What it measures

| Stroke | Metrics |
|---|---|
| All | peak wrist speed, prep time, follow-through, consistency (CV) |
| Forehand / Backhand | hip–shoulder separation (X-factor), contact point vs front hip, elbow extension at contact |
| Serve | knee bend depth, contact height (% of body height), arm extension |

## Building the training dataset

Run with `--label` to interactively confirm each stroke's auto-classification:

```bash
python stroke_analyzer.py session_pose.csv --hand right --label
```

The labeled CSVs feed a learned classifier (gradient boosting / 1D-CNN) that replaces the rule-based one after ~10 sessions.

## Roadmap

- [x] Pose extraction + stroke segmentation + rule-based classification
- [x] Per-stroke biomechanics metrics + coaching heuristics
- [ ] Apple Watch IMU sync (toss-arm timing on serve, tempo, stroke count)
- [ ] DIY racket sensor — XIAO ESP32-S3 + ICM-20948 at the butt cap (~$25, <15g)
- [ ] Fused stroke taxonomy: {FH, BH} × {topspin, slice, flat} + serve/smash/volley — camera classifies the wing, racket IMU classifies the spin
- [ ] Composite per-stroke-type score for session-over-session trends
- [ ] Learned stroke classifier from labeled sessions
- [ ] TrackNet-style ball tracking (speed, depth, placement)
- [ ] Real-time audio cues on court

## Notes

- Single markerless camera → depth is estimated; side-on framing minimizes error
- Metrics are trend tools (you vs. you across sessions), not lab-grade biomechanics
- Keep camera framing consistent between sessions so numbers are comparable
