# 🎾 swingsight

Computer vision + sensor fusion for tennis stroke training. Films your session, finds every forehand, backhand, and serve, and tells you what to fix.

## How it works

```
Phone video (60fps) ──▶ MediaPipe pose ──▶ stroke segmentation ──▶ FH/BH/serve
Apple Watch IMU ─────▶ swing tempo      ──▶ classification     ──▶ per-stroke metrics
DIY racket sensor ───▶ impact detection ──▶                    ──▶ coaching notes
```

Phase 1 (this repo, working today) is the vision pipeline. Phases 2–4 add Apple Watch IMU, a DIY ESP32 racket sensor, and real-time audio feedback — see [docs/PLAN.md](docs/PLAN.md).

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
- [ ] Apple Watch IMU sync (tempo, stroke count without camera)
- [ ] DIY racket sensor — XIAO ESP32-S3 + ICM-20948 at the butt cap (~$25, <15g)
- [ ] Learned stroke classifier from labeled sessions
- [ ] Real-time audio cues on court

## Notes

- Single markerless camera → depth is estimated; side-on framing minimizes error
- Metrics are trend tools (you vs. you across sessions), not lab-grade biomechanics
- Keep camera framing consistent between sessions so numbers are comparable
