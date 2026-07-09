# Tennis Stroke Trainer — System Plan

**Goal:** Train forehand, backhand, and serve using computer vision (phone) + IMU sensors (Apple Watch + DIY racket sensor), with post-session analysis first and real-time feedback later.

---

## Architecture at a Glance

```
┌─────────────┐   ┌──────────────┐   ┌──────────────────┐
│ Phone video │──▶│ Pose pipeline │──▶│                  │
│ (60fps)     │   │ (MediaPipe)   │   │  Fusion +        │
├─────────────┤   ├──────────────┤   │  Stroke engine   │──▶ Metrics, scores,
│ Apple Watch │──▶│ IMU pipeline  │──▶│  - segment       │    drill feedback
│ (CoreMotion)│   │ (100Hz accel/ │   │  - classify      │
├─────────────┤   │  gyro)        │   │  - measure       │
│ DIY racket  │──▶│ BLE stream    │   │                  │
│ sensor      │   │ (ESP32+IMU)   │   └──────────────────┘
└─────────────┘   └──────────────┘
```

**Why all three sources:**
- **Video/pose** → body mechanics: hip–shoulder separation, knee bend, contact point, stance, follow-through path
- **Watch IMU** (dominant wrist) → swing speed proxy, stroke count, timing consistency
- **Racket IMU** → the gold data: racket-head speed, impact detection (sharp accel spike), impact location consistency, swing plane

---

## Phase 1 — CV Pipeline (build now, prototype included)

**Capture protocol:**
- Phone on tripod, side-on view at baseline height, ~4–5m from hitting zone, 1080p @ 60fps
- Second angle (behind baseline) optional for serve — adds toss and contact-height accuracy
- Consistent framing session-to-session so metrics are comparable

**Pipeline (in the prototype code):**
1. `pose_extract.py` — video → MediaPipe Pose landmarks (33 joints, 3D) → CSV
2. `stroke_analyzer.py` — segments strokes via wrist-speed peaks, classifies FH/BH/serve with pose heuristics, computes per-stroke metrics, writes a session report

**Metrics computed per stroke:**

| Stroke | Metrics |
|---|---|
| All | wrist peak speed (px/s → normalized), prep time, follow-through length |
| Forehand/Backhand | hip–shoulder separation at prep (X-factor), contact point relative to front hip, elbow angle at contact |
| Serve | knee bend depth, trophy-position elbow angle, contact height (% of body height), toss-to-contact time |

**Classifier evolution:** Start rule-based (works day 1, no training data). After ~10 sessions, you'll have labeled clips → train a gradient-boosted classifier or 1D-CNN on windowed pose features (script stub included).

## Phase 2 — Apple Watch IMU

- **Easiest path:** SensorLog app ($5) — logs 100Hz accel/gyro to CSV, AirDrop after session
- **Better path:** small watchOS app using CoreMotion + `CMBatchedSensorManager` (200Hz during workouts) — I can write this when you're ready
- Sync with video via a "3 racket taps" clap event at session start (visible spike in both streams)
- Adds: stroke count without camera, tempo consistency, rough swing intensity

## Phase 3 — DIY Racket Sensor (~$25)

Fits your build style. Parts:

| Part | ~Cost | Notes |
|---|---|---|
| Seeed XIAO ESP32-S3 or ESP32-C3 | $8–10 | Tiny (21×17mm), BLE, battery charging built in |
| MPU-6050 breakout (or ICM-20948 for better range) | $3–12 | **Caveat:** MPU-6050 gyro saturates at 2000°/s — fine for learning, pros exceed it. ICM-20948 or an accel-only high-g add-on (ADXL375, ±200g) captures impact properly |
| 150mAh LiPo | $6 | ~3+ hr sessions |
| Case | — | 3D print or heat-shrink; mount at butt cap (lowest swing-weight impact, standard location for commercial sensors) |

Firmware: sample at 500–1000Hz into ring buffer, detect impact (accel spike > threshold), stream stroke windows over BLE to phone/laptop. Total added mass target: <15g.

## Phase 4 — Fusion + Real-Time

- Time-align streams (tap-sync + cross-correlation on impact events)
- Real-time: MediaPipe runs on-device (iOS) at 30fps+; watch/racket BLE latency <100ms → audio cue feedback ("late contact", "low toss") is feasible
- Long-term: per-stroke score trends, session dashboard

---

## Build Order

1. **This week:** Film one session (side-on, 60fps), run the prototype on it → baseline metrics
2. **Week 2–3:** Label strokes from session 1 output, tune segmentation thresholds
3. **Week 3–4:** Add Watch logging (SensorLog), sync, add tempo metrics
4. **Month 2:** Build racket sensor, integrate impact data
5. **Month 3:** Train learned classifier, prototype real-time audio cues

## Known Limitations

- MediaPipe is markerless single-camera → depth (Z) is estimated, side-on angle minimizes this
- Ball tracking not included in v1 (possible later with TrackNet) — contact inferred from racket-arm kinematics + IMU spike
- Metrics are *relative/trend* tools, not absolute biomechanics lab numbers — compare you-vs-you across sessions
