# Capture Protocol

Consistent capture = comparable metrics across sessions. Follow this every time.

## Camera setup

- **Position:** side-on to your hitting direction, perpendicular to the baseline
- **Distance:** 4–5m from your hitting zone — full body in frame with ~0.5m margin above raised arm (serve reach)
- **Height:** tripod at roughly hip-to-chest height
- **Settings:** 1080p @ 60fps minimum (slo-mo 120/240fps works too; higher fps = better impact timing)
- **Lighting:** avoid shooting into the sun; backlit silhouettes degrade pose tracking
- **Framing:** mark your tripod spot (tape/court landmark) and reuse it — metric comparability depends on it

## Session structure

1. **Sync event:** start every recording with 3 sharp racket taps on the strings, visible to the camera. This spikes both the Watch and racket-sensor IMU streams for time alignment later.
2. **Blocks by stroke:** hit in blocks (e.g., 20 forehands, 20 backhands, 10 serves) rather than mixed rallies for the first few sessions — makes labeling faster and thresholds easier to tune.
3. **One player in frame:** the pipeline tracks a single person. Keep hitting partners/ball machines out of frame or clearly in the background.

## Serve-specific

- A second angle from behind the baseline captures toss height and contact height more accurately — optional but valuable
- If using one camera for serves, side-on still works; contact height and knee bend are the primary metrics

## After the session

```bash
python pose_extract.py session.mp4
python stroke_analyzer.py session_pose.csv --hand right --label
```

Use `--label` for the first ~10 sessions — the confirmed labels become training data for the learned classifier.

## Common capture problems

| Symptom | Cause | Fix |
|---|---|---|
| Missed strokes | wrist leaves frame at full extension | step back / widen framing |
| Jittery metrics | low light or <60fps | more light, higher fps |
| Wrong FH/BH labels | camera not perpendicular to hitting direction | square the tripod to the baseline |
| No pose detected | backlighting / player too small in frame | reposition camera |
