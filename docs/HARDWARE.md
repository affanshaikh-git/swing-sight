# DIY Racket Sensor (Phase 3)

Butt-cap-mounted IMU streaming stroke data over BLE. Target: <15g added mass, ~$25–35 in parts, 3+ hour battery.

## Parts list

| Part | ~Cost | Notes |
|---|---|---|
| Seeed XIAO ESP32-S3 | $8–10 | 21×17.5mm, BLE 5, built-in LiPo charging. ESP32-C3 variant is $5 if you don't need the extra headroom |
| ICM-20948 breakout | $10–15 | 9-axis, gyro to ±2000°/s + better accel range than MPU-6050 |
| ADXL375 breakout (optional) | $8 | ±200g accelerometer — captures true impact shock, which saturates normal IMUs |
| 150mAh LiPo (502025 size) | $6 | ~3–4hr at 500Hz sampling + BLE |
| Case | — | 3D print a butt-cap sleeve, or heat-shrink + hook-and-loop strap for v1 |

**Why not MPU-6050 ($3):** its gyro saturates at 2000°/s and accel at ±16g. Recreational swings often stay under that; fast serves won't. Fine for a first prototype, but you'll replace it.

**Mount location:** butt cap. Lowest impact on swing weight, standard location for commercial sensors (Zepp, HEAD), and rotation there is still fully informative for swing-plane and racket-head-speed estimation (multiply angular velocity by racket length).

## Firmware outline (Arduino/ESP-IDF)

1. Sample IMU at 500–1000Hz into a ring buffer (last ~2s)
2. Impact detection: accel magnitude spike above threshold (start ~8g on ICM-20948, tune down; use ADXL375 channel if fitted)
3. On impact: freeze the window (1s pre + 0.5s post), timestamp it, queue for BLE
4. Stream stroke windows as notifications to phone/laptop; log to flash as backup
5. Deep sleep after 5 min without motion (wake on IMU interrupt)

## Data payload per stroke

- 6-axis (or 9-axis) IMU window @ 500Hz+
- Impact timestamp (ms since boot — aligned to video via the 3-tap sync event)
- Derived on-device or host-side: peak angular velocity → racket-head speed proxy, impact shock magnitude, swing plane orientation
- **Spin-type classification (racket-sensor exclusive):** swing-path direction (low-to-high = topspin, high-to-low = slice, flat = level) + racket-face angle from gyro at impact → {topspin, slice, flat} label per stroke. The camera cannot see racket face; this is the IMU's unique contribution to the fused taxonomy (see PLAN.md)

## Integration with the pipeline

- Host-side script pairs stroke windows to video-detected strokes by timestamp (after tap-sync alignment + cross-correlation on impact events)
- Adds to each stroke row: `racket_head_speed`, `impact_g`, `swing_plane_deg`
- These columns append to `*_strokes.csv` — same file the learned classifier trains on

## Safety/practical notes

- Balance check: 15g at the butt cap shifts balance ~2–3mm toward the handle — most players won't notice; heavier builds will change feel
- Secure everything — a battery flying off mid-serve is a bad day. Strain-relief the LiPo leads
- Charge LiPo only via the XIAO's onboard charger or a proper 1S charger
