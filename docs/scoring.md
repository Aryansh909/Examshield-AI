# ExamShield AI — Scoring Algorithm

## Suspicion Score

The suspicion score is a real-valued number in [0.0, 100.0]. It represents the real-time level of detected suspicious behaviour.

### Weights

| Violation Type | Weight | Rationale |
|---|---|---|
| `identity_mismatch` | **25** | Highest — definitive cheating evidence |
| `phone_detected` | **20** | High confidence YOLO detection |
| `multiple_faces` | **18** | Another person visible |
| `no_face` | **10** | Candidate left frame |
| `head_turned` | **8** | Strong behavioural signal |
| `gaze_off_screen` | **8** | Strong behavioural signal |
| `mouth_open` | **5** | Possible verbal communication |
| `hand_near_face` | **4** | Lowest — high false positive rate |

### Decay

When no violations are active, the score decays by `0.3` per processed frame:
```
score = max(0, score - 0.3)
```
This means a score of 100 reaches 0 in approximately 33 seconds of clear behaviour.

### Risk Levels

| Level | Score Range | UI Colour |
|-------|-------------|----------|
| LOW | 0 – 39 | Green |
| MEDIUM | 40 – 69 | Yellow/Amber |
| HIGH | 70 – 100 | Red |

### Evidence Capture

When `risk_level == "HIGH"`, an annotated frame is saved to `snapshots/` as a JPEG. A 120-second cooldown prevents disk flooding.

## Violation Guard System

Every signal passes through three filtering layers:

```
Raw Signal → [Debounce] → [Hysteresis] → [Cooldown] → Violation Logged
```

### 1. Debounce
Condition must persist for a minimum duration before counting:

| Signal | Duration |
|--------|----------|
| Gaze off-screen | 2.0s |
| Head turned | 1.5s |
| Mouth open | 2.0s |
| Phone detected | 1.5s |
| Multiple faces | 1.5s |
| No face | 3.0s |
| Hand near face | 1.0s |

### 2. Hysteresis
Once active, the condition must be **continuously clear** for the hysteresis duration before the active flag resets:

| Signal | Clear Duration |
|--------|---------------|
| Gaze | 3.0s on-screen |
| Head | 2.0s forward |
| Mouth | 2.0s closed |
| Phone | 2.5s absent |

This prevents flickering signals from generating repeated violations.

### 3. Cooldown
Even if debounce passes, the same event type cannot be logged again within the cooldown window:

| Event | Cooldown |
|-------|----------|
| `gaze_off_screen` | 30s |
| `head_turned` | 25s |
| `mouth_open` | 25s |
| `phone_detected` | 45s |
| `multiple_faces` | 45s |
| `identity_mismatch` | 60s |
| `hand_near_face` | 20s |

### 4. State Smoothing

MediaPipe blendshape readings are inherently noisy (per-frame). All three posture signals are smoothed via a **majority vote over a rolling buffer**:

| Signal | Buffer Size | Effective Window (~9 FPS) |
|--------|-------------|---------------------------|
| Head direction | 9 frames | ~1 second |
| Gaze direction | 9 frames | ~1 second |
| Mouth state | 7 frames | ~0.8 seconds |
