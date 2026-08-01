# ExamShield AI — System Architecture

## High-Level Overview

```
┌──────────────────────────────────────────────────────┐
│                   webcam frames                     │
│   Camera (─────────────────────────────────────) │
│                                                      │
│  Thread 1: _process_loop (MediaPipe + scoring)       │
│    ├── MediaPipe FaceLandmarker (async callback)      │
│    │     ├── 468-point face mesh (smoothed 9-frame)   │
│    │     ├── Blendshapes → gaze direction              │
│    │     └── Landmarks  → head pose                    │
│    ├── MediaPipe HandLandmarker (async callback)      │
│    │     └── Hand proximity → hand_near_face flag      │
│    ├── CNN Models (MobileNetV2, optional override)    │
│    ├── _process_violations() → debounce + hysteresis   │
│    ├── _update_score() → weighted sum + decay          │
│    └── _annotate() → MJPEG stream frame               │
│                                                      │
│  Thread 2: _yolo_loop (phone + face detection)       │
│    ├── YOLOv8n inference on raw frame every 0.5s      │
│    ├── phone_detected flag update                     │
│    └── face_count update (if yolo_face model present)  │
│                                                      │
│  Thread 3: _verify_identity_async (face_recognition) │
│    └── dlib embedding comparison every 2s             │
└──────────────────────────────────────────────────────┘
         │ camera.suspicion_score, risk_level, events
         ▼
┌──────────────────────────────────────────────────────┐
│  Flask Application (app.py)                          │
│    ├── 18 REST endpoints (/stats, /video, /history...) │
│    ├── MJPEG stream (/video)                          │
│    ├── Flask-SocketIO broadcast (1s interval)         │
│    └── Session auth (/admin/login)                    │
└──────────────────────────────────────────────────────┘
         │ JSON / MJPEG / WebSocket
         ▼
┌──────────────────────────────────────────────────────┐
│  Browser Dashboards (frontend/)                      │
│    ├── Candidate Monitor (index.html)                │
│    │     ├── MJPEG live feed                          │
│    │     ├── Chart.js score timeline                  │
│    │     └── Violation counters + event feed          │
│    └── Admin Dashboard (admin.html)                  │
│          ├── Candidate cards with live metrics        │
│          └── Session history + CSV export             │
└──────────────────────────────────────────────────────┘
         │ SQL
         ▼
┌──────────────────────────────────────────────────────┐
│  SQLite Database (db.py)                             │
│    ├── events table (per-violation log)               │
│    └── sessions table (per-session summary)           │
└──────────────────────────────────────────────────────┘
```

## Threading Model

The `Camera` class runs three concurrent threads:

| Thread | Frequency | Responsibility |
|--------|-----------|----------------|
| `_process_loop` | ~30 FPS | Frame capture, MediaPipe inference, scoring, annotation |
| `_yolo_loop` | 2 Hz (0.5s) | YOLO phone + face inference on raw frame |
| `_verify_identity_async` | 0.5 Hz (2s) | face_recognition dlib embedding comparison |

All threads share state via a `threading.Lock()`. The YOLO and identity threads read `_raw_frame` (an unannotated copy), avoiding any contention with the annotation pipeline.

## Signal Fusion

Each frame, all active violation signals are combined into a single `suspicion_score` in [0, 100]:

```
suspicion_score += sum(SCORE_WEIGHTS[signal] for each active violation)
suspicion_score = min(100, suspicion_score)

# Decay when no violations are active:
if no_violations_active:
    suspicion_score = max(0, suspicion_score - SCORE_DECAY)
```

### Violation Guard System

Each signal goes through three layers of filtering before incrementing its violation counter:

1. **Debounce**: Condition must persist continuously for `DEBOUNCE_DURATION` seconds
2. **Hysteresis**: Condition must be clear for `CLEAR_DURATION` seconds before resetting
3. **Cooldown**: Same event type is not logged more than once per `EVENT_COOLDOWN` seconds

