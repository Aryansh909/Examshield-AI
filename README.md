# ExamShield AI

**Computer Vision-Based Online Examination Proctoring System**

![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=flat-square&logo=python&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-3.0-000000?style=flat-square&logo=flask&logoColor=white)
![YOLOv8](https://img.shields.io/badge/YOLO-v8-FF6B35?style=flat-square)
![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10-0097A7?style=flat-square)
![PyTorch](https://img.shields.io/badge/PyTorch-2.2-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)

---

## Overview

ExamShield AI is a real-time, computer vision-based proctoring system that detects suspicious behaviour during online examinations. It combines YOLOv8 object detection, MediaPipe facial geometry, face recognition, and a probabilistic scoring engine to produce a live suspicion score — running entirely on a standard webcam with no cloud dependency.

**Stack:** Python · Flask · OpenCV · PyTorch · YOLOv8 · MediaPipe · SQLite · Chart.js

---

## Features

| Feature | Technology | Details |
|---|---|---|
| Phone Detection | YOLOv8n (COCO) | Dedicated inference thread at 2 Hz |
| Multi-Face Detection | YOLOv8n + MediaPipe | Detects additional persons in frame |
| Head Pose Estimation | MediaPipe 468-point mesh | Left/right/up/down with 9-frame majority smoothing |
| Gaze Tracking | MediaPipe blendshapes | 8 blendshapes fused, threshold 0.50, 9-frame smoothing |
| Mouth State | MediaPipe blendshapes | jawOpen score — open / speaking / closed |
| Hand Proximity | MediaPipe HandLandmarker | Detects hands near face region |
| Identity Verification | face_recognition + dlib | 128-D embedding comparison every 2s |
| Suspicion Scoring | Weighted sum + decay | 0–100 score, configurable weights and time-decay |
| Live Dashboard | Flask + Chart.js + MJPEG | Score graph, violation counters, annotated camera feed |
| Admin Dashboard | Flask-SocketIO | Multi-candidate monitoring with session history |
| Evidence Snapshots | OpenCV JPEG | Auto-saved at HIGH risk (120s cooldown) |
| Session Persistence | SQLite | All events and session summaries stored |
| CSV Export | Flask endpoint | Full session report download |
| WebSocket Push | Flask-SocketIO | Real-time telemetry broadcast |
| UI Themes | Vanilla CSS | Cyberpunk Neon, Emerald Matrix, Solar Cyber, Studio Light |
| CNN Training Pipeline | PyTorch MobileNetV2 | Fine-tunable classifiers for head/gaze/mouth |

---

## Architecture

```
Webcam (OpenCV)
     |
     v
+-------------------------------------------------------------+
|  Camera  (camera.py)                                        |
|                                                             |
|  Thread 1: _process_loop  (~30 FPS)                         |
|    +-- MediaPipe FaceLandmarker (async LIVE_STREAM)         |
|    |     +-- 468-point mesh  -> head pose (9-frame vote)    |
|    |     +-- Blendshapes     -> gaze + mouth (9/7-frame)    |
|    +-- MediaPipe HandLandmarker (async LIVE_STREAM)         |
|    +-- _process_violations()  [debounce + hysteresis]       |
|    +-- _update_score()        [weighted sum + decay]        |
|                                                             |
|  Thread 2: _yolo_loop  (2 Hz)                               |
|    +-- YOLOv8n inference -> phone_detected, face_count      |
|                                                             |
|  Thread 3: _verify_identity_async  (0.5 Hz)                 |
|    +-- dlib 128-D embedding -> identity_mismatch            |
+-------------------------------------------------------------+
     |
     v
+-------------------------------------------------------------+
|  Flask API  (app.py)  — 18 endpoints                        |
|    +-- /video     MJPEG annotated stream                    |
|    +-- /stats     JSON telemetry                            |
|    +-- /export    CSV session report                        |
|    +-- WebSocket  stats_update + alert broadcasts           |
+-------------------------------------------------------------+
     |
     v
Browser Dashboards (frontend/)        SQLite DB (db.py)
  +-- Candidate Monitor               +-- events table
  +-- Admin Dashboard                 +-- sessions table
```

### Violation Guard System

Each signal passes through three layers before a violation is counted:

```
Raw Signal -> [Debounce 1–3s] -> [Active Flag] -> [Cooldown check] -> Logged
                                       |
                           [Hysteresis: 2–3s clear] -> Flag Resets
```

### Scoring Formula

```
score += sum(WEIGHT[signal] for each active violation)
score  = max(0, score - 0.3)   # decay per frame when clear
score  = min(100, score)

LOW = [0, 39]  |  MEDIUM = [40, 69]  |  HIGH = [70, 100]
```

---

## Quick Start

**Prerequisites:** Python 3.11+, webcam, ~4 GB disk space for model files.

```bash
git clone https://github.com/Aryansh909/Examshield-AI.git
cd Examshield-AI

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt
cp .env.example .env            # set ADMIN_PASSWORD and SECRET_KEY

bash scripts/download_models.sh
python app.py
```

Or with make:
```bash
make setup && make run
```

| Interface | URL |
|---|---|
| Candidate Monitor | http://localhost:5000/ |
| Admin Dashboard | http://localhost:5000/admin |

Default admin credentials: `admin` / `admin` — change in `.env` before any production use.

---

## Detection Details

### Suspicion Score Weights

| Violation | Weight | Notes |
|---|---|---|
| Identity Mismatch | 25 | Highest — definitive cheating signal |
| Phone Detected | 20 | YOLO confidence > 0.5, 1.5s debounce |
| Multiple Faces | 18 | 2+ faces in frame, 1.5s debounce |
| No Face | 10 | Candidate left frame, 3s debounce |
| Head Turned | 8 | Any direction, 1.5s debounce, 25s cooldown |
| Gaze Off-Screen | 8 | Blendshape > 0.50, 2s debounce, 30s cooldown |
| Mouth Open | 5 | jawOpen > 0.35, 2s debounce, 25s cooldown |
| Hand Near Face | 4 | Wrist landmark inside face bounding box |

### Head Pose

```
dx = nose.x - face_midpoint.x
dy = nose.y - face_midpoint.y

if |dx| > |dy|:  left / right   (yaw threshold: 0.04)
else:            up / down      (pitch threshold: 0.05)
```

### Gaze Estimation

```
look_left  = eyeLookOutLeft  + eyeLookInRight
look_right = eyeLookOutRight + eyeLookInLeft
look_up    = eyeLookUpLeft   + eyeLookUpRight
look_down  = eyeLookDownLeft + eyeLookDownRight

if max(all) < 0.50  ->  "on-screen"
else                ->  direction of maximum blendshape
```

---

## API

Full reference: [`docs/api.md`](docs/api.md)

| Method | Endpoint | Auth | Description |
|---|---|---|---|
| GET | `/` | — | Candidate monitoring dashboard |
| GET | `/video` | — | MJPEG annotated live stream |
| GET | `/stats` | — | Real-time JSON telemetry |
| GET | `/events` | — | Last 100 violation events |
| GET | `/session_summary` | — | Current session statistics |
| POST | `/reset` | — | Reset session and re-enroll identity |
| GET | `/export` | Admin | CSV session report |
| GET | `/history` | Admin | All past session summaries |
| POST | `/simulate` | Admin | Inject test violations |
| WS | `stats_update` | — | Real-time telemetry push |

Sample `/stats` response:
```json
{
  "suspicion_score": 45.2,
  "risk_level": "MEDIUM",
  "face_count": 1,
  "gaze_direction": "on-screen",
  "head_direction": "forward",
  "phone_detected": false,
  "gaze_violations": 3,
  "head_violation_count": 2,
  "enrollment_complete": true,
  "fps": 28.4
}
```

---

## Training Custom CNN Models

ExamShield includes a MobileNetV2 training pipeline (`train_models.py`) with ImageNet pre-training and fine-tuned classification heads. MediaPipe blendshapes are used by default — custom models activate only when `.pth` files are present in `models/ml/`.

**Dataset format (ImageFolder):**
```
datasets/
    head_pose/
        forward/    img001.jpg ...
        left/       ...
        right/      ...
    gaze/
        on-screen/  ...
        off-screen/ ...
    mouth/
        open/       ...
        closed/     ...
```

**Training:**
```bash
python train_models.py --model head  --data ./datasets/head_pose  --epochs 15
python train_models.py --model gaze  --data ./datasets/gaze       --epochs 15
python train_models.py --model mouth --data ./datasets/mouth      --epochs 15
```

Features: frozen MobileNetV2 backbone, data augmentation (flip, color jitter, rotation), StepLR scheduler, best checkpoint saved per epoch.

---

## Project Structure

```
examshield-ai/
├── app.py                  # Flask REST API + WebSocket server
├── camera.py               # Core detection pipeline (Camera class)
├── config.py               # Centralised config with dotenv support
├── db.py                   # SQLite persistence layer
├── train_models.py         # MobileNetV2 CNN training pipeline
├── requirements.txt
├── Makefile
├── .env.example
├── models/
│   ├── yolo/               # yolov8n.pt (auto-downloaded)
│   ├── mediapipe/          # face_landmarker.task, hand_landmarker.task
│   └── ml/                 # head_model.pth, gaze_model.pth, mouth_model.pth
├── frontend/
│   ├── index.html          # Candidate monitoring dashboard
│   ├── admin.html          # Admin dashboard
│   └── login.html
├── scripts/
│   ├── setup.py
│   └── download_models.sh
├── docs/
│   ├── architecture.md
│   ├── api.md
│   └── scoring.md
├── tests/
│   ├── test_config.py
│   ├── test_db.py
│   └── test_api.py
├── datasets/               # Training data (gitignored)
└── snapshots/              # Evidence frames (gitignored)
```

---

## Configuration

All settings load from environment variables or a `.env` file.

| Variable | Default | Description |
|---|---|---|
| `FLASK_HOST` | `0.0.0.0` | Server bind address |
| `FLASK_PORT` | `5000` | Server port |
| `FLASK_DEBUG` | `false` | Debug mode |
| `ADMIN_PASSWORD` | `admin` | Admin dashboard password |
| `SECRET_KEY` | dev key | Flask session signing key |
| `CAMERA_INDEX` | `0` | Webcam device index |

---

## Tests

```bash
make test
# or
pytest tests/ -v
```

28 tests — no webcam required. Covers config validation, SQLite CRUD, and all Flask route smoke tests including auth enforcement.

---

## Dependencies

**Required:** Python 3.11+, OpenCV-compatible webcam, ~500 MB RAM

**Optional (graceful fallback if absent):**
- GPU (CUDA) — YOLO falls back to CPU
- `face_recognition` / `dlib` — identity verification disabled if absent
- Custom CNN `.pth` files — MediaPipe blendshapes used as fallback
- `flask-socketio` — falls back to HTTP polling

---

## Acknowledgements

| Library | Purpose |
|---|---|
| [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) | Phone and face detection |
| [MediaPipe](https://mediapipe.dev/) | Face mesh, hand landmarks, blendshapes |
| [face_recognition](https://github.com/ageitgey/face_recognition) | Identity verification |
| [PyTorch](https://pytorch.org/) | MobileNetV2 CNN fine-tuning |
| [Flask](https://flask.palletsprojects.com/) | REST API and MJPEG streaming |
| [Flask-SocketIO](https://flask-socketio.readthedocs.io/) | WebSocket broadcast |
| [Chart.js](https://www.chartjs.org/) | Live score visualisation |

---

## License

MIT — see [LICENSE](LICENSE).

---

Built by [Aryan Sharma](https://github.com/Aryansh909) · [API Docs](docs/api.md) · [Architecture](docs/architecture.md) · [Report an Issue](https://github.com/Aryansh909/Examshield-AI/issues)
