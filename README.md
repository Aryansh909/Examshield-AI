# ExamShield AI — Real-Time Multimodal Computer Vision Proctoring System

[![Python 3.11+](https://img.shields.io/badge/Python-3.11+-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![Flask 3.0](https://img.shields.io/badge/Flask-3.0-000000?style=flat-square&logo=flask&logoColor=white)](https://flask.palletsprojects.com/)
[![YOLOv8](https://img.shields.io/badge/YOLO-v8-FF6B35?style=flat-square)](https://github.com/ultralytics/ultralytics)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10-0097A7?style=flat-square)](https://mediapipe.dev/)
[![PyTorch 2.2](https://img.shields.io/badge/PyTorch-2.2-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![License MIT](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)

---

## Technical Overview

ExamShield AI is an edge-capable, multi-threaded computer vision proctoring engine engineered to analyze candidate behavior in online examinations. The system integrates **YOLOv8 object detection**, **MediaPipe 468-point 3D facial geometry**, **dlib face recognition**, and a **probabilistic signal-fusion guard engine** to evaluate threat scores in real time without cloud dependencies.

---

## Core Capabilities

- **Object & Person Detection**: Asynchronous YOLOv8 inference detecting prohibited devices (mobile phones) and unauthorized individuals at 2 Hz.
- **3D Head Pose & Gaze Tracking**: MediaPipe 468-point facial mesh and blendshapes evaluating head rotation (pitch/yaw/roll) and eye gaze deflection with 9-frame majority-vote smoothing.
- **Continuous Face Verification**: Asynchronous dlib 128-D facial feature embedding comparison evaluating candidate identity against baseline enrollment every 2 seconds.
- **Hand Proximity Detection**: MediaPipe HandLandmarker evaluating wrist position relative to face bounding geometry.
- **Defensive Signal Guards**: Time-window debouncing, clear-state hysteresis, and per-event cooldowns to eliminate false positives from transient facial movements.
- **Real-Time Telemetry & REST API**: Flask REST API with WebSocket push broadcasting score timelines and violation events to administrative dashboards.

---

## System Architecture & Threading Model

```
Webcam (OpenCV Frame Buffer)
          |
          v
+-----------------------------------------------------------------------+
|  Camera Pipeline Orchestrator  (camera.py)                            |
|                                                                       |
|  Thread 1: Main Inference Loop  (~30 FPS)                             |
|    +-- MediaPipe FaceLandmarker  (468-pt Mesh & Blendshapes)          |
|    +-- MediaPipe HandLandmarker  (Hand Proximity Detection)           |
|    +-- Signal Guard System       (Debounce & Hysteresis Filtering)    |
|    +-- Suspicion Scoring Engine  (Weighted Additive & Time Decay)     |
|                                                                       |
|  Thread 2: YOLO Detection Thread (2 Hz Asynchronous)                  |
|    +-- YOLOv8 Inference on Raw Frame -> phone_detected, face_count    |
|                                                                       |
|  Thread 3: Identity Verification Thread (0.5 Hz Asynchronous)         |
|    +-- dlib 128-D Facial Embedding -> identity_mismatch               |
+-----------------------------------------------------------------------+
          |
          v Shared Thread-Safe Memory Lock
+-----------------------------------------------------------------------+
|  Flask REST API & WebSocket Layer  (app.py)                           |
|    +-- GET  /video            MJPEG Video Feed                        |
|    +-- GET  /stats            JSON Telemetry Payload                  |
|    +-- GET  /export           CSV Session Report Download             |
|    +-- WS   stats_update      Real-Time Telemetry Broadcast           |
+-----------------------------------------------------------------------+
          |
          v
+------------------------------------+----------------------------------+
|  Browser Dashboards (frontend/)    |  SQLite Database Layer (db.py)   |
|  - Candidate Monitoring HUD        |  - events table                  |
|  - Administrative Multi-Node View  |  - sessions table                |
+------------------------------------+----------------------------------+
```

---

## Signal Guard System & Mathematical Formulation

To prevent transient sensor noise from triggering invalid violations, every detection signal passes through three defensive layers:

```
Raw Signal  ->  [Debounce Filter]  ->  [Active Flag Set]  ->  [Event Cooldown]  ->  Logged
                                              |
                                  [Hysteresis Delay]  ->  Flag Reset
```

### 1. Debounce (Time-Window Persistence)
A violation condition must persist continuously for duration $T_{debounce}$ before registering:
$$\text{State}_{\text{active}} = \begin{cases} \text{True} & \text{if } t_{\text{detected}} \ge T_{\text{debounce}} \\ \text{False} & \text{otherwise} \end{cases}$$

### 2. Hysteresis (Clear-State Recovery)
Once active, a signal must remain continuously clear for duration $T_{clear}$ before resetting to normal:
$$T_{\text{clear, gaze}} = 3.0\text{s}, \quad T_{\text{clear, head}} = 2.0\text{s}$$

### 3. Suspicion Scoring & Decay Formula
Suspicion score $S(t) \in [0, 100]$ is updated per frame:
$$S(t) = \min\left(100, S(t-1) + \sum_{i \in \text{Active}} W_i\right) \quad \text{when violations are present}$$
$$S(t) = \max\left(0, S(t-1) - \lambda_{\text{decay}}\right) \quad \text{when clear } (\lambda_{\text{decay}} = 0.3/\text{frame})$$

---

## Technical Weights & Thresholds

| Violation Type | Weight ($W_i$) | Debounce ($T_{\text{debounce}}$) | Cooldown Window |
| :--- | :--- | :--- | :--- |
| **Identity Mismatch** | **25** | 3 consecutive checks | 60s |
| **Phone Detected** | **20** | 1.5s | 45s |
| **Multiple Faces** | **18** | 1.5s | 45s |
| **No Face** | **10** | 3.0s | 30s |
| **Head Turned** | **8** | 1.5s | 25s |
| **Gaze Off-Screen** | **8** | 2.0s | 30s |
| **Mouth Open / Speaking** | **5** | 2.0s | 25s |
| **Hand Near Face** | **4** | 1.0s | 20s |

---

## REST API Reference

| Method | Endpoint | Auth | Response / Description |
| :--- | :--- | :--- | :--- |
| `GET` | `/` | — | Candidate monitoring web dashboard |
| `GET` | `/video` | — | MJPEG annotated video stream |
| `GET` | `/stats` | — | Real-time JSON telemetry (Score, Risk Level, FPS, Signals) |
| `GET` | `/events` | — | Array of recent violation log entries |
| `GET` | `/session_summary` | — | Aggregate statistics for active session |
| `POST` | `/reset` | — | Reset session state and re-enroll identity baseline |
| `GET` | `/export` | Admin | Session report download (`text/csv`) |
| `GET` | `/history` | Admin | Query all past historical sessions from SQLite |
| `WS` | `stats_update` | — | WebSocket real-time telemetry push (1 Hz) |

---

## Quick Start & Installation

### 1. Requirements
- Python 3.11+
- Linux / macOS / Windows
- OpenCV-compatible webcam

### 2. Environment Setup
```bash
git clone https://github.com/Aryansh909/Examshield-AI.git
cd Examshield-AI

python3 -m venv venv
source venv/bin/activate

pip install -r requirements.txt
cp .env.example .env
```

### 3. Model Assets Download
```bash
bash scripts/download_models.sh
```

### 4. Execution
```bash
python app.py
# or using ergonomics Makefile:
make run
```
Access dashboards:
- Candidate Monitor: `http://localhost:5000/`
- Administrative View: `http://localhost:5000/admin`

---

## Automated Test Verification

The project includes an automated test suite (`pytest`) mocking hardware video capture to verify routes, schema initialization, and configuration constraints:

```bash
make test
# or
pytest tests/ -v
```

---

## License

MIT License — see [LICENSE](LICENSE) for details.
