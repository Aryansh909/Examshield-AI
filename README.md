<div align="center">

![header](https://capsule-render.vercel.app/api?type=rect&color=0:0D1117,100:0E2A4A&height=90&text=%F0%9F%9B%A1%EF%B8%8F%20ExamShield%20AI&fontSize=32&fontColor=E6EDF3&fontAlignY=55&desc=Real-time%20Computer%20Vision%20Proctoring%20System&descSize=14&descAlignY=78&descColor=7EA8BE)

[![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org)
[![Flask](https://img.shields.io/badge/Flask-000000?style=flat-square&logo=flask&logoColor=white)](https://flask.palletsprojects.com)
[![License](https://img.shields.io/badge/License-MIT-22863A?style=flat-square)](LICENSE)

</div>

---

## Overview

ExamShield AI is a production-grade remote proctoring system built on three parallel computer vision models — YOLOv8 for object detection, MediaPipe for 3D facial geometry analysis, and dlib for continuous face verification. A probabilistic suspicion scoring engine aggregates signals from all three and broadcasts live telemetry to an admin dashboard over WebSockets.

Built during an ML internship at KvonTech Consultancy Services (Feb–Jun 2026).

## Features

- **Multi-model CV pipeline** — YOLOv8, MediaPipe 468-point Face Landmarker, and dlib face recognition in independent inference threads
- **Signal guards** — Debounce + hysteresis filtering on every detection signal to eliminate transient false positives  
- **Live dashboard** — Real-time MJPEG annotated video stream with Chart.js score visualisation and 4 UI themes
- **Admin panel** — Multi-student monitoring interface with session history and evidence snapshots
- **REST API** — 18+ Flask endpoints for telemetry, event logs, snapshots, and CSV export
- **WebSocket broadcasting** — Flask-SocketIO for sub-second score updates
- **Test suite** — 28 pytest tests covering all detection modes, runnable without a physical webcam

## Tech Stack

[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-00BFFF?style=flat-square&logoColor=white)]()
[![MediaPipe](https://img.shields.io/badge/MediaPipe-0097A7?style=flat-square&logoColor=white)]()
[![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?style=flat-square&logo=opencv&logoColor=white)](https://opencv.org)
[![dlib](https://img.shields.io/badge/dlib-008000?style=flat-square&logoColor=white)]()
[![Flask](https://img.shields.io/badge/Flask-000000?style=flat-square&logo=flask&logoColor=white)](https://flask.palletsprojects.com)
[![Flask-SocketIO](https://img.shields.io/badge/Flask--SocketIO-010101?style=flat-square&logoColor=white)]()
[![SQLite](https://img.shields.io/badge/SQLite-003B57?style=flat-square&logo=sqlite&logoColor=white)](https://sqlite.org)

## Getting Started

### Prerequisites

- Python 3.9+
- Webcam (or virtual camera for testing)

### Installation

```bash
git clone https://github.com/Aryansh909/Examshield-AI.git
cd Examshield-AI
cp .env.example .env          # configure credentials
pip install -r requirements.txt
bash scripts/download_models.sh
python app.py
```

Monitoring dashboard → `http://localhost:5000`  
Admin panel → `http://localhost:5000/admin`

### Environment Variables

| Variable | Default | Description |
|:--|:--|:--|
| `ADMIN_PASSWORD` | `admin` | Admin dashboard password |
| `SECRET_KEY` | — | Flask session secret (change in production) |
| `FLASK_PORT` | `5000` | Server port |
| `CAMERA_INDEX` | `0` | Webcam device index |

## API Reference

| Method | Endpoint | Description |
|:--|:--|:--|
| `GET` | `/` | Monitoring dashboard |
| `GET` | `/admin` | Admin multi-student view |
| `GET` | `/video` | MJPEG annotated stream |
| `GET` | `/stats` | Real-time JSON telemetry |
| `GET` | `/report` | Session event log |
| `GET` | `/timeline` | Chronological event data |
| `GET` | `/export` | CSV session report |
| `GET` | `/history` | All past sessions |
| `GET` | `/snapshots` | Evidence snapshot list |
| `POST` | `/reset` | Reset session & re-enroll identity |

## Project Structure

```
examshield-ai/
├── app.py              # Flask app, routes, WebSocket events
├── camera.py           # CV pipeline, inference threads, suspicion engine
├── config.py           # Environment configuration
├── db.py               # SQLite persistence layer
├── train_models.py     # Custom CNN training pipeline
├── frontend/           # HTML dashboards (monitoring, admin, login)
├── models/             # YOLO, MediaPipe, and CNN model weights
├── scripts/            # Setup and model download scripts
├── tests/              # pytest suite (28 tests)
└── snapshots/          # Evidence snapshot storage
```

## License

[MIT](LICENSE)
