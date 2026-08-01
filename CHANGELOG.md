# Changelog

All notable changes to ExamShield AI are documented here.

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

---

## [1.0.0] — 2026-08-01

### Added
- Real-time YOLO-based phone and multi-face detection (YOLOv8n)
- MediaPipe 468-point Face Landmarker for head pose and gaze estimation
- MediaPipe Hand Landmarker for hand-near-face proximity detection
- Custom MobileNetV2 CNN training pipeline (`train_models.py`)
- Probabilistic suspicion scoring engine with configurable weights and decay
- Identity enrollment and continuous face verification (face_recognition)
- Flask REST API with 18 endpoints
- MJPEG live video stream with CV annotations (`/video`)
- Real-time monitoring dashboard with Chart.js score visualisation
- Multi-student admin dashboard (`/admin`)
- 4 dynamic UI themes: Cyberpunk Neon, Emerald Matrix, Solar Cyber, Studio Light
- Flask-SocketIO WebSocket broadcast for real-time telemetry
- SQLite persistence layer for session history and event logs
- Automated evidence snapshot capture at HIGH risk threshold
- CSV export endpoint for session reports
- Admin authentication via session cookies
- Simulation override API for testing without physical violations

### Security
- Moved ADMIN_PASSWORD and SECRET_KEY to environment variables
- Added `.env.example` template

### Detection Logic Improvements
- Added majority-vote smoothing (9-frame buffer) for head/gaze/mouth signals
- Raised gaze detection threshold to 0.50 to reduce false positives
- Added per-event-type cooldowns (30s gaze, 25s head, 25s mouth)
- Added hysteresis: 3s sustained on-screen time required before gaze resets
- Increased debounce durations: gaze 0.8s→2.0s, head 0.8s→1.5s
- Snapshot cooldown extended from 30s to 120s
- YOLO detection runs in a dedicated daemon thread (no main loop blocking)
- Camera auto-retry across indices 0–2 if primary is locked
