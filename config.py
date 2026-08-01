"""
EXAMSHIELD AI — Configuration
Centralised path registry and ML configuration parameters.
All sensitive values are loaded from environment variables (see .env.example).
"""

import os

# Load .env if present (development convenience)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass   # python-dotenv optional; fall back to system env vars

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

PATHS = {
    # YOLO Models
    "yolo_phone":   os.path.join(BASE_DIR, "models", "yolo", "yolov8n.pt"),
    "yolo_face":    os.path.join(BASE_DIR, "models", "yolo", "yolov8n-face.pt"),

    # MediaPipe Task Files
    "mp_face":      os.path.join(BASE_DIR, "models", "mediapipe", "face_landmarker.task"),
    "mp_hand":      os.path.join(BASE_DIR, "models", "mediapipe", "hand_landmarker.task"),

    # Custom CNN Checkpoints
    "head_model":   os.path.join(BASE_DIR, "models", "ml", "head_model.pth"),
    "gaze_model":   os.path.join(BASE_DIR, "models", "ml", "gaze_model.pth"),
    "mouth_model":  os.path.join(BASE_DIR, "models", "ml", "mouth_model.pth"),

    # Frontend Static Files
    "frontend":     os.path.join(BASE_DIR, "frontend"),

    # Evidence Snapshots Output Directory
    "snapshots":    os.path.join(BASE_DIR, "snapshots"),

    # SQLite Database
    "database":     os.path.join(BASE_DIR, "examshield.db"),
}

# ML inference settings (shared across all CNN models)
ML_CONFIG = {
    "input_size":   224,       # MobileNetV2 input dimensions (224x224)
    "interval":     0.25,      # CNN inference interval in seconds (4x per second)
    "warmup_frames": 30,       # Frames to skip before enrollment
}

# Suspicion scoring weights per violation type
SCORE_WEIGHTS = {
    "phone_detected":       20,
    "multiple_faces":       18,
    "no_face":              10,
    "head_turned":           8,
    "gaze_off_screen":       8,
    "mouth_open":            5,
    "identity_mismatch":    25,
    "hand_near_face":        4,
}

# Score decay per frame during violation-free period
SCORE_DECAY = 0.3

# Risk level thresholds
RISK_THRESHOLDS = {
    "LOW":    (0,  39),
    "MEDIUM": (40, 69),
    "HIGH":   (70, 100),
}

# Snapshot capture: take evidence snapshot above this score
SNAPSHOT_THRESHOLD = 70

# Flask server settings
FLASK_HOST  = os.getenv("FLASK_HOST",  "0.0.0.0")
FLASK_PORT  = int(os.getenv("FLASK_PORT", "5000"))
FLASK_DEBUG = os.getenv("FLASK_DEBUG", "false").lower() == "true"

# Admin authentication (override in .env for production)
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "admin")

# Flask session signing key (MUST be overridden in production)
SECRET_KEY = os.getenv("SECRET_KEY", "examshield-dev-key-change-in-production")

# Head violation: consecutive intervals before registering a violation
HEAD_VIOLATION_CONSECUTIVE = 3

# Camera index (0 = default webcam, override via env for multi-camera setups)
CAMERA_INDEX = int(os.getenv("CAMERA_INDEX", "0"))

