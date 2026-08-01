"""
ExamShield AI — Project Setup Script
Run once before starting the server to create all required directories
and print model download instructions.

Usage:
    python scripts/setup.py
"""

import os
import sys

# Allow running from repo root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DIRS = [
    os.path.join(BASE_DIR, "models", "yolo"),
    os.path.join(BASE_DIR, "models", "mediapipe"),
    os.path.join(BASE_DIR, "models", "ml"),
    os.path.join(BASE_DIR, "snapshots"),
    os.path.join(BASE_DIR, "datasets", "head_pose"),
    os.path.join(BASE_DIR, "datasets", "gaze"),
    os.path.join(BASE_DIR, "datasets", "mouth"),
]

print("=" * 60)
print("  ExamShield AI — Directory Setup")
print("=" * 60)

for d in DIRS:
    os.makedirs(d, exist_ok=True)
    print(f"  [OK] {os.path.relpath(d, BASE_DIR)}")

print()
print("=" * 60)
print("  MODEL FILES REQUIRED")
print("=" * 60)
print()
print("1. YOLOv8n (phone detection):")
print("   Installed automatically via pip install ultralytics")
print("   Place at: models/yolo/yolov8n.pt")
print()
print("2. YOLOv8n-face (optional, face counting):")
print("   https://github.com/derronqi/yolov8-face")
print("   Place at: models/yolo/yolov8n-face.pt")
print()
print("3. MediaPipe Face Landmarker:")
print("   Run: bash scripts/download_models.sh")
print("   Place at: models/mediapipe/face_landmarker.task")
print()
print("4. MediaPipe Hand Landmarker:")
print("   Run: bash scripts/download_models.sh")
print("   Place at: models/mediapipe/hand_landmarker.task")
print()
print("5. Custom CNN models (optional — MediaPipe used as fallback):")
print("   python train_models.py --model head  --data ./datasets/head_pose")
print("   python train_models.py --model gaze  --data ./datasets/gaze")
print("   python train_models.py --model mouth --data ./datasets/mouth")
print()
print("=" * 60)
print("  QUICK START")
print("=" * 60)
print()
print("  cp .env.example .env          # configure credentials")
print("  pip install -r requirements.txt")
print("  bash scripts/download_models.sh")
print("  python app.py")
print()
print("  Monitor : http://localhost:5000/")
print("  Admin   : http://localhost:5000/admin")
print()

