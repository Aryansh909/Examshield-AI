"""
EXAMSHIELD AI — Directory Setup Script
Run this once before starting the server to create all required directories
and print instructions for downloading model files.

Usage:
    python setup_dirs.py
"""

import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

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
print("  EXAMSHIELD AI — Directory Setup")
print("=" * 60)

for d in DIRS:
    os.makedirs(d, exist_ok=True)
    print(f"  [OK] {os.path.relpath(d, BASE_DIR)}")

print()
print("=" * 60)
print("  MODEL FILES REQUIRED")
print("=" * 60)
print()
print("1. YOLOv8n (phone detection — COCO model):")
print("   pip install ultralytics")
print("   # yolov8n.pt downloads automatically on first run")
print("   # Place at: models/yolo/yolov8n.pt")
print()
print("2. YOLOv8n-face (face detection):")
print("   Download: https://github.com/derronqi/yolov8-face")
print("   Place at: models/yolo/yolov8n-face.pt")
print()
print("3. MediaPipe Face Landmarker:")
print("   Download: https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task")
print("   Place at: models/mediapipe/face_landmarker.task")
print()
print("4. MediaPipe Hand Landmarker:")
print("   Download: https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task")
print("   Place at: models/mediapipe/hand_landmarker.task")
print()
print("5. Custom CNN models (optional — train with train_models.py):")
print("   python train_models.py --model head  --data ./datasets/head_pose  --epochs 10")
print("   python train_models.py --model gaze  --data ./datasets/gaze       --epochs 10")
print("   python train_models.py --model mouth --data ./datasets/mouth      --epochs 10")
print()
print("=" * 60)
print("  QUICK START (models auto-download where possible)")
print("=" * 60)
print()
print("  pip install -r requirements.txt")
print("  python app.py")
print()
print("  Candidate Monitor : http://localhost:5000/")
print("  Admin Dashboard   : http://localhost:5000/admin")
print()
