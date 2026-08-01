#!/usr/bin/env bash
# ExamShield AI — MediaPipe Model Downloader
# Downloads the required MediaPipe .task files into models/mediapipe/

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
MEDIAPIPE_DIR="$PROJECT_ROOT/models/mediapipe"

mkdir -p "$MEDIAPIPE_DIR"

echo "[ExamShield] Downloading MediaPipe models..."

# Face Landmarker (float16, ~29 MB)
FACE_URL="https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
FACE_OUT="$MEDIAPIPE_DIR/face_landmarker.task"

if [ -f "$FACE_OUT" ]; then
  echo "  [SKIP] face_landmarker.task already exists"
else
  echo "  [DL]   Downloading face_landmarker.task..."
  curl -L -o "$FACE_OUT" "$FACE_URL"
  echo "  [OK]   face_landmarker.task"
fi

# Hand Landmarker (float16, ~17 MB)
HAND_URL="https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
HAND_OUT="$MEDIAPIPE_DIR/hand_landmarker.task"

if [ -f "$HAND_OUT" ]; then
  echo "  [SKIP] hand_landmarker.task already exists"
else
  echo "  [DL]   Downloading hand_landmarker.task..."
  curl -L -o "$HAND_OUT" "$HAND_URL"
  echo "  [OK]   hand_landmarker.task"
fi

echo "[ExamShield] MediaPipe models ready."
echo "  Next: python app.py"
