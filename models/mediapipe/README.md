# MediaPipe Task Files

This directory stores MediaPipe `.task` model files.

## Required Files

| File | Size | Purpose |
|------|------|--------|
| `face_landmarker.task` | ~29 MB | 468-point face mesh + blendshapes |
| `hand_landmarker.task` | ~17 MB | Hand landmark detection |

## Download

```bash
bash scripts/download_models.sh
```

Or manually:
```bash
curl -L -o models/mediapipe/face_landmarker.task \
  https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task

curl -L -o models/mediapipe/hand_landmarker.task \
  https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task
```
