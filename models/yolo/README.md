# YOLO Model Files

This directory stores YOLOv8 model weights. These files are excluded from version
control due to their size (LFS not configured). Download them separately.

## Required Files

| File | Size | Source | Purpose |
|------|------|--------|--------|
| `yolov8n.pt` | ~6 MB | Auto-downloaded by `ultralytics` | Phone detection (COCO classes) |
| `yolov8n-face.pt` | ~6 MB | [yolov8-face](https://github.com/derronqi/yolov8-face) | Face counting (optional) |

## Download

```bash
# yolov8n.pt — downloads automatically on first run
pip install ultralytics
python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"
mv yolov8n.pt models/yolo/

# yolov8n-face.pt — manual download
# Download from: https://github.com/derronqi/yolov8-face/releases
# Place at: models/yolo/yolov8n-face.pt
```

> **Note**: If `yolov8n-face.pt` is absent, ExamShield falls back to MediaPipe for face counting.
