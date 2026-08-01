# Custom CNN Checkpoints

This directory stores fine-tuned MobileNetV2 `.pth` checkpoints.

## Files (Optional)

| File | Purpose |
|------|--------|
| `head_model.pth` | Head pose classifier (forward/left/right/up/down) |
| `gaze_model.pth` | Gaze direction classifier (on-screen/off-screen) |
| `mouth_model.pth` | Mouth state classifier (open/closed/speaking) |

## Training

If these files are absent, ExamShield uses MediaPipe blendshapes as fallback (recommended).

```bash
python train_models.py --model head  --data ./datasets/head_pose  --epochs 15
python train_models.py --model gaze  --data ./datasets/gaze       --epochs 15
python train_models.py --model mouth --data ./datasets/mouth      --epochs 15
```

## Dataset Format

ImageFolder format — one subdirectory per class:
```
datasets/head_pose/
    forward/   img001.jpg ...
    left/      img001.jpg ...
    right/     img001.jpg ...
    up/        img001.jpg ...
    down/      img001.jpg ...
```
