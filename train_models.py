"""
EXAMSHIELD AI — Model Training Pipeline
Trains three MobileNetV2-based CNN classifiers:
  1. Head Pose  (forward / left / right / up / down)
  2. Gaze       (on-screen / off-screen)
  3. Mouth      (open / closed / speaking)

Usage:
  python train_models.py --model head   --data ./datasets/head_pose
  python train_models.py --model gaze   --data ./datasets/gaze
  python train_models.py --model mouth  --data ./datasets/mouth

Dataset folder structure (ImageFolder format):
  <data>/
    <class_name>/
      img001.jpg
      img002.jpg
      ...
"""

import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
from sklearn.preprocessing import LabelEncoder
from PIL import Image
import pandas as pd
import numpy as np

from config import PATHS, ML_CONFIG


# ── Dataset ────────────────────────────────────────────────────────────────────
class FaceDataset(Dataset):
    """
    Supports two modes:
      - ImageFolder mode: data_root with subdirectories per class
      - CSV mode: CSV file with columns [image_path, label]
    """

    TRAIN_TRANSFORM = transforms.Compose([
        transforms.Resize((ML_CONFIG["input_size"], ML_CONFIG["input_size"])),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])

    def __init__(self, data_root: str, encoder: LabelEncoder = None, csv_file: str = None):
        self.samples = []
        self.labels  = []

        if csv_file and os.path.exists(csv_file):
            # CSV mode
            df = pd.read_csv(csv_file)
            self.samples = df["image_path"].tolist()
            raw_labels   = df["label"].tolist()
        else:
            # ImageFolder mode
            raw_labels = []
            for cls_name in sorted(os.listdir(data_root)):
                cls_dir = os.path.join(data_root, cls_name)
                if not os.path.isdir(cls_dir):
                    continue
                for fname in os.listdir(cls_dir):
                    if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                        self.samples.append(os.path.join(cls_dir, fname))
                        raw_labels.append(cls_name)

        # Fit or reuse encoder
        if encoder is None:
            self.encoder = LabelEncoder()
            self.labels  = self.encoder.fit_transform(raw_labels)
        else:
            self.encoder = encoder
            self.labels  = encoder.transform(raw_labels)

        self.transform = self.TRAIN_TRANSFORM
        print(f"[Dataset] {len(self.samples)} images | classes: {list(self.encoder.classes_)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img = Image.open(self.samples[idx]).convert("RGB")
        return self.transform(img), int(self.labels[idx])


# ── Build model ────────────────────────────────────────────────────────────────
def build_model(num_classes: int) -> nn.Module:
    model = mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1)
    # Freeze feature extractor (fine-tune only classifier)
    for param in model.features.parameters():
        param.requires_grad = False
    model.classifier[1] = nn.Linear(model.last_channel, num_classes)
    return model


# ── Training loop ──────────────────────────────────────────────────────────────
def train(model_type: str, data_root: str, epochs: int = 10,
          batch_size: int = 32, lr: float = 1e-3, csv_file: str = None):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Train] model={model_type}  device={device}  epochs={epochs}")

    dataset = FaceDataset(data_root, csv_file=csv_file)
    loader  = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                         num_workers=2, pin_memory=True)

    num_classes = len(dataset.encoder.classes_)
    model       = build_model(num_classes).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.classifier.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)

    best_loss = float("inf")
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        correct = 0
        total   = 0

        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss    = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            preds    = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total   += labels.size(0)

        epoch_loss = running_loss / total
        epoch_acc  = 100.0 * correct / total
        print(f"  Epoch {epoch:02d}/{epochs}  loss={epoch_loss:.4f}  acc={epoch_acc:.1f}%")
        scheduler.step()

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            _save_checkpoint(model, dataset.encoder, model_type)

    print(f"[Train] Done. Best loss: {best_loss:.4f}")


def _save_checkpoint(model, encoder, model_type: str):
    key = f"{model_type}_model"
    out_path = PATHS.get(key, f"models/ml/{model_type}_model.pth")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    torch.save({
        "state_dict": model.state_dict(),
        "encoder":    encoder,
    }, out_path)
    print(f"  [Save] Checkpoint → {out_path}")


# ── CLI ────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EXAMSHIELD AI — CNN Trainer")
    parser.add_argument("--model",  required=True, choices=["head", "gaze", "mouth"],
                        help="Which classifier to train")
    parser.add_argument("--data",   required=True,
                        help="Path to dataset root (ImageFolder) or CSV")
    parser.add_argument("--csv",    default=None,
                        help="Optional CSV file with image_path,label columns")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch",  type=int, default=32)
    parser.add_argument("--lr",     type=float, default=1e-3)
    args = parser.parse_args()

    train(
        model_type=args.model,
        data_root=args.data,
        epochs=args.epochs,
        batch_size=args.batch,
        lr=args.lr,
        csv_file=args.csv,
    )
