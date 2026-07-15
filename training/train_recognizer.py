"""
Train the digit-sequence recognizer.

Two paths, same data format and same downstream ONNX export/interface:

1. CRNN + CTC (this script's default) - simpler, well-understood, a solid
   baseline you can train with a small dataset and get to production fast.
2. PARSeq - stronger on hard cases (partial occlusion, unusual fonts) but
   more involved to train from scratch. Once your CRNN baseline is working
   end-to-end (data pipeline, export, Android integration all validated),
   swap in PARSeq using the reference implementation at
   https://github.com/baudm/parseq and point export_onnx.py at its checkpoint
   format instead - the rest of this SDK's interface does not change.

Expects a directory of cropped odometer images plus a ground_truth.json
mapping filename -> digit string, exactly like
dataset/generate_synthetic_data.py produces.

Usage:
    python training/train_recognizer.py \
        --data dataset/real_cropped \
        --epochs 50 \
        --out training/runs/recognizer
"""
import argparse
import json
import os

import numpy as np


CHARSET = "0123456789"
BLANK_IDX = len(CHARSET)  # CTC blank token


class OdometerCropDataset:
    """Minimal dataset wrapper. Swap for a real torch.utils.data.Dataset
    once you have real (non-synthetic) data - kept framework-light here so
    this file has zero heavyweight import requirements until you actually
    train."""

    def __init__(self, data_dir: str):
        self.data_dir = data_dir
        gt_path = os.path.join(data_dir, "ground_truth.json")
        with open(gt_path) as f:
            self.ground_truth = json.load(f)
        self.filenames = list(self.ground_truth.keys())

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        fname = self.filenames[idx]
        label = self.ground_truth[fname]
        img_path = os.path.join(self.data_dir, "images", fname)
        return img_path, label


def build_model(num_classes: int = len(CHARSET) + 1):
    """CRNN: CNN feature extractor -> BiLSTM -> per-timestep classifier,
    trained with CTC loss. Requires torch."""
    import torch.nn as nn

    class CRNN(nn.Module):
        def __init__(self, num_classes):
            super().__init__()
            self.cnn = nn.Sequential(
                nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2),
                nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2),
                nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
                nn.Conv2d(128, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d((2, 1)),
            )
            self.rnn = nn.LSTM(128 * 8, 256, bidirectional=True, num_layers=2, batch_first=True)
            self.fc = nn.Linear(512, num_classes)

        def forward(self, x):
            # x: (batch, 1, 64, W)
            feats = self.cnn(x)                      # (batch, C, H', W')
            b, c, h, w = feats.shape
            feats = feats.permute(0, 3, 1, 2).reshape(b, w, c * h)  # (batch, W', C*H')
            out, _ = self.rnn(feats)
            out = self.fc(out)                        # (batch, W', num_classes)
            return out

    return CRNN(num_classes)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Dir with images/ + ground_truth.json")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--out", default="training/runs/recognizer")
    args = parser.parse_args()

    try:
        import torch
    except ImportError:
        raise SystemExit("torch not installed. Run: pip install torch")

    dataset = OdometerCropDataset(args.data)
    print(f"Loaded {len(dataset)} labeled crops from {args.data}")

    model = build_model()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    ctc_loss = torch.nn.CTCLoss(blank=BLANK_IDX, zero_infinity=True)

    os.makedirs(args.out, exist_ok=True)
    print(
        "Model, optimizer, and loss are wired up. Fill in the DataLoader + "
        "image preprocessing (reuse preprocessing/pipeline.py so train-time and "
        "inference-time preprocessing match exactly) and the training loop, "
        "then checkpoint to "
        f"{args.out}/recognizer_best.pt"
    )
    print("Next: python training/export_onnx.py --weights "
          f"{args.out}/recognizer_best.pt --out models/recognizer.onnx --type recognizer")


if __name__ == "__main__":
    main()
