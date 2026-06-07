"""Fit PaDiM on one MVTec category and save the model.

Example::

    python -m src.train --data_root /kaggle/input/mvtec-ad \
        --category bottle --backbone wide_resnet50_2
"""

import argparse
import os

import torch
from torch.utils.data import DataLoader

from .dataset import MVTecDataset
from .padim import PaDiM


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--category", required=True)
    ap.add_argument("--backbone", default="resnet18",
                    choices=["resnet18", "wide_resnet50_2"])
    ap.add_argument("--out", default="models")
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--num_workers", type=int, default=2)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[DefectEye] device={device} backbone={args.backbone} category={args.category}")

    train_ds = MVTecDataset(args.data_root, args.category, split="train")
    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=False, num_workers=args.num_workers)
    print(f"[DefectEye] {len(train_ds)} defect-free training images")

    model = PaDiM(backbone=args.backbone, device=device)
    model.fit(train_loader)
    model.calibrate(train_loader)
    print(f"[DefectEye] threshold={model.threshold:.3f} "
          f"score_range=({model.score_min:.3f}, {model.score_max:.3f})")

    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, f"{args.category}_{args.backbone}.pt")
    torch.save(model.state(), path)
    print(f"[DefectEye] saved -> {path}")


if __name__ == "__main__":
    main()
