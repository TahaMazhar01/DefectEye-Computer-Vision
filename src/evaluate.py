"""Evaluate a trained PaDiM model: image/pixel AUROC + sample visualizations.

Example::

    python -m src.evaluate --data_root /kaggle/input/mvtec-ad \
        --category bottle --backbone wide_resnet50_2 \
        --model models/bottle_wide_resnet50_2.pt
"""

import argparse
import os

import cv2
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

from .dataset import MVTecDataset
from .padim import PaDiM
from .utils import denormalize, make_overlay, upsample_map


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--category", required=True)
    ap.add_argument("--backbone", default="resnet18",
                    choices=["resnet18", "wide_resnet50_2"])
    ap.add_argument("--model", required=True)
    ap.add_argument("--results", default="results")
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--n_samples", type=int, default=6)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = PaDiM(backbone=args.backbone, device=device)
    model.load_state(torch.load(args.model, map_location="cpu"))

    test_ds = MVTecDataset(args.data_root, args.category, split="test")
    loader = DataLoader(test_ds, batch_size=args.batch_size,
                        shuffle=False, num_workers=args.num_workers)
    cropsize = test_ds.cropsize

    img_scores, img_labels = [], []
    pix_scores, pix_masks = [], []
    sample_imgs, sample_maps, sample_gt = [], [], []

    for imgs, labels, masks in loader:
        up = upsample_map(model.predict(imgs), cropsize)   # (B,224,224)
        for b in range(imgs.shape[0]):
            img_scores.append(float(up[b].max()))
            img_labels.append(int(labels[b]))
            pix_scores.append(up[b].ravel())
            pix_masks.append(masks[b, 0].numpy().ravel().astype(np.uint8))
            if len(sample_imgs) < args.n_samples and int(labels[b]) == 1:
                sample_imgs.append(denormalize(imgs[b]))
                sample_maps.append(up[b])
                sample_gt.append(masks[b, 0].numpy())

    img_auroc = roc_auc_score(img_labels, img_scores)
    pix_auroc = roc_auc_score(np.concatenate(pix_masks), np.concatenate(pix_scores))
    print(f"\n=== {args.category} ({args.backbone}) ===")
    print(f"Image-level AUROC: {img_auroc:.4f}")
    print(f"Pixel-level AUROC: {pix_auroc:.4f}")

    os.makedirs(args.results, exist_ok=True)
    # qualitative grid: original | heatmap overlay | ground truth
    if sample_maps:
        gmin = min(m.min() for m in sample_maps)
        gmax = max(m.max() for m in sample_maps)
        rows = []
        for img, m, gt in zip(sample_imgs, sample_maps, sample_gt):
            norm = (m - gmin) / (gmax - gmin + 1e-8)
            overlay = make_overlay(img, norm)
            orig = (img[..., ::-1] * 255).astype(np.uint8)
            gt_vis = cv2.cvtColor((gt * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
            rows.append(cv2.hconcat([orig, overlay, gt_vis]))
        grid = cv2.vconcat(rows)
        grid_path = os.path.join(args.results, f"{args.category}_{args.backbone}_samples.png")
        cv2.imwrite(grid_path, grid)
        print(f"Saved samples -> {grid_path}  (columns: input | prediction | ground truth)")

    with open(os.path.join(args.results, f"{args.category}_{args.backbone}_metrics.txt"), "w") as f:
        f.write(f"category={args.category}\nbackbone={args.backbone}\n")
        f.write(f"image_auroc={img_auroc:.4f}\npixel_auroc={pix_auroc:.4f}\n")


if __name__ == "__main__":
    main()
