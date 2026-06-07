"""Visualization and post-processing helpers."""

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD = np.array([0.229, 0.224, 0.225])


def denormalize(img_tensor):
    """(3,H,W) normalized tensor -> (H,W,3) RGB float image in [0,1]."""
    img = img_tensor.detach().cpu().numpy().transpose(1, 2, 0)
    img = img * IMAGENET_STD + IMAGENET_MEAN
    return np.clip(img, 0.0, 1.0)


def upsample_map(anomaly_map, size, sigma: int = 4):
    """Upsample a (B,H,W) feature-resolution map to (B,size,size) numpy and
    apply Gaussian smoothing (standard PaDiM post-processing)."""
    if isinstance(size, int):
        size = (size, size)
    m = F.interpolate(anomaly_map.unsqueeze(1).float(), size=size,
                      mode="bilinear", align_corners=False)
    m = m.squeeze(1).cpu().numpy()
    for i in range(m.shape[0]):
        m[i] = gaussian_filter(m[i], sigma=sigma)
    return m


def make_overlay(img01, score01, alpha: float = 0.5):
    """Blend a JET heatmap over the image. Returns uint8 BGR (for cv2.imwrite)."""
    heat = cv2.applyColorMap((score01 * 255).astype(np.uint8), cv2.COLORMAP_JET)
    img_bgr = (img01[..., ::-1] * 255).astype(np.uint8)
    return cv2.addWeighted(img_bgr, 1 - alpha, heat, alpha, 0)
