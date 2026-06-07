"""MVTec AD dataset loader.

Expected layout (the standard MVTec Anomaly Detection structure)::

    <root>/<category>/
        train/good/*.png            # defect-free, used for fitting
        test/good/*.png             # defect-free test images
        test/<defect_type>/*.png    # defective test images
        ground_truth/<defect_type>/*_mask.png

Training uses only `train/good`. The test split returns an image-level label
(0 = good, 1 = defect) and a pixel mask so we can report pixel-level AUROC.
"""

import glob
import os

import torch
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import Dataset

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def build_transforms(resize: int = 256, cropsize: int = 224):
    img_t = T.Compose([
        T.Resize(resize, T.InterpolationMode.BILINEAR),
        T.CenterCrop(cropsize),
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    mask_t = T.Compose([
        T.Resize(resize, T.InterpolationMode.NEAREST),
        T.CenterCrop(cropsize),
        T.ToTensor(),
    ])
    return img_t, mask_t


class MVTecDataset(Dataset):
    def __init__(self, root, category, split="train", resize=256, cropsize=224):
        self.root = os.path.join(root, category)
        self.split = split
        self.cropsize = cropsize
        self.img_t, self.mask_t = build_transforms(resize, cropsize)
        self.samples = []  # list of (img_path, label, mask_path | None)

        if split == "train":
            pattern = os.path.join(self.root, "train", "good", "*.*")
            for p in sorted(glob.glob(pattern)):
                self.samples.append((p, 0, None))
        else:
            test_dir = os.path.join(self.root, "test")
            for defect in sorted(os.listdir(test_dir)):
                ddir = os.path.join(test_dir, defect)
                if not os.path.isdir(ddir):
                    continue
                for p in sorted(glob.glob(os.path.join(ddir, "*.*"))):
                    if defect == "good":
                        self.samples.append((p, 0, None))
                    else:
                        stem = os.path.splitext(os.path.basename(p))[0]
                        mask = os.path.join(self.root, "ground_truth",
                                            defect, f"{stem}_mask.png")
                        self.samples.append((p, 1, mask if os.path.exists(mask) else None))

        if not self.samples:
            raise RuntimeError(
                f"No images found under {self.root} (split={split}). "
                "Check the dataset path / category name.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        path, label, mask_path = self.samples[i]
        img = self.img_t(Image.open(path).convert("RGB"))
        if mask_path:
            mask = self.mask_t(Image.open(mask_path).convert("L"))
            mask = (mask > 0.5).float()
        else:
            mask = torch.zeros(1, self.cropsize, self.cropsize)
        return img, label, mask
