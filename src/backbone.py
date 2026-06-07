"""Pretrained CNN backbones used as frozen feature extractors.

PaDiM does not train any weights. It reuses the rich multi-scale features of
an ImageNet-pretrained network and only *models the distribution* of those
features on defect-free images. This module exposes the intermediate feature
maps (layer1, layer2, layer3) that PaDiM relies on.
"""

import torch
import torch.nn.functional as F
from torchvision.models import (
    resnet18,
    ResNet18_Weights,
    wide_resnet50_2,
    Wide_ResNet50_2_Weights,
)

# total channels produced by (layer1, layer2, layer3) for each backbone
TOTAL_CHANNELS = {
    "resnet18": 64 + 128 + 256,          # 448
    "wide_resnet50_2": 256 + 512 + 1024,  # 1792
}


class FeatureExtractor(torch.nn.Module):
    """Frozen ResNet that returns feature maps from layer1, layer2, layer3."""

    def __init__(self, backbone: str = "resnet18"):
        super().__init__()
        if backbone == "resnet18":
            self.model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        elif backbone == "wide_resnet50_2":
            self.model = wide_resnet50_2(weights=Wide_ResNet50_2_Weights.IMAGENET1K_V1)
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

        self.backbone = backbone
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

        self._features = {}
        self.model.layer1.register_forward_hook(self._hook("layer1"))
        self.model.layer2.register_forward_hook(self._hook("layer2"))
        self.model.layer3.register_forward_hook(self._hook("layer3"))

    def _hook(self, name):
        def fn(_module, _inp, out):
            self._features[name] = out
        return fn

    @torch.no_grad()
    def forward(self, x):
        self._features = {}
        self.model(x)
        return [self._features["layer1"],
                self._features["layer2"],
                self._features["layer3"]]


def embed(features):
    """Concatenate multi-scale features by upsampling each map to the
    spatial size of the first (largest) map. Returns (B, C, H, W)."""
    target_size = features[0].shape[-2:]
    parts = [features[0]]
    for f in features[1:]:
        parts.append(F.interpolate(f, size=target_size, mode="nearest"))
    return torch.cat(parts, dim=1)
