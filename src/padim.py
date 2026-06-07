"""PaDiM: Patch Distribution Modeling for anomaly detection.

Reference: Defard et al., "PaDiM: a Patch Distribution Modeling Framework
for Anomaly Detection and Localization" (2020).

Idea
----
1. Pass defect-free images through a frozen pretrained backbone.
2. For every spatial patch position, model the embedding vectors across all
   training images as a multivariate Gaussian N(mean, covariance).
3. At test time, the Mahalanobis distance of a patch to its Gaussian is the
   anomaly score. High distance = the patch looks unlike anything seen in
   normal data = likely a defect.

No gradient training happens here -- "fitting" is just estimating the
per-position mean and covariance, which runs fine even on CPU.
"""

import numpy as np
import torch
from tqdm import tqdm

from .backbone import FeatureExtractor, TOTAL_CHANNELS, embed
from .utils import upsample_map

# number of randomly selected channel dimensions (PaDiM paper values).
# Random selection is as effective as PCA but far cheaper.
RAND_DIMS = {"resnet18": 100, "wide_resnet50_2": 550}


class PaDiM:
    def __init__(self, backbone: str = "resnet18", device: str = "cpu", seed: int = 1024):
        self.backbone_name = backbone
        self.device = device
        self.extractor = FeatureExtractor(backbone).to(device)

        self.d = RAND_DIMS[backbone]
        g = torch.Generator().manual_seed(seed)
        self.idx = torch.randperm(TOTAL_CHANNELS[backbone], generator=g)[: self.d]

        # learned distribution
        self.mean = None        # (d, P)
        self.inv_cov = None     # (P, d, d)
        self.hw = None          # (H, W) of the feature grid

        # calibration (set by calibrate())
        self.score_min = 0.0
        self.score_max = 1.0
        self.threshold = 0.5

    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def _embed_batch(self, x):
        feats = self.extractor(x.to(self.device))
        emb = embed(feats)            # (B, C, H, W)
        return emb[:, self.idx]       # (B, d, H, W)

    @staticmethod
    def _imgs_from_batch(batch):
        return batch[0] if isinstance(batch, (list, tuple)) else batch

    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def fit(self, loader, eps: float = 0.01):
        """Estimate the per-position Gaussian from defect-free images."""
        chunks = []
        for batch in tqdm(loader, desc="Extracting train features"):
            chunks.append(self._embed_batch(self._imgs_from_batch(batch)).cpu())
        emb = torch.cat(chunks, dim=0)          # (N, d, H, W)
        N, d, H, W = emb.shape
        self.hw = (H, W)
        P = H * W
        emb = emb.reshape(N, d, P)              # (N, d, P)

        self.mean = emb.mean(dim=0)             # (d, P)
        centered = emb - self.mean.unsqueeze(0)  # (N, d, P)

        identity = torch.eye(d)
        inv_cov = torch.empty(P, d, d)
        # process positions in chunks to keep peak memory bounded
        step = 256
        for s in tqdm(range(0, P, step), desc="Fitting Gaussians"):
            e = min(s + step, P)
            c = centered[:, :, s:e]                       # (N, d, p)
            cov = torch.einsum("ndp,nep->pde", c, c) / (N - 1)
            cov = cov + eps * identity
            inv_cov[s:e] = torch.linalg.inv(cov)
        self.inv_cov = inv_cov
        return self

    @torch.no_grad()
    def predict(self, x):
        """Return the raw anomaly map (B, H, W) of Mahalanobis distances."""
        emb = self._embed_batch(x).cpu()        # (B, d, H, W)
        B, d, H, W = emb.shape
        emb = emb.reshape(B, d, H * W)
        diff = (emb - self.mean.unsqueeze(0)).permute(0, 2, 1)  # (B, P, d)
        tmp = torch.einsum("bpd,pde->bpe", diff, self.inv_cov)  # (B, P, d)
        dist2 = torch.einsum("bpe,bpe->bp", tmp, diff)          # (B, P)
        dist = torch.sqrt(torch.clamp(dist2, min=0.0))
        return dist.reshape(B, H, W)

    @torch.no_grad()
    def calibrate(self, loader, cropsize: int = 224, k: float = 3.0):
        """Derive an image-level threshold + score range from normal data.

        Since training images are all defect-free, mean + k*std of their
        scores is a sensible unsupervised PASS/FAIL boundary for the demo.
        """
        scores = []
        for batch in tqdm(loader, desc="Calibrating"):
            amap = self.predict(self._imgs_from_batch(batch))
            up = upsample_map(amap, cropsize)
            scores.extend(up.reshape(up.shape[0], -1).max(axis=1).tolist())
        scores = np.asarray(scores)
        self.score_min = float(scores.min())
        self.score_max = float(scores.max())
        self.threshold = float(scores.mean() + k * scores.std())
        return self

    # ------------------------------------------------------------------ #
    def state(self):
        return {
            "backbone": self.backbone_name,
            "idx": self.idx,
            "mean": self.mean,
            "inv_cov": self.inv_cov,
            "hw": self.hw,
            "score_min": self.score_min,
            "score_max": self.score_max,
            "threshold": self.threshold,
        }

    def load_state(self, state):
        assert state["backbone"] == self.backbone_name, "backbone mismatch"
        self.idx = state["idx"]
        self.mean = state["mean"]
        self.inv_cov = state["inv_cov"]
        self.hw = state["hw"]
        self.score_min = state.get("score_min", 0.0)
        self.score_max = state.get("score_max", 1.0)
        self.threshold = state.get("threshold", 0.5)
        return self
