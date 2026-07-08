"""Brain-camera decoder — a tiny 3D-CNN over the fused `[C, H, W, T]` EEG+fNIRS surface-video.

The honest test of the spatiotemporal fusion: does reading the co-registered geometry + time (which flat
feature-concat destroyed) cash any of the oracle headroom the collapsed-feature fusions couldn't? Kept
deliberately tiny — 702 blocks, 26 subjects — heavy pooling + dropout + weight decay, or it just memorizes.
"""
from __future__ import annotations

import numpy as np
import torch
from pydantic import BaseModel
from torch import nn


class BrainCameraConfig(BaseModel):
    """BrainCameraNet training hyperparameters — kept heavily regularized for the tiny fused set."""
    n_classes: int = 3
    epochs: int = 40
    lr: float = 3e-3
    weight_decay: float = 1e-2
    dropout: float = 0.5
    batch: int = 32
    seed: int = 0


class BrainCameraNet:
    """Tiny 3D-CNN: two conv blocks (aggressive spatial+temporal pooling) → global pool → dropout → linear.
    sklearn-ish `fit`/`predict_proba` so it drops into the same CV harness. CUDA if present."""

    def __init__(self, config: BrainCameraConfig):
        self.cfg = config

    def _build(self, n_channels):
        return nn.Sequential(
            nn.Conv3d(n_channels, 16, kernel_size=3, padding=1), nn.BatchNorm3d(16), nn.ReLU(),
            nn.MaxPool3d((2, 2, 4)),                                    # H,W /2, T /4
            nn.Conv3d(16, 32, kernel_size=3, padding=1), nn.BatchNorm3d(32), nn.ReLU(),
            nn.AdaptiveAvgPool3d(1), nn.Flatten(),                      # -> [B, 32]
            nn.Dropout(self.cfg.dropout), nn.Linear(32, self.cfg.n_classes),
        )

    def fit(self, X, y):
        cfg = self.cfg
        torch.manual_seed(cfg.seed)
        self.dev_ = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net_ = self._build(X.shape[1]).to(self.dev_)
        opt = torch.optim.AdamW(self.net_.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        lossf = torch.nn.CrossEntropyLoss()
        inputs = torch.as_tensor(X, dtype=torch.float32)
        labels = torch.as_tensor(np.asarray(y), dtype=torch.long)
        n = len(labels)
        rng = np.random.default_rng(cfg.seed)
        self.net_.train()
        for _ in range(cfg.epochs):
            for idx in np.array_split(rng.permutation(n), max(1, n // cfg.batch)):
                xb = inputs[idx].to(self.dev_)
                yb = labels[idx].to(self.dev_)
                opt.zero_grad()
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=(self.dev_.type == "cuda")):
                    loss = lossf(self.net_(xb), yb)
                loss.backward()                            # bf16 autocast; no GradScaler needed
                opt.step()
        return self

    @property
    def classes_(self):
        return np.arange(self.cfg.n_classes)

    def predict_proba(self, X):
        self.net_.eval()
        inputs = torch.as_tensor(X, dtype=torch.float32)
        out = []
        with torch.no_grad():
            for idx in np.array_split(np.arange(len(inputs)), max(1, len(inputs) // 64)):
                p = torch.softmax(self.net_(inputs[idx].to(self.dev_)), dim=1)
                out.append(p.cpu().numpy())
        return np.concatenate(out, axis=0)
