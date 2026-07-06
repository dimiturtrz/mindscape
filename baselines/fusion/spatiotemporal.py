"""Brain-camera decoder — a tiny 3D-CNN over the fused `[C, H, W, T]` EEG+fNIRS surface-video.

The honest test of the spatiotemporal fusion: does reading the co-registered geometry + time (which flat
feature-concat destroyed) cash any of the oracle headroom the collapsed-feature fusions couldn't? Kept
deliberately tiny — 702 blocks, 26 subjects — heavy pooling + dropout + weight decay, or it just memorizes.
"""
from __future__ import annotations

import numpy as np


class BrainCameraNet:
    """Tiny 3D-CNN: two conv blocks (aggressive spatial+temporal pooling) → global pool → dropout → linear.
    sklearn-ish `fit`/`predict_proba` so it drops into the same CV harness. CUDA if present."""

    def __init__(self, n_classes=3, epochs=40, lr=3e-3, weight_decay=1e-2, dropout=0.5, batch=32, seed=0):
        self.n_classes = n_classes
        self.epochs = epochs
        self.lr = lr
        self.weight_decay = weight_decay
        self.dropout = dropout
        self.batch = batch
        self.seed = seed

    def _build(self, C):
        import torch.nn as nn
        return nn.Sequential(
            nn.Conv3d(C, 16, kernel_size=3, padding=1), nn.BatchNorm3d(16), nn.ReLU(),
            nn.MaxPool3d((2, 2, 4)),                                    # H,W /2, T /4
            nn.Conv3d(16, 32, kernel_size=3, padding=1), nn.BatchNorm3d(32), nn.ReLU(),
            nn.AdaptiveAvgPool3d(1), nn.Flatten(),                      # -> [B, 32]
            nn.Dropout(self.dropout), nn.Linear(32, self.n_classes),
        )

    def fit(self, X, y):
        import torch
        torch.manual_seed(self.seed)
        self.dev_ = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net_ = self._build(X.shape[1]).to(self.dev_)
        opt = torch.optim.AdamW(self.net_.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        lossf = torch.nn.CrossEntropyLoss()
        Xt = torch.as_tensor(X, dtype=torch.float32)
        yt = torch.as_tensor(np.asarray(y), dtype=torch.long)
        n = len(yt)
        rng = np.random.default_rng(self.seed)
        self.net_.train()
        for _ in range(self.epochs):
            for idx in np.array_split(rng.permutation(n), max(1, n // self.batch)):
                xb = Xt[idx].to(self.dev_); yb = yt[idx].to(self.dev_)
                opt.zero_grad()
                lossf(self.net_(xb), yb).backward()
                opt.step()
        return self

    @property
    def classes_(self):
        return np.arange(self.n_classes)

    def predict_proba(self, X):
        import torch
        self.net_.eval()
        Xt = torch.as_tensor(X, dtype=torch.float32)
        out = []
        with torch.no_grad():
            for idx in np.array_split(np.arange(len(Xt)), max(1, len(Xt) // 64)):
                p = torch.softmax(self.net_(Xt[idx].to(self.dev_)), dim=1)
                out.append(p.cpu().numpy())
        return np.concatenate(out, axis=0)
