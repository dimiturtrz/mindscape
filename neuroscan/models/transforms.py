"""Input transforms for the decoders — standardization + sliding-window crops.

Extracted from decoders.py so the trainer stays about *training*, and the transforms are independently
testable + swappable. Standardizers share one interface: `fit(X) -> self`, `__call__(X) -> X'`.
"""
from __future__ import annotations

import numpy as np
from braindecode.preprocessing import exponential_moving_standardize
from jaxtyping import Float


class ZScore:
    """Per-channel z-score, fit on train (mean/std over epochs+time per channel)."""
    def fit(self, X):
        self.mu = X.mean(axis=(0, 2), keepdims=True)
        self.sd = X.std(axis=(0, 2), keepdims=True) + 1e-6
        return self

    def __call__(self, X):
        return ((X - self.mu) / self.sd).astype(np.float32)


class ExpMovingStd:
    """Exponential-moving standardization (braindecode-canonical 2a preprocessing), per epoch.
    NOTE: the *correct* EMS runs on the continuous recording before epoching (see
    core/data/eeg/braindecode_pre.py); this per-epoch form is a fallback. Stateless (no fit)."""
    def __init__(self, factor_new: float = 1e-3, init_block_size: int = 1000):
        self.factor_new, self.init_block_size = factor_new, init_block_size

    def fit(self, X):
        return self

    def __call__(self, X):
        ib = min(self.init_block_size, X.shape[2])
        return np.stack([exponential_moving_standardize(e, factor_new=self.factor_new, init_block_size=ib)
                         for e in X]).astype(np.float32)


class Identity:
    """No-op — for data already standardized upstream (continuous-signal EMS in preprocessing)."""
    def fit(self, X):
        return self

    def __call__(self, X):
        return X.astype(np.float32)


STANDARDIZERS = {"zscore": ZScore, "ems": ExpMovingStd, "none": Identity}


class Transforms:
    @staticmethod
    def standardizer(kind: str):
        """Build a standardizer by name; unknown -> z-score."""
        return STANDARDIZERS.get(kind, ZScore)()

    @staticmethod
    def crops(X: Float[np.ndarray, "n ch t"], crop_len: int, n_crops: int):
        """Cut each trial [.,ch,T] into `n_crops` evenly-spaced windows of length `crop_len`.
        Returns (Xc [N*n_crops, ch, crop_len], trial_index [N*n_crops])."""
        T = X.shape[2]
        starts = np.linspace(0, T - crop_len, n_crops).round().astype(int)
        idx = starts[:, None] + np.arange(crop_len)             # [n_crops, crop_len] sample indices per crop
        Xc = X[:, :, idx].transpose(2, 0, 1, 3).reshape(-1, X.shape[1], crop_len)   # [n_crops*N, ch, crop_len]
        tidx = np.tile(np.arange(len(X)), n_crops)              # trial each crop-row came from
        return Xc, tidx
