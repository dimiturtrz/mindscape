"""Input standardizers for the decoders — the per-channel scaling stage.

Extracted from decoders.py so the trainer stays about *training*, and independently testable + swappable.
Standardizers share one interface: `fit(X) -> self`, `__call__(X) -> X'`. `Transforms.standardizer`
(transforms.py) builds one by name off the `STANDARDIZERS` table here.
"""
from __future__ import annotations

from typing import Protocol

import numpy as np
from braindecode.preprocessing import exponential_moving_standardize
from jaxtyping import Float


class Standardizer(Protocol):
    """Per-channel standardizer contract: `fit(X) -> self` then `__call__(X) -> X'` (fit on train only)."""

    def fit(self, X: Float[np.ndarray, "n ch t"]) -> Standardizer:
        ...

    def __call__(self, X: Float[np.ndarray, "n ch t"]) -> Float[np.ndarray, "n ch t"]:
        ...


class ZScore:
    """Per-channel z-score, fit on train (mean/std over epochs+time per channel)."""
    def fit(self, X: Float[np.ndarray, "n ch t"]):
        self.mu = X.mean(axis=(0, 2), keepdims=True)
        self.sd = X.std(axis=(0, 2), keepdims=True) + 1e-6
        return self

    def __call__(self, X: Float[np.ndarray, "n ch t"]) -> Float[np.ndarray, "n ch t"]:
        return ((X - self.mu) / self.sd).astype(np.float32)


class ExpMovingStd:
    """Exponential-moving standardization (braindecode-canonical 2a preprocessing), per epoch.
    NOTE: the *correct* EMS runs on the continuous recording before epoching (see
    core/data/eeg/braindecode_pre.py); this per-epoch form is a fallback. Stateless (no fit)."""
    def __init__(self, factor_new: float = 1e-3, init_block_size: int = 1000):
        self.factor_new, self.init_block_size = factor_new, init_block_size

    def fit(self, X: Float[np.ndarray, "n ch t"]):
        return self

    def __call__(self, X: Float[np.ndarray, "n ch t"]) -> Float[np.ndarray, "n ch t"]:
        ib = min(self.init_block_size, X.shape[2])
        return np.stack([exponential_moving_standardize(e, factor_new=self.factor_new, init_block_size=ib)
                         for e in X]).astype(np.float32)


class Identity:
    """No-op — for data already standardized upstream (continuous-signal EMS in preprocessing)."""
    def fit(self, X: Float[np.ndarray, "n ch t"]):
        return self

    def __call__(self, X: Float[np.ndarray, "n ch t"]) -> Float[np.ndarray, "n ch t"]:
        return X.astype(np.float32)


STANDARDIZERS = {"zscore": ZScore, "ems": ExpMovingStd, "none": Identity}
