"""Per-channel z-score normalizer — the standard numerical conditioner."""
from __future__ import annotations

import numpy as np
from jaxtyping import Float

from core.normalization.normalization import Normalizer

_EPS = 1e-7


class ZScore(Normalizer):
    """Per-channel z-score over time, per epoch: each channel → zero mean, unit variance within a trial. The
    numerical conditioner that keeps a downstream net's running stats well-scaled. Stateless — every trial
    normalizes against itself, so there is nothing to fit."""

    def apply(self, X: Float[np.ndarray, "n ch t"]) -> Float[np.ndarray, "n ch t"]:
        mean = X.mean(axis=2, keepdims=True)
        std = X.std(axis=2, keepdims=True)
        return ((X - mean) / (std + _EPS)).astype(np.float32)
