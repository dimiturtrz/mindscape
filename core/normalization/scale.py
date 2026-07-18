"""Constant-factor amplitude scaling — hand a pretrained backbone the input scale it was trained on."""
from __future__ import annotations

from typing import override

import numpy as np
from jaxtyping import Float, Int

from core.normalization.normalization import Normalizer


class Scale(Normalizer):
    """Multiply every sample by a constant `factor` — **amplitude-preserving**: relative channel amplitudes and
    waveform shape are untouched, only the overall scale changes. This is how a pretrained backbone is fed the
    scale it saw in pretraining rather than a z-score that flattens per-channel amplitude. CBraMod, for
    instance, was pretrained on microvolts ÷ 100. Stateless — nothing to fit, `groups` ignored."""

    def __init__(self, factor: float):
        self.factor = factor

    @override
    def apply(self, X: Float[np.ndarray, "n ch t"],
              groups: Int[np.ndarray, "n"] | None = None) -> Float[np.ndarray, "n ch t"]:
        return (X * self.factor).astype(np.float32)
