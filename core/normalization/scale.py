"""Constant-factor amplitude scaling — hand a pretrained backbone the input scale it was trained on."""
from __future__ import annotations

import numpy as np
from jaxtyping import Float

from core.normalization.normalization import Normalizer, NormContext


class Scale(Normalizer):
    """Multiply every sample by a constant `factor` — **amplitude-preserving**: relative channel amplitudes and
    waveform shape are untouched, only the overall scale changes. This is how a pretrained backbone is fed the
    scale it saw in pretraining rather than a z-score that flattens per-channel amplitude. CBraMod, for
    instance, was pretrained on microvolts ÷ 100, so its chain scales our raw signal into that range. Stateless
    — ignores `ctx`."""

    def __init__(self, factor: float):
        self.factor = factor

    def apply(self, X: Float[np.ndarray, "n ch t"], ctx: NormContext) -> Float[np.ndarray, "n ch t"]:
        return (X * self.factor).astype(np.float32)
