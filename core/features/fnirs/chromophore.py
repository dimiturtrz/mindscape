"""fNIRS chromophore combination — CBSI, the two-wavelength neural estimate (single-modality fNIRS prep)."""
from __future__ import annotations

import numpy as np


class Chromophore:
    """fNIRS chromophore combination — CBSI, the two-wavelength neural estimate (helper folded in as a
    staticmethod, public name kept)."""

    @staticmethod
    def cbsi_neural(hbo: np.ndarray, hbr: np.ndarray) -> np.ndarray:
        """CBSI neural map (Cui 2010) — activation makes HbO/HbR anti-correlated, motion/systemic makes them
        common-mode; `HbO − α·HbR` (α = std(HbO)/std(HbR)) keeps the neural part, cancels the systemic. Uses BOTH
        chromophores — the whole point of two wavelengths. `hbo`/`hbr` are `[n, ch, t]` -> `[n, ch, t]`."""
        a = hbo.std(axis=2, keepdims=True) / (hbr.std(axis=2, keepdims=True) + 1e-9)
        return 0.5 * (hbo - a * hbr)
