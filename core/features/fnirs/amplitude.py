"""fNIRS amplitude — the hemodynamic feature. Cognitive load lives in the *amplitude and shape* of the
ΔHbO/ΔHbR response (how high it rises, how steeply), exactly what covariance methods discard by centering.
Per-channel mean + slope + peak is the field-standard fNIRS-BCI feature triple."""
from __future__ import annotations

from functools import lru_cache

import numpy as np
from jaxtyping import Float


class Amplitude:
    """fNIRS amplitude — the hemodynamic feature (free helpers folded in as staticmethods, public names kept)."""

    @staticmethod
    @lru_cache(maxsize=8)
    def time_axis(t: int) -> tuple[np.ndarray, float]:
        """Centred time axis + its sum-of-squares (the OLS-slope denominator) — constants for a window length t,
        so cache them. f32 to keep the feature path f32; the sum-of-squares is f64 for a stable denominator."""
        tc = (np.arange(t) - (t - 1) / 2.0).astype(np.float32)
        return tc, float((tc.astype(np.float64) ** 2).sum())

    @staticmethod
    def amplitude_features(X: Float[np.ndarray, "n ch t"]) -> Float[np.ndarray, "n three_ch"]:
        """Per-channel temporal mean + slope + peak -> `[n, 3*ch]` — the canonical fNIRS feature triple (the
        hemodynamic response's amplitude/shape). peak = the extreme deviation (max |value|, signed): HbO rises
        positive, HbR dips negative."""
        tc, tc_ss = Amplitude.time_axis(X.shape[2])
        mean = X.mean(axis=2)                                    # response amplitude
        slope = (X * tc).sum(axis=2) / tc_ss                     # response trend (OLS)
        peak = np.take_along_axis(X, np.abs(X).argmax(2)[:, :, None], axis=2)[:, :, 0]   # signed extreme
        return np.concatenate([mean, slope, peak], axis=1)      # native dtype in -> out; LDA needs no f64
