"""EEG↔fNIRS hemodynamic coupling — derive the neurovascular offset + decay from the data (cross-modal).

The genuinely-fusion timing bridge: neural activity (EEG envelope) drives the blood response (fNIRS CBSI) through
a causal hemodynamic kernel. We FIT that kernel instead of hardcoding a 5 s shift — no magic numbers.
"""
from __future__ import annotations

import numpy as np
from jaxtyping import Float
from scipy.signal import fftconvolve

_HRF_WIDTH = (2.0, 5.0)      # physiological hemodynamic dispersion (s): HRF FWHM ~5 s -> std ~2-5 s. A width
                             # FLOOR stops the coupling fit railing to a degenerate spike on weak/short data.


class Coupling:
    """EEG↔fNIRS hemodynamic coupling — derive the neurovascular offset + decay from the data (free helpers
    folded in as staticmethods, public names kept)."""

    @staticmethod
    def _gamma_kernel(t: np.ndarray, peak: float, width: float) -> np.ndarray:
        """Causal single-gamma hemodynamic kernel on time axis `t` (s), parameterized by its center-of-mass `peak`
        (the delay) and dispersion `width` (both seconds), normalized to unit area. `mean = a·b = peak`,
        `std = √a·b = width` -> shape `a = (peak/width)²`, scale `b = width²/peak`."""
        a = (peak / width) ** 2
        b = width ** 2 / peak
        g = np.where(t > 0, np.power(np.clip(t, 1e-6, None), a - 1.0) * np.exp(-t / b), 0.0)
        s = g.sum()
        return g / s if s > 0 else g

    @staticmethod
    def estimate_coupling(drive: Float[np.ndarray, "n t"], resp: Float[np.ndarray, "n t"], fs: float, *,
                          lag_max: float = 12.0,
                          klen: float = 30.0):
        """Derive the EEG→blood coupling from the data instead of hardcoding a 5 s shift. `drive` = EEG band-power
        envelope, `resp` = fNIRS CBSI, both `[n, T]` global (channel-mean) on ONE **zero-lag** grid at rate `fs`.
        Fit a causal gamma kernel `g(peak, width)` maximizing the correlation between the EEG-**predicted** blood
        `drive ⊛ g` and the measured `resp` (grid search: delay 2-12 s; dispersion constrained to the physiological
        HRF range `_HRF_WIDTH` — an HRF is a smooth bump, NOT a spike, so a width floor keeps the fit from railing to
        a degenerate delta on weak/short data: physics > statistics). Returns `(lag, decay, beta)`: `lag` = kernel
        center-of-mass (s, the offset), `decay` = tail time-constant `b` (s, the smearing), `beta` = least-squares
        gain (EEG-envelope → CBSI unit bridge). Self-calibrates per subject — neurovascular latency varies."""
        d = drive.astype(np.float64)
        r = resp.astype(np.float64)
        rz = (r - r.mean(1, keepdims=True)) / (r.std(1, keepdims=True) + 1e-9)
        tk = np.arange(0, klen, 1.0 / fs)
        best = (-1.0, 6.0, _HRF_WIDTH[0])                                  # (|corr|², peak, width)
        for peak in np.arange(2.0, lag_max + 1e-9, 0.5):
            for width in np.arange(_HRF_WIDTH[0], min(_HRF_WIDTH[1], peak) + 1e-9, 0.5):
                g = Coupling._gamma_kernel(tk, peak, width)
                pred = fftconvolve(d, g[None, :], axes=1)[:, :d.shape[1]]
                pz = (pred - pred.mean(1, keepdims=True)) / (pred.std(1, keepdims=True) + 1e-9)
                score = float((pz * rz).mean(1).mean()) ** 2               # coupling STRENGTH (sign-agnostic — β
                if score > best[0]:                                       # carries the direction; don't assume +)
                    best = (score, peak, width)
        _, peak, width = best
        pred = fftconvolve(d, Coupling._gamma_kernel(tk, peak, width)[None, :], axes=1)[:, :d.shape[1]]
        beta = float((pred * r).sum() / ((pred ** 2).sum() + 1e-12))       # raw LS gain (units bridge)
        return float(peak), float(width ** 2 / peak), beta
