"""EEG band-power — the oscillatory feature. Cognitive load / motor imagery reshape rhythm *magnitude*
(theta rises / alpha suppresses with load; mu/beta desynchronize with imagery), which per-channel band-power
reads and covariance normalizes away."""
from __future__ import annotations

import numpy as np
from scipy.signal import welch

# workload-relevant EEG rhythms (theta ↑ / alpha ↓ with load); MI mu/beta live in this range too
CANONICAL_BANDS = (("theta", 4.0, 7.0), ("alpha", 8.0, 13.0), ("beta", 13.0, 30.0))


class BandPower:
    """EEG band-power — the oscillatory feature (free helpers folded in as staticmethods, public names kept)."""

    @staticmethod
    def band_powers(X: np.ndarray, fs: float, bands=CANONICAL_BANDS, *, relative: bool = False) -> np.ndarray:
        """Per-channel log band-power in each band -> `[n, ch*len(bands)]`. One Welch PSD over the time axis
        (vectorized across n and ch), then integrate each band.

        `relative=False` -> log absolute power (best *within*-subject; the absolute scale is subject-specific so
        it transfers poorly). `relative=True` -> each band as a FRACTION of the epoch's total band-power (per
        channel), which divides out the subject/session amplitude offset — the standard cross-subject fix."""
        nperseg = min(X.shape[2], int(round(fs * 2)))            # 2 s segments (or the whole epoch if shorter)
        freqs, psd = welch(X, fs=fs, nperseg=nperseg, axis=2)    # psd: [n, ch, f]
        P = np.stack([psd[:, :, (freqs >= lo) & (freqs < hi)].sum(axis=2) for _n, lo, hi in bands], axis=0)
        if relative:
            P = P / (P.sum(axis=0, keepdims=True) + 1e-12)      # fraction of total -> scale-free
        return np.concatenate([np.log(P[b] + 1e-12) for b in range(P.shape[0])], axis=1)
