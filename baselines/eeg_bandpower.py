"""EEG band-power baseline — the workload-native decode: per-channel θ/α/β log-power -> scaler -> LDA.

Covariance methods (CSP, Riemannian) read the *spatial covariance* of the oscillations and are the
EEG-native tool for **motor imagery** (lateralized mu/beta ERD is a covariance-structure signal). But n-back
**workload** is a different signal: its signature is the *magnitude* of specific rhythms — frontal-midline
**theta rises** and parietal **alpha suppresses** as load grows (Klimesch 1999; Gevins & Smith 2003). That is
absolute band-power, which covariance normalizes away — the same mismatch that makes covariance fail on the
fNIRS amplitude signal (see baselines/fnirs_features.py). A single *broadband* log-variance (4–30 Hz) also
misses it: theta↑ and alpha↓ partly cancel inside one band. So the right EEG-workload feature is band-power
split into theta / alpha / beta per channel, then shrinkage-LDA — mirroring the fNIRS workhorse on the EEG side.

Interface = the harness contract: `fit(X, y) -> self`, `predict_proba(X) -> probs[n, C]`. X is [n, ch, t]
sampled at `fs` (the epoch resample rate; the Shin EEG recipe is 100 Hz over a 40 s block).
"""
from __future__ import annotations

import numpy as np

from baselines.base import Baseline

# workload-relevant rhythms; theta rises with load, alpha suppresses — split so they don't cancel
_BANDS = (("theta", 4.0, 7.0), ("alpha", 8.0, 13.0), ("beta", 13.0, 30.0))


def _bandpower(X: np.ndarray, fs: float, relative: bool = False) -> np.ndarray:
    """Per-channel band-power in theta/alpha/beta -> [n, ch*3]. One Welch PSD over the time axis
    (vectorized across n and ch), then integrate each band.

    `relative=False` -> log absolute power (best *within*-subject, but the absolute scale is subject-specific
    so it transfers poorly). `relative=True` -> each band as a FRACTION of the epoch's total band-power
    (per channel), which divides out the subject/session amplitude offset — the standard cross-subject fix,
    and it needs no subject labels so it still satisfies the fit(X, y) contract."""
    from scipy.signal import welch

    nperseg = min(X.shape[2], int(round(fs * 2)))            # 2 s segments (or the whole epoch if shorter)
    freqs, psd = welch(X, fs=fs, nperseg=nperseg, axis=2)    # psd: [n, ch, f]
    bands = []
    for _name, lo, hi in _BANDS:
        sel = (freqs >= lo) & (freqs < hi)
        bands.append(psd[:, :, sel].sum(axis=2))            # [n, ch] absolute power per band
    P = np.stack(bands, axis=0)                             # [band, n, ch]
    if relative:
        P = P / (P.sum(axis=0, keepdims=True) + 1e-12)      # fraction of total -> scale-free
    feats = [np.log(P[b] + 1e-12) for b in range(P.shape[0])]
    return np.concatenate(feats, axis=1)                    # [n, ch*len(_BANDS)]


class EegBandpower(Baseline):
    """θ/α/β per-channel log-band-power -> StandardScaler -> shrinkage-LDA. The workload-native EEG decode,
    counterpart to FnirsLda on the hemodynamic side. `fs` = the epoch sample rate (Hz)."""

    def __init__(self, fs: float = 100.0, relative: bool = False):
        self.fs = fs
        self.relative = relative

    def _build(self):
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        return make_pipeline(StandardScaler(),
                             LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto"))

    def fit(self, X, y):
        self.pipe_ = self._build()
        self.pipe_.fit(_bandpower(X, self.fs, self.relative), y)
        return self

    def predict_proba(self, X):
        return self.pipe_.predict_proba(_bandpower(X, self.fs, self.relative))


def fit(X: np.ndarray, y: np.ndarray) -> Baseline:
    """Back-compat shim — prefer `EegBandpower().fit(X, y)`."""
    return EegBandpower().fit(X, y)


def score(clf: Baseline, X: np.ndarray) -> np.ndarray:
    return clf.predict_proba(X)
