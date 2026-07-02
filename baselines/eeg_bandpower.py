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

from baselines.base import Baseline
from core.features import band_powers


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
        self.pipe_.fit(band_powers(X, self.fs, relative=self.relative), y)
        return self

    def predict_proba(self, X):
        return self.pipe_.predict_proba(band_powers(X, self.fs, relative=self.relative))


def fit(X: np.ndarray, y: np.ndarray) -> Baseline:
    """Back-compat shim — prefer `EegBandpower().fit(X, y)`."""
    return EegBandpower().fit(X, y)


def score(clf: Baseline, X: np.ndarray) -> np.ndarray:
    return clf.predict_proba(X)
