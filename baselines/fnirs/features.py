"""fNIRS feature baseline — the field-standard decode: per-channel temporal features -> scaler -> LDA.

Unlike EEG (class signal in the covariance/oscillatory structure that CSP + Riemannian methods read), the
fNIRS workload signal is in the **amplitude and shape of the hemodynamic response**: how high ΔHbO rises
(mean, peak) and how steeply it climbs (slope) over the task window. So the canonical fNIRS feature set is
the per-channel temporal **mean + slope + peak** of each chromophore — exactly what covariance methods
discard by centering. Standardize (per feature) then LDA — the fNIRS-BCI workhorse (shrinkage-LDA on the
tiny per-subject sets); NOT CSP/Riemannian, which are EEG-native and mismatch fNIRS.

Refs: Noori 2016 (mean+peak ~93% HbO), Naseer & Hong 2015 review (mean+slope canonical), MNE-NIRS decoding
example (Scaler→Vectorizer→linear). See research/deep_dives/2026-07-01_fnirs_decoding_methods.md.

Interface = the harness contract: `fit(X, y) -> clf`, `score(clf, X) -> probs[n, C]`. X is [n, ch, t]
with channels = HbO then HbR (see core/data/fnirs/shin2017.py).
"""
from __future__ import annotations

import numpy as np

from baselines.base import Baseline
from core.features import amplitude_features


class FnirsLda(Baseline):
    """Per-channel mean+slope+peak features -> StandardScaler -> shrinkage-LDA (the fNIRS-BCI workhorse).
    The fitted pipeline lives on `self.pipe_`; features are `core.features.amplitude_features`."""

    def _build(self):
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        return make_pipeline(StandardScaler(),
                             LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto"))

    def fit(self, X, y):
        self.pipe_ = self._build()
        self.pipe_.fit(amplitude_features(X), y)
        return self

    def predict_proba(self, X):
        return self.pipe_.predict_proba(amplitude_features(X))


def fit(X: np.ndarray, y: np.ndarray) -> Baseline:
    """Back-compat shim — prefer `FnirsLda().fit(X, y)`."""
    return FnirsLda().fit(X, y)


def score(clf: Baseline, X: np.ndarray) -> np.ndarray:
    return clf.predict_proba(X)
