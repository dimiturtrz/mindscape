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


def _features(X: np.ndarray) -> np.ndarray:
    """Per-channel temporal mean + slope + peak -> [n, 3*ch]. The canonical fNIRS feature triple.
    peak = the extreme deviation (max |value|, signed) — HbO rises positive, HbR dips negative."""
    n, ch, t = X.shape
    tc = np.arange(t) - (t - 1) / 2.0                       # centred time axis
    mean = X.mean(axis=2)                                   # response amplitude
    slope = (X * tc).sum(axis=2) / (tc ** 2).sum()          # response trend (OLS)
    peak = np.take_along_axis(X, np.abs(X).argmax(2)[:, :, None], axis=2)[:, :, 0]  # signed extreme
    return np.concatenate([mean, slope, peak], axis=1).astype(np.float64)


def fit(X: np.ndarray, y: np.ndarray):
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    pipe = make_pipeline(StandardScaler(),
                         LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto"))
    return pipe.fit(_features(X), y)


def score(clf, X: np.ndarray) -> np.ndarray:
    return clf.predict_proba(_features(X))
