"""fNIRS descriptor bank — the full per-channel temporal-feature family set, for the weighted-feature
importance search (Optuna). Where `amplitude.py` gives the canonical mean/slope/peak triple, this is the
*wide* bank: every stat that might carry the hemodynamic signal, each as one named block `[n, ch]` so the
search can weight families independently and read which ones matter.

The block signal is `[n, ch, t]` (Shin n-back: ch=72 = 36 HbO then 36 HbR, t=220 ≈ 22 s @ 10 Hz). Each
descriptor collapses the time axis to one value per channel — that *is* the temporal aggregation; the
spatial (72-channel) and fold aggregation live downstream (the study keeps all channels + averages folds).

Two pieces:
  - `extract_bank(X)` -> `(F[n, ch*K], fam[ch*K])`: all K families concatenated, with a column→family map
    so a per-family weight can be broadcast to that family's channels. Deterministic column order.
  - `WeightedFamilyScaler`: fit per-feature standardisation on TRAIN, then apply a per-family weight — the
    weight MUST come after scaling (before, StandardScaler divides it back out and it does nothing).
"""
from __future__ import annotations

import numpy as np

from core.features.fnirs.amplitude import _time_axis


def _slope(X: np.ndarray, tc: np.ndarray, tc_ss: float) -> np.ndarray:
    """OLS slope of each channel over the (centred) time axis — the response's trend."""
    return (X * tc).sum(axis=2) / tc_ss


def _f_mean(X):    return X.mean(axis=2)                                    # response amplitude
def _f_var(X):     return X.var(axis=2)                                     # response variability
def _f_min(X):     return X.min(axis=2)
def _f_max(X):     return X.max(axis=2)
def _f_range(X):   return X.max(axis=2) - X.min(axis=2)
def _f_auc(X):     return np.trapezoid(X, axis=2)                           # area under the response (trapezoid)
def _f_final(X):   return X[:, :, -max(1, X.shape[2] // 10):].mean(axis=2)  # plateau (last ~10% of window)


def _f_peak(X):
    idx = np.abs(X).argmax(axis=2)                                          # signed extreme (max |value|)
    return np.take_along_axis(X, idx[:, :, None], axis=2)[:, :, 0]


def _f_time_to_peak(X):
    return np.abs(X).argmax(axis=2).astype(np.float32) / X.shape[2]         # latency of the extreme, in [0,1)


def _f_skew(X):
    from scipy.stats import skew
    return np.nan_to_num(skew(X, axis=2)).astype(np.float32)               # asymmetry (flat channel -> 0)


def _f_kurtosis(X):
    from scipy.stats import kurtosis
    return np.nan_to_num(kurtosis(X, axis=2)).astype(np.float32)          # peakedness (flat channel -> 0)


def _f_zero_crossings(X):
    return (np.diff(np.signbit(X), axis=2).sum(axis=2)).astype(np.float32)  # # sign changes over time


def _f_slope(X):
    tc, tc_ss = _time_axis(X.shape[2])
    return _slope(X, tc, tc_ss)


def _f_early_slope(X):
    h = X.shape[2] // 2                                                     # first-half rise
    tc, tc_ss = _time_axis(h)
    return _slope(X[:, :, :h], tc, tc_ss)


def _f_late_slope(X):
    h = X.shape[2] // 2                                                     # second-half plateau/decay
    Xl = X[:, :, h:]
    tc, tc_ss = _time_axis(Xl.shape[2])
    return _slope(Xl, tc, tc_ss)


# name -> fn(X[n,ch,t]) -> [n,ch]. Order fixed (dict is insertion-ordered) so columns are deterministic.
FNIRS_FEATURE_FNS = {
    "mean": _f_mean, "slope": _f_slope, "peak": _f_peak, "variance": _f_var,
    "skew": _f_skew, "kurtosis": _f_kurtosis, "auc": _f_auc, "time_to_peak": _f_time_to_peak,
    "min": _f_min, "max": _f_max, "range": _f_range, "final": _f_final,
    "early_slope": _f_early_slope, "late_slope": _f_late_slope, "zero_crossings": _f_zero_crossings,
}


def family_names() -> list[str]:
    """The descriptor families, in column order."""
    return list(FNIRS_FEATURE_FNS)


def extract_bank(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Extract every family and concatenate: `X[n,ch,t]` -> `(F[n, ch*K], fam[ch*K])`. `fam[j]` is the
    family name of column j, so a per-family weight maps to its channels via `fam == name`. f64 for a
    stable scaler downstream (the study standardises then weights)."""
    X = np.asarray(X, dtype=np.float64)
    ch = X.shape[1]
    blocks, fam = [], []
    for name, fn in FNIRS_FEATURE_FNS.items():
        blocks.append(np.asarray(fn(X), dtype=np.float64))                 # [n, ch]
        fam.extend([name] * ch)
    return np.concatenate(blocks, axis=1), np.array(fam)


class WeightedFamilyScaler:
    """Per-feature standardisation (fit on TRAIN) followed by a per-family weight. sklearn transformer
    contract (`fit`/`transform`), so it sits in a Pipeline and standardises on train only — no leakage.

    The weight is applied AFTER standardisation on purpose: a weight applied before is exactly divided back
    out by the per-feature std, so it would have no effect. `weights` maps family name -> w (missing = 1.0);
    w≈0 effectively drops the family. `fam` is the column→family map from `extract_bank`."""

    def __init__(self, fam: np.ndarray, weights: dict[str, float]):
        self.fam = fam
        self.weights = weights

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=np.float64)
        self.mean_ = X.mean(axis=0)
        self.std_ = X.std(axis=0) + 1e-8                                   # guard zero-variance columns
        self.w_ = np.array([float(self.weights.get(f, 1.0)) for f in self.fam])
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=np.float64)
        return ((X - self.mean_) / self.std_) * self.w_                    # standardise, THEN weight
