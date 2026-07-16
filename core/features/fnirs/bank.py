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
from jaxtyping import Float
from scipy.stats import kurtosis, skew

from core.features.fnirs.amplitude import Amplitude


class DescriptorBank:
    """fNIRS descriptor bank — the wide per-channel temporal-feature family set (free helpers folded in as
    staticmethods, public names kept). `FNIRS_FEATURE_FNS` maps family name -> the staticmethod."""

    @classmethod
    def _slope(cls, X: np.ndarray, tc: np.ndarray, tc_ss: float) -> np.ndarray:
        """OLS slope of each channel over the (centred) time axis — the response's trend."""
        return (X * tc).sum(axis=2) / tc_ss

    @classmethod
    def f_mean(cls, X):    return X.mean(axis=2)                                    # response amplitude
    @classmethod
    def f_var(cls, X):     return X.var(axis=2)                                     # response variability
    @classmethod
    def f_min(cls, X):     return X.min(axis=2)
    @classmethod
    def f_max(cls, X):     return X.max(axis=2)
    @classmethod
    def f_range(cls, X):   return X.max(axis=2) - X.min(axis=2)
    @classmethod
    def f_auc(cls, X):     return np.trapezoid(X, axis=2)                 # area under the response (trapezoid)
    @classmethod
    def f_final(cls, X):   return X[:, :, -max(1, X.shape[2] // 10):].mean(axis=2)  # plateau (last ~10% of window)

    @classmethod
    def f_peak(cls, X):
        idx = np.abs(X).argmax(axis=2)                                          # signed extreme (max |value|)
        return np.take_along_axis(X, idx[:, :, None], axis=2)[:, :, 0]

    @classmethod
    def f_time_to_peak(cls, X):
        return np.abs(X).argmax(axis=2).astype(np.float32) / X.shape[2]         # latency of the extreme, in [0,1)

    @classmethod
    def f_skew(cls, X):
        return np.nan_to_num(skew(X, axis=2)).astype(np.float32)               # asymmetry (flat channel -> 0)

    @classmethod
    def f_kurtosis(cls, X):
        return np.nan_to_num(kurtosis(X, axis=2)).astype(np.float32)          # peakedness (flat channel -> 0)

    @classmethod
    def f_zero_crossings(cls, X):
        return (np.diff(np.signbit(X), axis=2).sum(axis=2)).astype(np.float32)  # # sign changes over time

    @classmethod
    def f_slope(cls, X):
        tc, tc_ss = Amplitude.time_axis(X.shape[2])
        return cls._slope(X, tc, tc_ss)

    @classmethod
    def f_early_slope(cls, X):
        h = X.shape[2] // 2                                                     # first-half rise
        tc, tc_ss = Amplitude.time_axis(h)
        return cls._slope(X[:, :, :h], tc, tc_ss)

    @classmethod
    def f_late_slope(cls, X):
        h = X.shape[2] // 2                                                     # second-half plateau/decay
        Xl = X[:, :, h:]
        tc, tc_ss = Amplitude.time_axis(Xl.shape[2])
        return cls._slope(Xl, tc, tc_ss)

    @classmethod
    def family_names(cls) -> list[str]:
        """The descriptor families, in column order."""
        return list(FNIRS_FEATURE_FNS)

    @classmethod
    def extract_bank(cls, X: Float[np.ndarray, "n ch t"]) -> tuple[np.ndarray, np.ndarray]:
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


# name -> fn(X[n,ch,t]) -> [n,ch]. Order fixed (dict is insertion-ordered) so columns are deterministic.
FNIRS_FEATURE_FNS = {
    "mean": DescriptorBank.f_mean, "slope": DescriptorBank.f_slope, "peak": DescriptorBank.f_peak,
    "variance": DescriptorBank.f_var, "skew": DescriptorBank.f_skew, "kurtosis": DescriptorBank.f_kurtosis,
    "auc": DescriptorBank.f_auc, "time_to_peak": DescriptorBank.f_time_to_peak, "min": DescriptorBank.f_min,
    "max": DescriptorBank.f_max, "range": DescriptorBank.f_range, "final": DescriptorBank.f_final,
    "early_slope": DescriptorBank.f_early_slope, "late_slope": DescriptorBank.f_late_slope,
    "zero_crossings": DescriptorBank.f_zero_crossings,
}


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
