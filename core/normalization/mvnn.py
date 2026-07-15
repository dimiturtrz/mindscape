"""Multivariate noise normalization (MVNN) — the official THINGS-EEG2 / Guggenmos-2018 preprocessing whitening.

Within an image condition the *signal* is fixed, so the trial-to-trial variance IS the noise. MVNN estimates
that noise covariance `Σ` (per subject), then whitens every trial by `Σ^{-1/2}` so the channels the decoder
sees carry spatially-white, unit-variance noise — the Mahalanobis frame in which a Euclidean decoder is
optimal. This is the step Gifford (2022) applied to THINGS-EEG2 and a documented reason the NICE baseline
works.

As a `Normalizer`, the per-trial structure it needs — subject id (`groups`) and image id (`conditions`) — is
its **own constructor** argument, not part of the general interface: `fit(X)` estimates one whitener per
subject from that subject's within-condition residuals; `apply(X)` applies them. The estimate pools those
residuals across ALL conditions before a single Ledoit-Wolf shrinkage fit — THINGS-EEG2 train has only 4
reps/concept, so any one condition's 63×63 covariance is rank-deficient; averaging the residual structure over
thousands of conditions (plus shrinkage) is what makes `Σ` well-conditioned. `Σ` is per subject (a subject's
own trials), so it applies unsupervised to a held-out subject.
"""
from __future__ import annotations

import numpy as np
from jaxtyping import Float, Int
from pyriemann.utils.base import invsqrtm
from sklearn.covariance import LedoitWolf

from core.normalization.normalization import Normalizer

_MAX_COV_SAMPLES = 100_000   # a 63×63 covariance is well-determined by ~10⁵ residual samples (»63²); more only
                             # adds compute. The whitener still applies to every trial — only the ESTIMATE is capped.


class Mvnn(Normalizer):
    """Per-subject multivariate noise normalization (Guggenmos 2018) — a data-fitted `Normalizer`; `groups`
    (subject per trial) + `conditions` (image per trial) are its constructor state, aligned with the epochs it
    is fit/applied on."""

    def __init__(self, groups: Int[np.ndarray, "n"], conditions: Int[np.ndarray, "n"]):
        self.groups = np.asarray(groups)
        self.conditions = np.asarray(conditions)
        self._whiteners: dict[int, np.ndarray] = {}

    @staticmethod
    def _condition_residual(Xg: Float[np.ndarray, "m ch t"], conditions: Int[np.ndarray, "m"]
                            ) -> Float[np.ndarray, "m ch t"]:
        """`trial − its condition mean` for every trial (vectorized over conditions) — the within-condition
        noise, since the signal is fixed within a condition."""
        codes = np.unique(conditions, return_inverse=True)[1]
        sums = np.zeros((codes.max() + 1, *Xg.shape[1:]), dtype=Xg.dtype)
        np.add.at(sums, codes, Xg)
        means = sums / np.bincount(codes)[:, None, None]
        return Xg - means[codes]

    @staticmethod
    def _noise_whitener(residuals: Float[np.ndarray, "m ch"]) -> Float[np.ndarray, "ch ch"]:
        """`Σ^{-1/2}` from pooled within-condition residuals via Ledoit-Wolf shrinkage (robust for 63 channels
        on few-trials-per-condition data), strided down to `_MAX_COV_SAMPLES` for the fit."""
        if len(residuals) > _MAX_COV_SAMPLES:
            residuals = residuals[np.linspace(0, len(residuals) - 1, _MAX_COV_SAMPLES).astype(int)]
        sigma = LedoitWolf(assume_centered=True).fit(residuals).covariance_
        return invsqrtm(sigma)

    def fit(self, X: Float[np.ndarray, "n ch t"]) -> Mvnn:
        """Estimate `Σ_g^{-1/2}` per subject `g`: residualize each of its trials against its condition mean,
        pool the residuals over trials+time, Ledoit-Wolf fit. `X` rows align with the constructor `groups`."""
        X = np.asarray(X, dtype=np.float64)
        for g in np.unique(self.groups):
            idx = self.groups == g
            residual = Mvnn._condition_residual(X[idx], self.conditions[idx])
            pooled = residual.transpose(0, 2, 1).reshape(-1, X.shape[1])          # [trials*time, ch]
            self._whiteners[g] = Mvnn._noise_whitener(pooled)
        return self

    def apply(self, X: Float[np.ndarray, "n ch t"]) -> Float[np.ndarray, "n ch t"]:
        """Whiten each trial by its subject's `Σ^{-1/2}` (`X` rows align with the constructor `groups`)."""
        if not self._whiteners:
            raise RuntimeError("Mvnn.apply before fit — call fit(X) to estimate the per-subject whiteners first")
        X = np.asarray(X, dtype=np.float64)
        out = np.empty_like(X)
        for g in np.unique(self.groups):
            idx = self.groups == g
            out[idx] = np.einsum("ij,njt->nit", self._whiteners[g], X[idx])
        return out.astype(np.float32)
