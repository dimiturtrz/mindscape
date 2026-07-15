"""Multivariate noise normalization (MVNN) — the official THINGS-EEG2 / Guggenmos-2018 preprocessing whitening.

Within an image condition the *signal* is fixed, so the trial-to-trial variance IS the noise. MVNN estimates
that noise covariance `Σ` (per subject), then whitens every trial by `Σ^{-1/2}` so the channels the decoder
sees carry spatially-white, unit-variance noise — the Mahalanobis frame in which a Euclidean decoder is
optimal. This is the step Gifford (2022) applied to THINGS-EEG2 and a documented reason the NICE baseline
works.

FIT-ON-TRAIN, APPLY-ANYWHERE (deployment-clean, bd u9sv). `fit(X)` estimates ONE whitener from the TRAINING
epochs — the per-trial structure it needs (subject id `groups`, image id `conditions`) is its **own
constructor** argument, aligned with that fit data, not part of the general interface. Each training subject's
trials are residualized against THAT subject's condition means (so between-subject signal differences are not
mistaken for noise), the residuals are POOLED across subjects+conditions, and a single Ledoit-Wolf shrinkage
fit yields `Σ^{-1/2}`. `apply(X)` then multiplies that one whitener into any epochs — train OR a held-out test
subject — so nothing is ever fit on the eval set (the earlier per-test-subject fit was a transductive leak).
Pooling is what makes `Σ` well-conditioned: THINGS-EEG2 train has only 4 reps/concept, so any single
condition's 63×63 covariance is rank-deficient.
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
    """Multivariate noise normalization (Guggenmos 2018) — a data-fitted `Normalizer`. `groups` (subject per
    trial) + `conditions` (image per trial) are its constructor state, aligned with the TRAIN epochs it is fit
    on; `apply` uses the single fitted whitener and needs no grouping, so it works on held-out test epochs."""

    def __init__(self, groups: Int[np.ndarray, "n"], conditions: Int[np.ndarray, "n"]):
        self.groups = np.asarray(groups)
        self.conditions = np.asarray(conditions)
        self._whitener: np.ndarray | None = None

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
        """Estimate ONE `Σ^{-1/2}` from the TRAIN epochs `X` (rows align with the constructor grouping):
        residualize each subject's trials against THAT subject's condition means, pool all residuals over
        subjects+trials+time, Ledoit-Wolf fit. Per-subject residualizing keeps between-subject signal out of
        the noise estimate; pooling makes it well-conditioned."""
        X = np.asarray(X, dtype=np.float64)
        residuals = [Mvnn._condition_residual(X[self.groups == g], self.conditions[self.groups == g])
                     for g in np.unique(self.groups)]
        pooled = np.concatenate(residuals).transpose(0, 2, 1).reshape(-1, X.shape[1])   # [Σtrials*time, ch]
        self._whitener = Mvnn._noise_whitener(pooled)
        return self

    def apply(self, X: Float[np.ndarray, "n ch t"]) -> Float[np.ndarray, "n ch t"]:
        """Whiten every trial by the single train-fit `Σ^{-1/2}` — grouping-free, so it applies unchanged to a
        held-out test subject (fit never saw the eval set)."""
        if self._whitener is None:
            raise RuntimeError("Mvnn.apply before fit — call fit(X_train) to estimate the whitener first")
        return np.einsum("ij,njt->nit", self._whitener, np.asarray(X, dtype=np.float64)).astype(np.float32)
