"""Multivariate noise normalization (MVNN) — the official THINGS-EEG2 / Guggenmos-2018 preprocessing whitening.

Within an image condition the *signal* is fixed, so the trial-to-trial variance IS the noise. MVNN estimates
that noise covariance `Σ` **per subject**, then whitens every trial by that subject's `Σ^{-1/2}` so the
channels the decoder sees carry spatially-white, unit-variance noise — the Mahalanobis frame in which a
Euclidean decoder is optimal. This is the step Gifford (2022) applied to THINGS-EEG2 and a documented reason
the NICE baseline works. It is intrinsically PER SUBJECT: each person's sensor noise geometry differs, so one
pooled whitener (a train-subject average) mis-fits a held-out subject and throws the benefit away.

FIT-ON-CALIBRATION, APPLY-PER-SUBJECT (leak-free + deployment-real, bd — per-subject calibration). `fit(X)`
estimates ONE `Σ_g^{-1/2}` **per subject** `g` from that subject's CALIBRATION epochs — for a held-out test
subject, its own *training-image* trials, which are disjoint from the scored test-image trials, so the eval
target is never touched (the earlier per-test-subject fit on the *scored* trials, using their image labels,
was the transductive leak). `apply(X, groups)` selects each row's subject whitener and multiplies it in — a
fixed `[ch×ch]` matrix per subject, so it is a single matmul per trial and works UNBATCHED: enroll a subject
once (a calibration session), then every incoming single trial is whitened by that stored matrix. Per subject,
the within-condition residuals are pooled over all its conditions+time and Ledoit-Wolf shrunk (robust for a
63×63 covariance on few-reps-per-condition data).
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
    """Multivariate noise normalization (Guggenmos 2018) — a data-fitted, PER-SUBJECT `Normalizer`. `groups`
    (subject per trial) + `conditions` (image per trial) are its constructor state, aligned with the CALIBRATION
    epochs it is fit on; `apply(X, groups)` picks each row's subject whitener, so it whitens both the training
    subjects and a held-out test subject — each by its own `Σ^{-1/2}`."""

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
        """`Σ^{-1/2}` from within-condition residuals via Ledoit-Wolf shrinkage (robust for 63 channels on
        few-trials-per-condition data), strided down to `_MAX_COV_SAMPLES` for the fit."""
        if len(residuals) > _MAX_COV_SAMPLES:
            residuals = residuals[np.linspace(0, len(residuals) - 1, _MAX_COV_SAMPLES).astype(int)]
        sigma = LedoitWolf(assume_centered=True).fit(residuals).covariance_
        return invsqrtm(sigma)

    def fit(self, X: Float[np.ndarray, "n ch t"]) -> Mvnn:
        """Estimate `Σ_g^{-1/2}` for each subject `g` from ITS calibration epochs (rows aligned with the
        constructor grouping): residualize that subject's trials against its own condition means (within-
        condition noise), pool over its conditions+time, Ledoit-Wolf fit. One whitener per subject, keyed by id.
        Each subject's whitener is fit only on that subject's calibration data — never on another subject's, and
        for the test subject never on the scored test-image trials."""
        X = np.asarray(X, dtype=np.float64)
        for g in np.unique(self.groups):
            mask = self.groups == g
            residuals = Mvnn._condition_residual(X[mask], self.conditions[mask])
            pooled = residuals.transpose(0, 2, 1).reshape(-1, X.shape[1])   # [trials*time, ch]
            self._whiteners[int(g)] = Mvnn._noise_whitener(pooled)
        return self

    def apply(self, X: Float[np.ndarray, "n ch t"],
              groups: Int[np.ndarray, "n"] | None = None) -> Float[np.ndarray, "n ch t"]:
        """Whiten each row by ITS subject's `Σ^{-1/2}`. `groups` = subject id per applied row (defaults to the
        constructor grouping, for whitening the calibration data itself). A subject with no fitted whitener
        errors — it was never calibrated."""
        if not self._whiteners:
            raise RuntimeError("Mvnn.apply before fit — call fit(X_calibration) to estimate the whiteners first")
        groups = self.groups if groups is None else np.asarray(groups)
        X = np.asarray(X, dtype=np.float64)
        out = np.empty_like(X)
        for g in np.unique(groups):
            if int(g) not in self._whiteners:
                raise RuntimeError(f"Mvnn.apply: subject {int(g)} has no whitener — not seen in the calibration fit")
            mask = groups == g
            out[mask] = np.einsum("ij,njt->nit", self._whiteners[int(g)], X[mask])
        return out.astype(np.float32)
