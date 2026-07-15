"""Multivariate noise normalization (MVNN) — the official THINGS-EEG2 / Guggenmos-2018 preprocessing whitening.

Within an image condition the *signal* is fixed, so the trial-to-trial variance IS the noise. MVNN estimates
that noise covariance `Σ` (per subject), then whitens every trial by `Σ^{-1/2}` so the channels the decoder
sees carry spatially-white, unit-variance noise — the Mahalanobis frame in which a Euclidean decoder is
optimal. This is the step Gifford (2022) applied to THINGS-EEG2 and a documented reason the NICE baseline
works; as a `Normalizer` it is a data-fitted link — it reads `ctx.groups` (subject) and `ctx.conditions`
(image) to fit one whitener per subject from that subject's own trials.

The estimate pools **within-condition residuals** (each trial minus its condition mean) across ALL conditions
before a single Ledoit-Wolf shrinkage fit: THINGS-EEG2 train carries only 4 reps/concept, so any one
condition's 63×63 covariance is rank-deficient — averaging the residual structure over the thousands of
conditions (plus shrinkage) is what makes `Σ` well-conditioned. `Σ` is per subject (a subject's own trials),
so it applies unsupervised to a held-out subject at deployment.
"""
from __future__ import annotations

import numpy as np
from jaxtyping import Float, Int
from pyriemann.utils.base import invsqrtm
from sklearn.covariance import LedoitWolf

from core.normalization.normalization import Normalizer, NormContext

_MAX_COV_SAMPLES = 100_000   # a 63×63 covariance is well-determined by ~10⁵ residual samples (»63²); more only
                             # adds compute. The whitener still applies to every trial — only the ESTIMATE is capped.


class Mvnn(Normalizer):
    """Per-subject multivariate noise normalization (Guggenmos 2018) — a data-fitted `Normalizer`."""

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
        on few-trials-per-condition data). `residuals [m, ch]` = every trial-minus-its-condition-mean sample
        (pooled over trials and time), strided down to `_MAX_COV_SAMPLES` for the fit."""
        if len(residuals) > _MAX_COV_SAMPLES:
            residuals = residuals[np.linspace(0, len(residuals) - 1, _MAX_COV_SAMPLES).astype(int)]
        sigma = LedoitWolf(assume_centered=True).fit(residuals).covariance_
        return invsqrtm(sigma)

    def apply(self, X: Float[np.ndarray, "n ch t"], ctx: NormContext) -> Float[np.ndarray, "n ch t"]:
        """Whiten each trial by its subject's noise covariance: per group `g`, residualize every trial against
        its condition mean, pool the residuals over trials+time, fit `Σ_g` (Ledoit-Wolf), and apply
        `X -> Σ_g^{-1/2} X` to every trial of that group. Requires `ctx.groups` (subject) + `ctx.conditions`
        (image) — the within-condition noise is undefined without them."""
        if ctx.groups is None or ctx.conditions is None:
            raise ValueError("Mvnn needs ctx.groups (subject) and ctx.conditions (image) to fit the noise covariance")
        X = np.asarray(X, dtype=np.float64)
        groups = np.asarray(ctx.groups)
        conditions = np.asarray(ctx.conditions)
        out = np.empty_like(X)
        for g in np.unique(groups):
            idx = groups == g
            Xg = X[idx]
            residual = Mvnn._condition_residual(Xg, conditions[idx])
            pooled = residual.transpose(0, 2, 1).reshape(-1, Xg.shape[1])          # [trials*time, ch]
            whitener = Mvnn._noise_whitener(pooled)
            out[idx] = np.einsum("ij,njt->nit", whitener, Xg)
        return out.astype(np.float32)
