"""Covariance-space transforms — the EEG geometric feature + the manifold alignment ops the Riemannian
methods and cross-subject transfer sit on. Each trial's channel covariance is a point on the SPD manifold;
these produce it (`time_delay_embed` before the covariance) and move whole clouds around on the manifold
(`recenter_covariances`, `scale_to_identity` — the RPA transfer steps)."""
from __future__ import annotations

import numpy as np
from jaxtyping import Float, Int
from pyriemann.utils.base import invsqrtm, powm
from pyriemann.utils.distance import distance_riemann
from pyriemann.utils.mean import mean_riemann


class Covariance:
    """Covariance-space transforms — the EEG geometric feature + the manifold alignment ops (free helpers
    folded in as staticmethods, public names kept)."""

    @staticmethod
    def time_delay_embed(X: Float[np.ndarray, "n ch t"], order: int, lag: int) -> Float[np.ndarray, "n cho tl"]:
        """Augmented Covariance Method embedding: stack `order` lagged copies of each trial so the covariance
        becomes `[ch*order, ch*order]` and encodes temporal dynamics, not just spatial structure.
        `X [n, ch, t] -> [n, ch*order, t-(order-1)*lag]` — folds *time* into the SPD matrix without the
        instability of a short sliding window (Carrara & Papadopoulo)."""
        n, ch, t = X.shape
        L = t - (order - 1) * lag
        if L <= 0:
            raise ValueError(f"order*lag too large for trial length {t} (order={order}, lag={lag})")
        return np.concatenate([X[:, :, k * lag:k * lag + L] for k in range(order)], axis=1)

    @staticmethod
    def recenter_covariances(C: Float[np.ndarray, "n ch ch"]) -> Float[np.ndarray, "n ch ch"]:
        """Congruence-transport one domain's covariances to the identity: `C -> M^{-1/2} C M^{-1/2}`, where M is
        the domain's Riemannian (Fréchet) mean. Removes the per-domain LOCATION shift on the SPD manifold
        (Zanini et al. 2018) while preserving the relative class geometry — the manifold version of whitening,
        applied per subject to kill the between-subject nuisance. Unsupervised → deployment-friendly."""
        C = np.asarray(C, dtype=np.float64)
        W = invsqrtm(mean_riemann(C))
        return np.einsum("ij,njk,kl->nil", W, C, W)

    @staticmethod
    def recenter_signals(X: Float[np.ndarray, "n ch t"], groups: Int[np.ndarray, "n"], *, max_ref: int = 512,
                         shrinkage: float = 0.0) -> Float[np.ndarray, "n ch t"]:
        """Whiten raw multichannel *signals* per domain by that domain's mean covariance: `X -> M^{-1/2} X`, with
        `M` the Riemannian mean of the domain's per-trial channel covariances. The time-series analog of
        `recenter_covariances` — it removes the per-subject spatial displacement from the SIGNALS an encoder
        consumes (not just from covariance features), so a contrastive EEG->image encoder sees each subject in a
        common spatial frame. Unsupervised (no labels) → applies to a held-out subject at deployment.
        `X [n, ch, t]`, `groups [n]` (subject id per trial) -> `[n, ch, t]`.

        `M` is estimated from an evenly-strided sample of ≤ `max_ref` trials per domain — a channel-covariance
        mean needs a few hundred trials, not all of them, and the iterative Fréchet mean over the full ~10⁴-trial
        set is the cost bottleneck. The whitening `W` is still applied to every trial.

        `shrinkage` ∈ [0,1] pulls `M` toward its scaled identity (`(1-s)·M + s·(trM/ch)·I`) before inverting, so
        the whitening stops amplifying the low-variance (noise) directions — full whitening on an ill-conditioned
        `M` boosts noise channels and injects dispersion the encoder can't use. `0` = exact whitening."""
        X = np.asarray(X, dtype=np.float64)
        groups = np.asarray(groups)
        ch = X.shape[1]
        out = np.empty_like(X)
        for g in np.unique(groups):
            idx = groups == g
            Xg = X[idx]
            ref = Xg if len(Xg) <= max_ref else Xg[np.linspace(0, len(Xg) - 1, max_ref).astype(int)]
            C = np.einsum("nct,ndt->ncd", ref, ref) / ref.shape[2]     # channel covariance of the sample [m,ch,ch]
            M = mean_riemann(C)
            if shrinkage > 0:
                M = (1 - shrinkage) * M + shrinkage * (np.trace(M) / ch) * np.eye(ch)
            out[idx] = np.einsum("ij,njt->nit", invsqrtm(M), Xg)
        return out.astype(np.float32)

    @staticmethod
    def scale_to_identity(C: Float[np.ndarray, "n ch ch"], target_disp: float = 1.0) -> Float[np.ndarray, "n ch ch"]:
        """Normalize dispersion (RPA step 2): after re-centering to the identity, stretch each covariance so the
        mean squared Riemannian distance to I equals `target_disp` — matches the domains' *spread*, not just
        their location. `C -> C**p` with `p = sqrt(target_disp / current_dispersion)`."""
        eye = np.eye(C.shape[-1])
        disp = float(np.mean([distance_riemann(c, eye) ** 2 for c in C])) + 1e-12
        p = np.sqrt(target_disp / disp)
        return np.stack([powm(c, p) for c in C])
