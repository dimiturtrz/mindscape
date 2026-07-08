"""Covariance-space transforms — the EEG geometric feature + the manifold alignment ops the Riemannian
methods and cross-subject transfer sit on. Each trial's channel covariance is a point on the SPD manifold;
these produce it (`time_delay_embed` before the covariance) and move whole clouds around on the manifold
(`recenter_covariances`, `scale_to_identity` — the RPA transfer steps)."""
from __future__ import annotations

import numpy as np
from pyriemann.utils.base import invsqrtm, powm
from pyriemann.utils.distance import distance_riemann
from pyriemann.utils.mean import mean_riemann


def time_delay_embed(X: np.ndarray, order: int, lag: int) -> np.ndarray:
    """Augmented Covariance Method embedding: stack `order` lagged copies of each trial so the covariance
    becomes `[ch*order, ch*order]` and encodes temporal dynamics, not just spatial structure.
    `X [n, ch, t] -> [n, ch*order, t-(order-1)*lag]` — folds *time* into the SPD matrix without the
    instability of a short sliding window (Carrara & Papadopoulo)."""
    n, ch, t = X.shape
    L = t - (order - 1) * lag
    if L <= 0:
        raise ValueError(f"order*lag too large for trial length {t} (order={order}, lag={lag})")
    return np.concatenate([X[:, :, k * lag:k * lag + L] for k in range(order)], axis=1)


def recenter_covariances(C: np.ndarray) -> np.ndarray:
    """Congruence-transport one domain's covariances to the identity: `C -> M^{-1/2} C M^{-1/2}`, where M is
    the domain's Riemannian (Fréchet) mean. Removes the per-domain LOCATION shift on the SPD manifold
    (Zanini et al. 2018) while preserving the relative class geometry — the manifold version of whitening,
    applied per subject to kill the between-subject nuisance. Unsupervised → deployment-friendly."""
    C = np.asarray(C, dtype=np.float64)
    W = invsqrtm(mean_riemann(C))
    return np.einsum("ij,njk,kl->nil", W, C, W)


def scale_to_identity(C: np.ndarray, target_disp: float = 1.0) -> np.ndarray:
    """Normalize dispersion (RPA step 2): after re-centering to the identity, stretch each covariance so the
    mean squared Riemannian distance to I equals `target_disp` — matches the domains' *spread*, not just
    their location. `C -> C**p` with `p = sqrt(target_disp / current_dispersion)`."""
    eye = np.eye(C.shape[-1])
    disp = float(np.mean([distance_riemann(c, eye) ** 2 for c in C])) + 1e-12
    p = np.sqrt(target_disp / disp)
    return np.stack([powm(c, p) for c in C])
