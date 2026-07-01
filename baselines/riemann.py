"""Riemannian geometry baseline — the modern BCI workhorse, parallel to CSP+LDA.

Instead of learning spatial filters (CSP) or raw-waveform features (the deep nets), this treats each
trial's **channel covariance matrix** as a point on the Riemannian manifold of symmetric-positive-definite
matrices and classifies by that curved geometry. Two variants share the harness contract:

  - "ts"  (default): tangent-space mapping (project covariances to the tangent plane at their geometric
            mean -> Euclidean vectors) -> logistic regression. The strong, SOTA-competitive classical
            method; what MOABB benchmarks treat as the reference and what wins real BCI competitions.
  - "mdm": Minimum Distance to Mean — assign a trial to the class whose geometric-mean covariance is
            closest by Riemannian distance. Parameter-free, no real training.

Covariance is the feature (no filter learning, no backprop); robust on tiny per-subject sets and the
natural base for cross-subject transfer (manifold re-centering). Interface = the harness contract:
`fit(X, y) -> clf`, `score(clf, X) -> probs[n, C]`.
"""
from __future__ import annotations

import numpy as np


def _augment(X: np.ndarray, order: int, lag: int) -> np.ndarray:
    """Time-delay embedding (Augmented Covariance Method): stack `order` lagged copies of each trial so
    the covariance becomes [ch*order, ch*order] and encodes temporal dynamics, not just spatial structure.
    X [n, ch, t] -> [n, ch*order, t-(order-1)*lag]. This is the SOTA-flavoured Riemannian trick — it folds
    *time* into the SPD matrix without the instability of a short sliding window (Carrara & Papadopoulo)."""
    n, ch, t = X.shape
    L = t - (order - 1) * lag
    if L <= 0:
        raise ValueError(f"order*lag too large for trial length {t} (order={order}, lag={lag})")
    return np.concatenate([X[:, :, k * lag:k * lag + L] for k in range(order)], axis=1)


def recenter_covariances(C: np.ndarray) -> np.ndarray:
    """Congruence-transport one domain's covariances to the identity: C -> M^{-1/2} C M^{-1/2}, where M
    is the domain's Riemannian (Fréchet) mean. Removes the per-domain LOCATION shift on the SPD manifold
    (Zanini et al. 2018 recentering) while preserving the relative class geometry — the manifold version
    of the M2 whitening (C^{-1/2}), applied per subject to kill the between-subject nuisance.

    Cross-subject EEG fails because each subject's whole covariance cloud sits at a different point on the
    manifold (head/skin/electrode/noise) — a domain shift, not a difference in the shared ERD contrast.
    Recentering every subject (train AND the unlabeled target) to a common origin aligns the clouds so the
    shared discriminative structure lines up. Unsupervised for the target -> deployment-friendly."""
    from pyriemann.utils.base import invsqrtm
    from pyriemann.utils.mean import mean_riemann

    C = np.asarray(C, dtype=np.float64)
    W = invsqrtm(mean_riemann(C))
    return np.einsum("ij,njk,kl->nil", W, C, W)


def fit(X: np.ndarray, y: np.ndarray, method: str = "ts", estimator: str = "oas",
        order: int = 4, lag: int = 8):
    """Fit a Riemannian classifier. `method`:
      - "ts"  : tangent space + LR (strong classical reference)
      - "mdm" : Minimum Distance to Mean (parameter-free)
      - "acm" : Augmented Covariance Method — time-delay-embedded covariance (`order` lagged copies at
                stride `lag`) + tangent space + LR. Folds temporal dynamics into the SPD matrix.
    `estimator`: covariance shrinkage (OAS keeps matrices SPD on short/noisy trials)."""
    from pyriemann.classification import MDM
    from pyriemann.estimation import Covariances
    from pyriemann.tangentspace import TangentSpace
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import FunctionTransformer

    cov = Covariances(estimator=estimator)
    ts_lr = [TangentSpace(metric="riemann"), LogisticRegression(max_iter=500, C=1.0)]
    aug = FunctionTransformer(_augment, kw_args={"order": order, "lag": lag}, validate=False)
    builders = {                                              # only the chosen pipeline is built
        "mdm": lambda: make_pipeline(cov, MDM(metric="riemann")),
        "ts": lambda: make_pipeline(cov, *ts_lr),
        "acm": lambda: make_pipeline(aug, cov, *ts_lr),
    }
    if method not in builders:
        raise ValueError(f"unknown riemann method {method!r}; use one of {sorted(builders)}")
    pipe = builders[method]()
    pipe.fit(X.astype(np.float64), y)
    return pipe


def score(pipe, X: np.ndarray) -> np.ndarray:
    return pipe.predict_proba(X.astype(np.float64))
