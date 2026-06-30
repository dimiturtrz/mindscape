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


def fit(X: np.ndarray, y: np.ndarray, method: str = "ts", estimator: str = "oas"):
    """Fit a Riemannian classifier. `method`: "ts" (tangent space + LR) or "mdm". `estimator`: the
    covariance shrinkage estimator (OAS keeps the matrices SPD on short/noisy trials)."""
    from pyriemann.classification import MDM
    from pyriemann.estimation import Covariances
    from pyriemann.tangentspace import TangentSpace
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline

    cov = Covariances(estimator=estimator)
    if method == "mdm":
        pipe = make_pipeline(cov, MDM(metric="riemann"))
    elif method == "ts":
        pipe = make_pipeline(
            cov,
            TangentSpace(metric="riemann"),
            LogisticRegression(max_iter=500, C=1.0),
        )
    else:
        raise ValueError(f"unknown riemann method {method!r}; use 'ts' or 'mdm'")
    pipe.fit(X.astype(np.float64), y)
    return pipe


def score(pipe, X: np.ndarray) -> np.ndarray:
    return pipe.predict_proba(X.astype(np.float64))
