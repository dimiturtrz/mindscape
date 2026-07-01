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

from baselines.base import Baseline


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


class _RiemannBaseline(Baseline):
    """Covariance estimate -> a subclass-defined manifold classifier. Shares the float64 cast pyriemann
    needs; subclasses supply the sklearn pipeline via `_build`. The fitted pipeline lives on `self.pipe_`."""

    def __init__(self, estimator: str = "oas"):
        self.estimator = estimator                       # covariance shrinkage (OAS keeps short trials SPD)

    def _build(self):                                    # -> sklearn Pipeline
        raise NotImplementedError

    def fit(self, X, y):
        self.pipe_ = self._build()
        self.pipe_.fit(np.asarray(X, dtype=np.float64), y)
        return self

    def score(self, X):
        return self.pipe_.predict_proba(np.asarray(X, dtype=np.float64))


def _cov(estimator: str):
    from pyriemann.estimation import Covariances
    return Covariances(estimator=estimator)


def _tangent_lr():
    from pyriemann.tangentspace import TangentSpace as _TS
    from sklearn.linear_model import LogisticRegression
    return [_TS(metric="riemann"), LogisticRegression(max_iter=500, C=1.0)]


class TangentSpace(_RiemannBaseline):
    """Tangent-space projection at the geometric mean -> logistic regression. The strong classical
    reference (what MOABB benchmarks and what wins real BCI competitions)."""

    def _build(self):
        from sklearn.pipeline import make_pipeline
        return make_pipeline(_cov(self.estimator), *_tangent_lr())


class Mdm(_RiemannBaseline):
    """Minimum Distance to Mean — assign each trial to the nearest class geometric-mean covariance
    (Riemannian distance). Parameter-free, no real training."""

    def _build(self):
        from pyriemann.classification import MDM
        from sklearn.pipeline import make_pipeline
        return make_pipeline(_cov(self.estimator), MDM(metric="riemann"))


class Acm(_RiemannBaseline):
    """Augmented Covariance Method: time-delay-embed (`order` lagged copies at stride `lag`) before the
    covariance, folding temporal dynamics into the SPD matrix, then tangent space + LR."""

    def __init__(self, order: int = 4, lag: int = 8, estimator: str = "oas"):
        super().__init__(estimator)
        self.order, self.lag = order, lag

    def _build(self):
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import FunctionTransformer
        aug = FunctionTransformer(_augment, kw_args={"order": self.order, "lag": self.lag}, validate=False)
        return make_pipeline(aug, _cov(self.estimator), *_tangent_lr())


_METHODS = {"ts": TangentSpace, "mdm": Mdm, "acm": Acm}


def fit(X: np.ndarray, y: np.ndarray, method: str = "ts", estimator: str = "oas",
        order: int = 4, lag: int = 8) -> Baseline:
    """Back-compat shim — build the method object for `method` and fit it. Prefer the classes directly
    (TangentSpace/Mdm/Acm); this keeps the old `fit(X, y, method=...)` call sites working."""
    if method not in _METHODS:
        raise ValueError(f"unknown riemann method {method!r}; use one of {sorted(_METHODS)}")
    kw = {"order": order, "lag": lag, "estimator": estimator} if method == "acm" else {"estimator": estimator}
    return _METHODS[method](**kw).fit(X, y)


def score(clf: Baseline, X: np.ndarray) -> np.ndarray:
    """Back-compat shim — the fitted object scores itself."""
    return clf.score(X)
