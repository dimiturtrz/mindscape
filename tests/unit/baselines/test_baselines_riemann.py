"""Riemannian baseline smoke: covariance-structured classes must decode through both variants.

Builds trials whose CLASS is encoded purely in channel covariance (different mixing per class, same
marginal scale) — the signal CSP/Riemann are built to read — and checks tangent-space LR and MDM both
beat chance through the `fit/score` harness contract.
"""
import numpy as np
import pytest

from baselines import riemann


def _cov_dataset(n_per_class=40, n_ch=4, n_t=128, seed=0):
    rng = np.random.default_rng(seed)
    # two source-mixing matrices -> two distinct covariance structures, same per-channel variance budget
    A0 = rng.normal(size=(n_ch, n_ch))
    A1 = rng.normal(size=(n_ch, n_ch))
    X, y = [], []
    for cls, A in ((0, A0), (1, A1)):
        for _ in range(n_per_class):
            s = rng.normal(size=(n_ch, n_t))
            X.append(A @ s)
            y.append(cls)
    return np.asarray(X, dtype=np.float64), np.asarray(y)


@pytest.mark.parametrize("method", ["ts", "mdm", "acm"])
def test_riemann_decodes_covariance_signal(method):
    X, y = _cov_dataset(seed=1)
    clf = riemann.fit(X, y, method=method)
    proba = riemann.score(clf, X)
    assert proba.shape == (len(X), 2)
    assert np.allclose(proba.sum(1), 1.0, atol=1e-5)
    acc = (proba.argmax(1) == y).mean()
    assert acc > 0.7        # covariance-encoded classes must decode well


def test_unknown_method_raises():
    X, y = _cov_dataset(n_per_class=8, seed=2)
    with pytest.raises(ValueError):
        riemann.fit(X, y, method="nope")


def test_recenter_makes_domain_mean_identity():
    # recentering a domain by its own Riemannian mean must move that mean to the identity
    from pyriemann.utils.mean import mean_riemann
    rng = np.random.default_rng(0)
    shift = (lambda A: A @ A.T + 4 * np.eye(4))(rng.normal(size=(4, 4)))   # an SPD location far from I
    C = np.stack([shift @ (B @ B.T / 8) @ shift                            # covs centered away from I
                  for B in rng.normal(size=(30, 4, 8))])
    M = mean_riemann(riemann.recenter_covariances(C))
    assert np.allclose(M, np.eye(4), atol=1e-4)
