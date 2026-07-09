"""Per-subject signal re-centering (`recenter_signals`) — the time-series analog of `recenter_covariances`.

Equivalence classes: (a) each group's mean covariance is whitened to ~identity (the invariant the transfer
fix relies on); (b) shape/dtype preserved; (c) groups are handled independently (a second subject's mixing
doesn't leak into the first).
"""
import numpy as np
from pyriemann.utils.mean import mean_riemann

from core.features.eeg.covariance import recenter_signals


def _mixed(rng, mixing, n, t):
    ch = mixing.shape[0]
    return np.einsum("ij,njt->nit", mixing, rng.standard_normal((n, ch, t)))


def test_each_group_whitened_to_identity():
    rng = np.random.default_rng(0)
    n, ch, t = 30, 6, 80
    x1 = _mixed(rng, rng.standard_normal((ch, ch)), n, t)
    x2 = _mixed(rng, rng.standard_normal((ch, ch)), n, t)   # different displacement
    x = np.concatenate([x1, x2])
    groups = np.array([0] * n + [1] * n)

    y = recenter_signals(x, groups)

    assert y.shape == x.shape and y.dtype == np.float32
    for g in (0, 1):
        cov = np.einsum("nct,ndt->ncd", y[groups == g], y[groups == g]) / t
        m = mean_riemann(cov)
        np.testing.assert_allclose(m, np.eye(ch), atol=1e-4)   # domain mean -> identity


def test_single_group_is_still_whitened():
    rng = np.random.default_rng(1)
    x = _mixed(rng, rng.standard_normal((5, 5)), 25, 60)
    y = recenter_signals(x, np.zeros(25))
    cov = np.einsum("nct,ndt->ncd", y, y) / y.shape[2]
    np.testing.assert_allclose(mean_riemann(cov), np.eye(5), atol=1e-4)
