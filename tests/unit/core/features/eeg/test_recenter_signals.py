"""Per-subject signal re-centering (`recenter_signals`) — the time-series analog of `recenter_covariances`.

Equivalence classes: (a) each group's mean covariance is whitened to ~identity (the invariant the transfer
fix relies on); (b) shape/dtype preserved; (c) groups are handled independently (a second subject's mixing
doesn't leak into the first).
"""
import numpy as np
from pyriemann.utils.mean import mean_riemann

from core.features.eeg.covariance import Covariance


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

    y = Covariance.recenter_signals(x, groups)

    assert y.shape == x.shape and y.dtype == np.float32
    for g in (0, 1):
        cov = np.einsum("nct,ndt->ncd", y[groups == g], y[groups == g]) / t
        m = mean_riemann(cov)
        np.testing.assert_allclose(m, np.eye(ch), atol=1e-4)   # domain mean -> identity


def test_single_group_is_still_whitened():
    rng = np.random.default_rng(1)
    x = _mixed(rng, rng.standard_normal((5, 5)), 25, 60)
    y = Covariance.recenter_signals(x, np.zeros(25))
    cov = np.einsum("nct,ndt->ncd", y, y) / y.shape[2]
    np.testing.assert_allclose(mean_riemann(cov), np.eye(5), atol=1e-4)


def test_shrinkage_keeps_output_off_identity_and_finite():
    """shrinkage > 0 must NOT whiten fully to identity (it aligns only the dominant directions, leaving the
    noise floor un-boosted) — so the residual covariance stays off-identity, and finite."""
    rng = np.random.default_rng(2)
    x = _mixed(rng, rng.standard_normal((6, 6)), 40, 90)
    g = np.zeros(40)
    full = np.einsum("nct,ndt->ncd", Covariance.recenter_signals(x, g), Covariance.recenter_signals(x, g)) / 90
    shr = Covariance.recenter_signals(x, g, shrinkage=0.5)
    cov = np.einsum("nct,ndt->ncd", shr, shr) / 90
    assert np.isfinite(shr).all()
    np.testing.assert_allclose(mean_riemann(full), np.eye(6), atol=1e-4)         # full -> identity
    assert np.abs(mean_riemann(cov) - np.eye(6)).mean() > 1e-2                    # shrunk -> not identity
