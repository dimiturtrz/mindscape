"""Equivalence-class tests for the MVNN normalizer (bd b40j, ported to the fit/apply interface cx7x).

Partition: (1) the defining property — after fit+apply, the pooled within-condition residual covariance is
~identity; (2) per-subject independence — two groups with different noise colour are each whitened by their
OWN covariance; (3) shape/dtype preserved; (4) apply before fit errors (the whiteners aren't estimated yet)."""
import numpy as np
import pytest

from core.normalization.mvnn import Mvnn


def _colored(rng, mix, n, t):
    white = rng.standard_normal((n, mix.shape[0], t))
    return np.einsum("ij,njt->nit", mix, white)


def _residual_cov(X, conditions):
    res = np.empty_like(X)
    for c in np.unique(conditions):
        idx = conditions == c
        res[idx] = X[idx] - X[idx].mean(axis=0, keepdims=True)
    return np.cov(res.transpose(0, 2, 1).reshape(-1, X.shape[1]), rowvar=False)


def test_whitens_within_condition_noise_to_identity():
    rng = np.random.default_rng(0)
    mix = np.array([[1.5, 0.0, 0.0], [0.9, 1.1, 0.0], [0.3, 0.6, 0.8]])
    conditions = np.repeat(np.arange(40), 8)
    signal = rng.standard_normal((40, 3, 1))[np.repeat(np.arange(40), 8)]
    X = signal + _colored(rng, mix, len(conditions), 16)
    mvnn = Mvnn(np.zeros(len(conditions), dtype=int), conditions)
    cov = _residual_cov(mvnn.fit(X).apply(X), conditions)
    assert np.allclose(cov, np.eye(3), atol=0.15)


def test_each_group_whitened_by_own_covariance():
    rng = np.random.default_rng(1)
    conditions = np.tile(np.repeat(np.arange(30), 8), 2)
    n_half = 30 * 8
    groups = np.array([0] * n_half + [1] * n_half)
    mix_a = np.array([[1.0, 0.0], [0.5, 1.0]])
    mix_b = np.array([[2.2, 0.0], [-1.3, 0.7]])
    X = np.concatenate([_colored(rng, mix_a, n_half, 16), _colored(rng, mix_b, n_half, 16)])
    white = Mvnn(groups, conditions).fit(X).apply(X)
    for g in (0, 1):
        cov = _residual_cov(white[groups == g], conditions[groups == g])
        assert np.allclose(cov, np.eye(2), atol=0.2)


def test_preserves_shape_and_dtype():
    rng = np.random.default_rng(2)
    X = rng.standard_normal((24, 5, 10))
    conditions = np.repeat(np.arange(6), 4)
    white = Mvnn(np.zeros(24, dtype=int), conditions).fit(X).apply(X)
    assert white.shape == (24, 5, 10)
    assert white.dtype == np.float32


def test_apply_before_fit_errors():
    """The whiteners must be estimated first — apply without fit is a usage error."""
    X = np.random.default_rng(3).standard_normal((4, 3, 5))
    conditions = np.repeat(np.arange(2), 2)
    with pytest.raises(RuntimeError, match="before fit"):
        Mvnn(np.zeros(4, dtype=int), conditions).apply(X)
