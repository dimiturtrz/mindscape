"""Equivalence-class tests for the MVNN normalizer (bd b40j, ported to the Normalizer interface cx7x).

Partition: (1) the defining property — after whitening, the pooled within-condition residual covariance is
~identity; (2) per-subject independence — two groups with different noise colour are each whitened by their
OWN covariance; (3) shape/dtype preserved; (4) it errors without the groups/conditions it needs to fit."""
import numpy as np
import pytest

from core.normalization.mvnn import Mvnn
from core.normalization.normalization import NormContext


def _colored(rng, mix, n, t):
    ch = mix.shape[0]
    white = rng.standard_normal((n, ch, t))
    return np.einsum("ij,njt->nit", mix, white)


def _residual_cov(X, conditions):
    res = np.empty_like(X)
    for c in np.unique(conditions):
        idx = conditions == c
        res[idx] = X[idx] - X[idx].mean(axis=0, keepdims=True)
    pooled = res.transpose(0, 2, 1).reshape(-1, X.shape[1])
    return np.cov(pooled, rowvar=False)


def test_whitens_within_condition_noise_to_identity():
    rng = np.random.default_rng(0)
    mix = np.array([[1.5, 0.0, 0.0], [0.9, 1.1, 0.0], [0.3, 0.6, 0.8]])
    conditions = np.repeat(np.arange(40), 8)
    signal = rng.standard_normal((40, 3, 1))[np.repeat(np.arange(40), 8)]
    X = signal + _colored(rng, mix, len(conditions), 16)
    ctx = NormContext(groups=np.zeros(len(conditions), dtype=int), conditions=conditions)
    cov = _residual_cov(Mvnn().apply(X, ctx), conditions)
    assert np.allclose(cov, np.eye(3), atol=0.15)


def test_each_group_whitened_by_own_covariance():
    rng = np.random.default_rng(1)
    conditions = np.tile(np.repeat(np.arange(30), 8), 2)
    n_half = 30 * 8
    groups = np.array([0] * n_half + [1] * n_half)
    mix_a = np.array([[1.0, 0.0], [0.5, 1.0]])
    mix_b = np.array([[2.2, 0.0], [-1.3, 0.7]])
    X = np.concatenate([_colored(rng, mix_a, n_half, 16), _colored(rng, mix_b, n_half, 16)])
    white = Mvnn().apply(X, NormContext(groups=groups, conditions=conditions))
    for g in (0, 1):
        cov = _residual_cov(white[groups == g], conditions[groups == g])
        assert np.allclose(cov, np.eye(2), atol=0.2)


def test_preserves_shape_and_dtype():
    rng = np.random.default_rng(2)
    X = rng.standard_normal((24, 5, 10))
    conditions = np.repeat(np.arange(6), 4)
    white = Mvnn().apply(X, NormContext(groups=np.zeros(24, dtype=int), conditions=conditions))
    assert white.shape == (24, 5, 10)
    assert white.dtype == np.float32


def test_errors_without_groups_or_conditions():
    """MVNN's within-condition noise is undefined without the subject + image structure."""
    X = np.random.default_rng(3).standard_normal((4, 3, 5))
    with pytest.raises(ValueError, match="ctx.groups"):
        Mvnn().apply(X, NormContext())
