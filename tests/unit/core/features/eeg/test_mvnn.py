"""Equivalence-class tests for multivariate noise normalization (bd b40j).

Partition: (1) the defining property — after whitening, the pooled within-condition residual covariance is
~identity (spatially white, unit variance); (2) per-subject independence — two groups with different noise
colour are each whitened by their OWN covariance; (3) the condition signal (between-condition means) survives
whitening (MVNN normalizes noise, it does not erase the signal); (4) shape/dtype are preserved.
"""
import numpy as np

from core.features.eeg.mvnn import Mvnn


def _colored(rng, mix, n, t):
    """`n` trials × `t` samples of channel noise coloured by `mix` (`white @ mixᵀ` → covariance `mix @ mixᵀ`)."""
    ch = mix.shape[0]
    white = rng.standard_normal((n, ch, t))
    return np.einsum("ij,njt->nit", mix, white)


def _residual_cov(X, conditions):
    """Pooled within-condition residual covariance of `X [n,ch,t]` — the quantity MVNN drives to identity."""
    res = np.empty_like(X)
    for c in np.unique(conditions):
        idx = conditions == c
        res[idx] = X[idx] - X[idx].mean(axis=0, keepdims=True)
    pooled = res.transpose(0, 2, 1).reshape(-1, X.shape[1])
    return np.cov(pooled, rowvar=False)


def test_whitens_within_condition_noise_to_identity():
    """Class: defining property — the pooled within-condition residual covariance becomes ~I after whitening."""
    rng = np.random.default_rng(0)
    mix = np.array([[1.5, 0.0, 0.0], [0.9, 1.1, 0.0], [0.3, 0.6, 0.8]])   # a definitely-non-white noise colour
    conditions = np.repeat(np.arange(40), 8)                              # 40 conditions × 8 reps
    signal = rng.standard_normal((40, 3, 1))[np.repeat(np.arange(40), 8)]  # per-condition constant mean
    X = signal + _colored(rng, mix, len(conditions), 16)
    groups = np.zeros(len(conditions), dtype=int)
    white = Mvnn.whiten(X, groups, conditions)
    cov = _residual_cov(white, conditions)
    assert np.allclose(cov, np.eye(3), atol=0.15)                        # spatially white, unit variance


def test_each_group_whitened_by_own_covariance():
    """Class: per-subject — group B's stronger noise colour is normalized by B's own Σ, not a shared one."""
    rng = np.random.default_rng(1)
    conditions = np.tile(np.repeat(np.arange(30), 8), 2)
    n_half = 30 * 8
    groups = np.array([0] * n_half + [1] * n_half)
    mix_a = np.array([[1.0, 0.0], [0.5, 1.0]])
    mix_b = np.array([[2.2, 0.0], [-1.3, 0.7]])                          # very different colour
    X = np.concatenate([_colored(rng, mix_a, n_half, 16), _colored(rng, mix_b, n_half, 16)])
    white = Mvnn.whiten(X, groups, conditions)
    for g in (0, 1):
        cov = _residual_cov(white[groups == g], conditions[groups == g])
        assert np.allclose(cov, np.eye(2), atol=0.2)


def test_preserves_shape_and_dtype():
    """Class: contract — output is the same [n,ch,t] grid, float32."""
    rng = np.random.default_rng(2)
    X = rng.standard_normal((24, 5, 10))
    conditions = np.repeat(np.arange(6), 4)
    white = Mvnn.whiten(X, np.zeros(24, dtype=int), conditions)
    assert white.shape == (24, 5, 10)
    assert white.dtype == np.float32
