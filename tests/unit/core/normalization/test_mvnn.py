"""Equivalence-class tests for the MVNN normalizer (bd b40j; fit-on-train / apply-anywhere, bd u9sv).

Partition: (1) the defining property — a single train-fit whitener drives the pooled within-condition residual
covariance to ~identity; (2) fit-on-train / apply-to-HELD-OUT — the whitener is fit on one set and applied to
different rows (grouping-free), never fitting the eval set; (3) shape/dtype preserved; (4) apply before fit
errors."""
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


def test_whitens_pooled_within_condition_noise_to_identity():
    rng = np.random.default_rng(0)
    mix = np.array([[1.5, 0.0, 0.0], [0.9, 1.1, 0.0], [0.3, 0.6, 0.8]])
    conditions = np.repeat(np.arange(40), 8)
    signal = rng.standard_normal((40, 3, 1))[np.repeat(np.arange(40), 8)]
    X = signal + _colored(rng, mix, len(conditions), 16)
    white = Mvnn(np.zeros(len(conditions), dtype=int), conditions).fit(X).apply(X)
    assert np.allclose(_residual_cov(white, conditions), np.eye(3), atol=0.15)


def test_fit_on_train_apply_to_heldout_subject():
    """The whitener is fit on TRAIN subjects and applied to a HELD-OUT subject's rows — grouping-free apply,
    so the eval set is never fit. Whitening a held-out subject with the train colour still reduces its
    channel correlation toward the train frame."""
    rng = np.random.default_rng(1)
    mix = np.array([[1.4, 0.0], [1.0, 0.9]])
    cond = np.repeat(np.arange(30), 8)
    train = _colored(rng, mix, len(cond), 16)                    # subjects 0..3 share the colour
    groups = np.repeat(np.arange(4), len(cond) // 4)
    mvnn = Mvnn(groups, cond).fit(train)
    heldout = _colored(rng, mix, 60, 16)                         # a NEW subject, same-family noise, unseen at fit
    out = mvnn.apply(heldout)
    off_before = abs(np.corrcoef(heldout.transpose(1, 0, 2).reshape(2, -1))[0, 1])
    off_after = abs(np.corrcoef(out.transpose(1, 0, 2).reshape(2, -1))[0, 1])
    assert out.shape == heldout.shape and np.isfinite(out).all()
    assert off_after < off_before                                # train whitener decorrelates the held-out set


def test_preserves_shape_and_dtype():
    rng = np.random.default_rng(2)
    conditions = np.repeat(np.arange(6), 4)
    white = Mvnn(np.zeros(24, dtype=int), conditions).fit(rng.standard_normal((24, 5, 10))).apply(
        rng.standard_normal((24, 5, 10)))
    assert white.shape == (24, 5, 10) and white.dtype == np.float32


def test_apply_before_fit_errors():
    X = np.random.default_rng(3).standard_normal((4, 3, 5))
    with pytest.raises(RuntimeError, match="before fit"):
        Mvnn(np.zeros(4, dtype=int), np.repeat(np.arange(2), 2)).apply(X)
