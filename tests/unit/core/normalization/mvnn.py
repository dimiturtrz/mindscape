"""Equivalence-class tests for the per-subject MVNN normalizer (bd — per-subject calibration).

Partition: (1) the defining property — each subject's own whitener drives THAT subject's pooled within-condition
residual covariance to ~identity; (2) per-subject, not pooled — two subjects with different noise colour get
different whiteners, each whitening its own set to identity; (3) calibration→held-out-trials — a subject's
whitener fit on its calibration rows decorrelates DIFFERENT rows of the same subject (apply selects by group);
(4) shape/dtype preserved; (5) apply before fit errors; (6) apply to an uncalibrated subject errors."""
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


def test_whitens_each_subjects_within_condition_noise_to_identity():
    rng = np.random.default_rng(0)
    mix = np.array([[1.5, 0.0, 0.0], [0.9, 1.1, 0.0], [0.3, 0.6, 0.8]])
    conditions = np.repeat(np.arange(40), 8)
    signal = rng.standard_normal((40, 3, 1))[np.repeat(np.arange(40), 8)]
    X = signal + _colored(rng, mix, len(conditions), 16)
    groups = np.zeros(len(conditions), dtype=int)
    white = Mvnn(groups, conditions).fit(X).apply(X, groups)
    assert np.allclose(_residual_cov(white, conditions), np.eye(3), atol=0.15)


def test_per_subject_whiteners_differ_and_each_whitens_its_own():
    """Two subjects with DIFFERENT noise colour get DIFFERENT whiteners; applying each subject's own whitener
    drives both subjects' within-condition covariance to identity (a single pooled whitener could not)."""
    rng = np.random.default_rng(1)
    mix_a = np.array([[1.6, 0.0], [1.2, 0.7]])
    mix_b = np.array([[0.7, 0.0], [-1.1, 1.4]])
    cond1 = np.repeat(np.arange(30), 8)
    conditions = np.concatenate([cond1, cond1])
    groups = np.concatenate([np.zeros(len(cond1), int), np.ones(len(cond1), int)])
    X = np.concatenate([_colored(rng, mix_a, len(cond1), 16), _colored(rng, mix_b, len(cond1), 16)])
    mvnn = Mvnn(groups, conditions).fit(X)
    assert not np.allclose(mvnn._whiteners[0], mvnn._whiteners[1], atol=0.2)   # per-subject, not one pooled
    white = mvnn.apply(X, groups)
    for g in (0, 1):
        assert np.allclose(_residual_cov(white[groups == g], conditions[groups == g]), np.eye(2), atol=0.2)


def test_calibration_whitener_applies_to_heldout_trials_of_same_subject():
    """A subject's whitener is fit on its CALIBRATION trials, then applied to DIFFERENT trials of that same
    subject (the deployment path: enroll once, whiten new incoming trials) — reduces their channel correlation."""
    rng = np.random.default_rng(2)
    mix = np.array([[1.4, 0.0], [1.0, 0.9]])
    cond = np.repeat(np.arange(30), 8)
    calib = _colored(rng, mix, len(cond), 16)
    mvnn = Mvnn(np.full(len(cond), 7, int), cond).fit(calib)     # subject 7 calibrated on `calib`
    fresh = _colored(rng, mix, 60, 16)                           # same subject, NEW trials, unseen at fit
    groups = np.full(60, 7, int)
    out = mvnn.apply(fresh, groups)
    off_before = abs(np.corrcoef(fresh.transpose(1, 0, 2).reshape(2, -1))[0, 1])
    off_after = abs(np.corrcoef(out.transpose(1, 0, 2).reshape(2, -1))[0, 1])
    assert out.shape == fresh.shape and np.isfinite(out).all()
    assert off_after < off_before


def test_preserves_shape_and_dtype():
    rng = np.random.default_rng(3)
    conditions = np.repeat(np.arange(6), 4)
    groups = np.zeros(24, dtype=int)
    white = Mvnn(groups, conditions).fit(rng.standard_normal((24, 5, 10))).apply(
        rng.standard_normal((24, 5, 10)), groups)
    assert white.shape == (24, 5, 10) and white.dtype == np.float32


def test_apply_before_fit_errors():
    X = np.random.default_rng(4).standard_normal((4, 3, 5))
    with pytest.raises(RuntimeError, match="before fit"):
        Mvnn(np.zeros(4, dtype=int), np.repeat(np.arange(2), 2)).apply(X, np.zeros(4, dtype=int))


def test_apply_to_uncalibrated_subject_errors():
    rng = np.random.default_rng(5)
    cond = np.repeat(np.arange(4), 4)
    mvnn = Mvnn(np.zeros(16, int), cond).fit(rng.standard_normal((16, 3, 8)))   # only subject 0 calibrated
    with pytest.raises(RuntimeError, match="no whitener"):
        mvnn.apply(rng.standard_normal((4, 3, 8)), np.full(4, 9, int))          # subject 9 never seen
