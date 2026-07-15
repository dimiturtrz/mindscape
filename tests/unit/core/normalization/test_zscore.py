"""Equivalence-class tests for the z-score normalizer (bd cx7x/qsae).

Partition: (1) each channel becomes ~zero-mean, unit-variance over time; (2) a flat (zero-variance) channel
does not blow up (eps guard); (3) shape/dtype preserved. Stateless — no fit needed."""
import numpy as np

from core.normalization.zscore import ZScore


def test_per_channel_zero_mean_unit_variance():
    rng = np.random.default_rng(0)
    X = (rng.standard_normal((8, 5, 200)) * 3.0 + 7.0).astype(np.float32)   # off-mean, non-unit scale
    out = ZScore().apply(X)
    assert np.allclose(out.mean(axis=2), 0.0, atol=1e-5)
    assert np.allclose(out.std(axis=2), 1.0, atol=1e-3)


def test_flat_channel_does_not_blow_up():
    out = ZScore().apply(np.ones((2, 3, 10), dtype=np.float32))            # zero variance
    assert np.isfinite(out).all()


def test_preserves_shape_and_dtype():
    out = ZScore().apply(np.random.default_rng(1).standard_normal((4, 6, 12)).astype(np.float32))
    assert out.shape == (4, 6, 12)
    assert out.dtype == np.float32
