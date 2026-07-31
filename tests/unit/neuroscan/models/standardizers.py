"""Standardizers — per-channel z-score / EMS / identity, the `fit`/`__call__` scaling stage."""
import numpy as np

from neuroscan.models.standardizers import ExpMovingStd, Identity, ZScore

np.random.seed(0)


def test_zscore_normalizes_per_channel():
    X = np.random.RandomState(1).randn(20, 3, 50).astype(np.float32) * 5 + 2
    z = ZScore().fit(X)(X)
    # per-channel mean ~0, std ~1 across epochs+time
    assert np.allclose(z.mean(axis=(0, 2)), 0, atol=1e-3)
    assert np.allclose(z.std(axis=(0, 2)), 1, atol=1e-2)


def test_identity_passthrough():
    X = np.random.RandomState(2).randn(4, 2, 6).astype(np.float32)
    assert np.array_equal(Identity().fit(X)(X), X)


def test_ems_preserves_shape():
    import pytest
    pytest.importorskip("braindecode")
    X = np.random.RandomState(3).randn(5, 4, 200).astype(np.float32)
    out = ExpMovingStd()(X)
    assert out.shape == X.shape
