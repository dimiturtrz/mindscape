"""Transforms — sliding-window crops + standardizers."""
import numpy as np

from neuroscan.models import transforms as T


def test_crops_shape_and_trial_index():
    X = np.arange(4 * 2 * 10).reshape(4, 2, 10).astype(np.float32)
    Xc, tidx = T.crops(X, crop_len=6, n_crops=3)
    assert Xc.shape == (12, 2, 6)                 # 4 trials x 3 crops
    # each trial index appears exactly n_crops times
    counts = np.bincount(tidx)
    assert list(counts) == [3, 3, 3, 3]


def test_crops_full_length_is_identity_window():
    X = np.random.RandomState(0).randn(3, 2, 8).astype(np.float32)
    Xc, _ = T.crops(X, crop_len=8, n_crops=1)
    assert np.array_equal(Xc, X)


def test_crops_windows_match_source_slices():
    # value-level: row k*N+i must be trial i's window at starts[k], and tidx must agree
    X = np.arange(4 * 2 * 10).reshape(4, 2, 10).astype(np.float32)
    N, cl, nc = 4, 6, 3
    Xc, tidx = T.crops(X, cl, nc)
    starts = np.linspace(0, 10 - cl, nc).round().astype(int)
    for k, s in enumerate(starts):
        for i in range(N):
            assert np.array_equal(Xc[k * N + i], X[i, :, s:s + cl])
            assert tidx[k * N + i] == i


def test_zscore_normalizes_per_channel():
    X = np.random.RandomState(1).randn(20, 3, 50).astype(np.float32) * 5 + 2
    z = T.ZScore().fit(X)(X)
    # per-channel mean ~0, std ~1 across epochs+time
    assert np.allclose(z.mean(axis=(0, 2)), 0, atol=1e-3)
    assert np.allclose(z.std(axis=(0, 2)), 1, atol=1e-2)


def test_identity_passthrough():
    X = np.random.RandomState(2).randn(4, 2, 6).astype(np.float32)
    assert np.array_equal(T.Identity().fit(X)(X), X)


def test_standardizer_registry():
    assert isinstance(T.standardizer("zscore"), T.ZScore)
    assert isinstance(T.standardizer("none"), T.Identity)
    assert isinstance(T.standardizer("unknown"), T.ZScore)   # fallback


def test_ems_preserves_shape():
    import pytest
    pytest.importorskip("braindecode")
    X = np.random.RandomState(3).randn(5, 4, 200).astype(np.float32)
    out = T.ExpMovingStd()(X)
    assert out.shape == X.shape
