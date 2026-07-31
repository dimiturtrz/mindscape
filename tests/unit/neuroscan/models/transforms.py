"""Transforms — sliding-window crops + the standardizer-by-name factory."""
import numpy as np

from neuroscan.models import transforms as T
from neuroscan.models.standardizers import Identity, ZScore

np.random.seed(0)


def test_crops_shape_and_trial_index():
    X = np.arange(4 * 2 * 10).reshape(4, 2, 10).astype(np.float32)
    Xc, tidx = T.Transforms.crops(X, crop_len=6, n_crops=3)
    assert Xc.shape == (12, 2, 6)                 # 4 trials x 3 crops
    # each trial index appears exactly n_crops times
    counts = np.bincount(tidx)
    assert list(counts) == [3, 3, 3, 3]


def test_crops_full_length_is_identity_window():
    X = np.random.RandomState(0).randn(3, 2, 8).astype(np.float32)
    Xc, _ = T.Transforms.crops(X, crop_len=8, n_crops=1)
    assert np.array_equal(Xc, X)


def test_crops_windows_match_source_slices():
    # value-level: row k*N+i must be trial i's window at starts[k], and tidx must agree
    X = np.arange(4 * 2 * 10).reshape(4, 2, 10).astype(np.float32)
    N, cl, nc = 4, 6, 3
    Xc, tidx = T.Transforms.crops(X, cl, nc)
    starts = np.linspace(0, 10 - cl, nc).round().astype(int)
    for k, s in enumerate(starts):
        for i in range(N):
            assert np.array_equal(Xc[k * N + i], X[i, :, s:s + cl])
            assert tidx[k * N + i] == i


def test_standardizer_registry():
    assert isinstance(T.Transforms.standardizer("zscore"), ZScore)
    assert isinstance(T.Transforms.standardizer("none"), Identity)
    assert isinstance(T.Transforms.standardizer("unknown"), ZScore)   # fallback
