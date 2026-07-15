"""Equivalence-class tests for the Scale normalizer (bd 7mi4).

Partition: (1) multiplies by the factor; (2) amplitude-PRESERVING — relative channel amplitudes and waveform
shape are unchanged (only the overall scale moves); (3) shape/dtype preserved. Stateless — no fit needed."""
import numpy as np

from core.normalization.scale import Scale


def test_multiplies_by_factor():
    out = Scale(1e4).apply(np.ones((2, 3, 4), dtype=np.float32))
    assert np.allclose(out, 1e4)


def test_amplitude_preserving_ratios_unchanged():
    """Relative channel amplitudes survive — scaling is uniform, unlike a per-channel z-score."""
    rng = np.random.default_rng(0)
    X = (rng.standard_normal((5, 4, 20)) * np.array([1.0, 2.0, 3.0, 4.0])[None, :, None]).astype(np.float32)
    out = Scale(0.01).apply(X)
    np.testing.assert_allclose(out, X * 0.01, rtol=1e-5)
    ratio_in = X.std(axis=(0, 2)) / X.std(axis=(0, 2))[0]
    ratio_out = out.std(axis=(0, 2)) / out.std(axis=(0, 2))[0]
    np.testing.assert_allclose(ratio_in, ratio_out, rtol=1e-4)


def test_preserves_shape_and_dtype():
    out = Scale(2.0).apply(np.random.default_rng(1).standard_normal((3, 6, 8)).astype(np.float32))
    assert out.shape == (3, 6, 8)
    assert out.dtype == np.float32
