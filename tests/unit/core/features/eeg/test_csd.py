"""EEG surface-Laplacian / CSD (`core.features.eeg.csd.CSD`).

CSD is a reference-free spatial high-pass: it must preserve the `[n, ch, t]` shape and, being a Laplacian,
suppress the spatially-uniform (common) component far more than a spatially-varying one.
"""
import numpy as np
import pytest

mne = pytest.importorskip("mne")

from core.features.eeg.csd import CSD   # noqa: E402

_CH = ["Fz", "Cz", "Pz", "Oz", "C3", "C4", "F3", "F4", "P3", "P4"]


def test_csd_preserves_shape_and_is_finite():
    rng = np.random.default_rng(0)
    Xe = rng.standard_normal((3, len(_CH), 500))
    out = CSD.csd_transform(Xe, _CH, 100.0)
    assert out.shape == Xe.shape
    assert np.isfinite(out).all()
    assert not np.allclose(out, Xe)                      # a spatial high-pass actually changed the data


def test_csd_suppresses_spatially_uniform_signal():
    """A signal identical on every channel is pure common-mode: the surface Laplacian must map it toward ~0,
    far smaller than the CSD of a spatially-varying signal of the same per-channel amplitude."""
    rng = np.random.default_rng(1)
    base = rng.standard_normal((2, 1, 500))
    uniform = np.repeat(base, len(_CH), axis=1)                         # same waveform on all channels
    varying = base * rng.standard_normal((2, len(_CH), 1))              # per-channel spatial variation
    e_uniform = np.abs(CSD.csd_transform(uniform, _CH, 100.0)).mean()
    e_varying = np.abs(CSD.csd_transform(varying, _CH, 100.0)).mean()
    assert e_uniform < 0.1 * e_varying
