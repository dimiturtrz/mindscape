"""core.features.fnirs.chromophore — CBSI, the two-wavelength neural estimate.

Unit-level algebraic properties of `cbsi_neural` (distinct from the end-to-end ground-truth recovery in
test_coupling.py): a pure COMMON-MODE signal (HbO == HbR, systemic/motion) cancels to ~0, and a perfectly
ANTI-CORRELATED equal-amplitude pair (the idealized neural response) is kept. Shape is preserved [n,ch,t].
"""
import numpy as np

from core.features.fnirs.chromophore import Chromophore


def _sig(n=2, ch=3, t=200, seed=0):
    return np.random.default_rng(seed).standard_normal((n, ch, t))


def test_cbsi_cancels_common_mode():
    """HbO == HbR is pure systemic/motion (common mode) — CBSI(HbO,HbO) = 0.5(HbO - 1·HbO) = 0."""
    s = _sig()
    assert np.allclose(Chromophore.cbsi_neural(s, s), 0.0, atol=1e-6)   # ~1e-10 residual from the std-ratio epsilon


def test_cbsi_keeps_anticorrelated_neural():
    """Equal-std, perfectly anti-correlated HbO/HbR (idealized activation): a = std/std = 1, so
    0.5(HbO - HbR) = 0.5(HbO - (-HbO)) = HbO — the neural part passes through unattenuated."""
    hbo = _sig(seed=1)
    hbr = -hbo
    assert np.allclose(Chromophore.cbsi_neural(hbo, hbr), hbo, atol=1e-6)


def test_cbsi_preserves_shape():
    s = _sig(n=4, ch=5, t=120)
    assert Chromophore.cbsi_neural(s, s * 0.5).shape == (4, 5, 120)
