"""Physics-forward synthetic fNIRS generator (bd 7jn) — HRF shape + the forward's neural/systemic structure."""
import numpy as np

from core.data.fnirs.synthetic import Synthetic, SynthConfig


def test_double_gamma_hrf():
    hrf = Synthetic.double_gamma_hrf(5.0)
    assert hrf.max() == 1.0                                       # peak-normalized
    assert hrf.min() < 0                                          # the undershoot (distinguishes from single-gamma)
    assert np.argmax(hrf) < np.argmin(hrf)                        # positive lobe precedes undershoot


def test_synthesize_paired():
    """With systemic + noise off, HbO and HbR are pure anti-correlated neural response (what CBSI keeps)."""
    cfg = SynthConfig(systemic_amp=0.0, noise_std=0.0)
    rng = np.random.default_rng(0)
    drive = (rng.standard_normal((4, 500)) > 1.0).astype(float)   # sparse activations
    hbo, hbr = Synthetic.synthesize_paired(drive, 5.0, cfg, seed=0)
    assert hbo.shape == hbr.shape == (4, 500) and hbo.dtype == np.float32
    # neural-only: HbR = -hbr_ratio * HbO response -> strongly anti-correlated
    assert np.corrcoef(hbo[0], hbr[0])[0, 1] < -0.9


def test_systemic():
    """Systemic oscillations are common-mode and non-zero."""
    rng = np.random.default_rng(42)
    cfg = SynthConfig()
    sys = Synthetic.systemic(n=3, length=400, fs=10.0, cfg=cfg, rng=rng)
    assert sys.shape == (3, 400)
    assert sys.dtype == np.float64 or sys.dtype == np.float32
    assert np.sum(np.abs(sys)) > 0.0  # systemic contains actual oscillations
    assert np.isfinite(sys).all()  # no NaN or inf
