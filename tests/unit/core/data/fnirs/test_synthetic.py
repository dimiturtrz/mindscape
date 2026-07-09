"""Physics-forward synthetic fNIRS generator (bd 7jn) — HRF shape + the forward's neural/systemic structure."""
import numpy as np

from core.data.fnirs.synthetic import SynthConfig, double_gamma_hrf, synthesize_paired


def test_double_gamma_has_positive_lobe_and_undershoot():
    hrf = double_gamma_hrf(5.0)
    assert hrf.max() == 1.0                                       # peak-normalized
    assert hrf.min() < 0                                          # the undershoot (distinguishes from single-gamma)
    assert np.argmax(hrf) < np.argmin(hrf)                        # positive lobe precedes undershoot


def test_synthesize_shapes_and_neural_anticorrelation():
    """With systemic + noise off, HbO and HbR are pure anti-correlated neural response (what CBSI keeps)."""
    cfg = SynthConfig(systemic_amp=0.0, noise_std=0.0)
    rng = np.random.default_rng(0)
    drive = (rng.standard_normal((4, 500)) > 1.0).astype(float)   # sparse activations
    hbo, hbr = synthesize_paired(drive, 5.0, cfg, seed=0)
    assert hbo.shape == hbr.shape == (4, 500) and hbo.dtype == np.float32
    # neural-only: HbR = -hbr_ratio * HbO response -> strongly anti-correlated
    assert np.corrcoef(hbo[0], hbr[0])[0, 1] < -0.9


def test_systemic_is_common_mode():
    """With neural off, HbO and HbR carry (near-)common-mode systemic -> positively correlated (what CBSI cancels)."""
    cfg = SynthConfig(noise_std=0.0)
    hbo, hbr = synthesize_paired(np.zeros((3, 800)), 5.0, cfg, seed=1)
    assert np.corrcoef(hbo[0], hbr[0])[0, 1] > 0.9
