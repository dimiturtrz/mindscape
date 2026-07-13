"""Ground-truth validation of estimate_coupling + CBSI (bd uqw) against the INDEPENDENT synthetic forward.

Non-circular by construction: the synthetic hemodynamics are a double-gamma HRF (core.data.fnirs.synthetic),
a different shape than estimate_coupling's single-gamma fit. So recovering the planted lag/sign here tests the
estimator, not its own assumptions. Two claims: (1) CBSI separates the anti-correlated neural response from the
common-mode systemic; (2) estimate_coupling recovers a physiologically-correct lag + the positive coupling sign.
"""
import numpy as np
from scipy.signal import fftconvolve

from core.data.fnirs.synthetic import SynthConfig, double_gamma_hrf, synthesize_paired
from core.features.fnirs.chromophore import Chromophore
from core.features.fusion.coupling import Coupling

_FS = 5.0


def _drive(rng, length=2000, n_blocks=25):
    d = np.zeros((1, length))
    for s in rng.integers(0, length - 100, n_blocks):
        d[0, s:s + rng.integers(20, 60)] += 1.0
    return d


def test_cbsi_recovers_neural_from_systemic():
    """CBSI(HbO,HbR) tracks the true neural response despite the injected common-mode systemic."""
    cfg = SynthConfig()
    rng = np.random.default_rng(0)
    drive = _drive(rng)
    hbo, hbr = synthesize_paired(drive, _FS, cfg, seed=1)
    cbsi = Chromophore.cbsi_neural(hbo[:, None, :], hbr[:, None, :])[:, 0, :]
    true_resp = fftconvolve(drive, double_gamma_hrf(_FS, cfg)[None, :], axes=1)[:, :drive.shape[1]]
    assert np.corrcoef(cbsi[0], true_resp[0])[0, 1] > 0.9        # neural recovered, systemic cancelled


def test_estimate_coupling_recovers_lag_and_sign():
    """Across seeds, the recovered lag sits in the HRF's physiological delay window and the coupling sign is
    positive (a neural drive raises HbO). Tolerances are wide — the single-gamma estimator can't match the
    double-gamma peak exactly, which is the point (independent test)."""
    cfg = SynthConfig()
    lags = []
    for seed in range(3):
        rng = np.random.default_rng(seed)
        drive = _drive(rng)
        hbo, hbr = synthesize_paired(drive, _FS, cfg, seed=seed + 10)
        cbsi = Chromophore.cbsi_neural(hbo[:, None, :], hbr[:, None, :])[:, 0, :]
        lag, _decay, beta = Coupling.estimate_coupling(drive, cbsi, _FS)
        assert beta > 0                                          # neural -> HbO up: positive coupling
        assert 2.0 <= lag <= 9.0                                 # physiological, near the 6 s HRF peak
        lags.append(lag)
    assert np.std(lags) < 2.0                                    # stable across seeds
