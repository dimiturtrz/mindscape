"""Joint EEG+fNIRS forward generator (bd shd) — the PURE forwards, tested without the fsaverage template.
Equivalence classes partition each forward's behaviour: latent support, sensitivity distance-monotonicity,
lead-field linearity, and the HbO/HbR anti-correlation the shared latent must produce."""
import numpy as np

from core.data.fnirs.synthetic import SynthConfig
from core.features.fusion.joint_forward import Grid, JointConfig, JointForward


def test_plant_latent_support_matches_active():
    """Class: latent structure — exactly n_active parcels nonzero per trial, and they ARE the reported active
    indices; every other parcel is silent (zero)."""
    cfg = JointConfig(n_active=2)
    lat = JointForward.plant_latent(Grid(n_trials=4, n_parcels=6, n_times=50, sfreq=10.0), cfg=cfg, seed=1)
    assert lat.source.shape == (4, 6, 50)
    assert lat.active.shape == (4, 2)
    for i in range(4):
        nonzero = np.where(lat.source[i].any(axis=1))[0]
        assert set(nonzero.tolist()) == set(lat.active[i].tolist())


def test_sensitivity_decreases_with_distance():
    """Class: geometric sensitivity — a channel weights a NEAR parcel more than a FAR one, and all weights are
    positive (a Gaussian of the cortical distance)."""
    parcels = np.array([[0.0, 0.0, 0.05], [0.0, 0.0, -0.05], [0.08, 0.0, 0.0]], dtype=np.float32)
    channel = np.array([[0.0, 0.0, 0.09]], dtype=np.float32)          # scalp above parcel 0
    a = JointForward.sensitivity(parcels, channel, JointConfig())
    assert a.shape == (1, 3)
    assert (a > 0).all()
    assert a[0, 0] > a[0, 1] and a[0, 0] > a[0, 2]                    # nearest parcel is most sensitive


def test_eeg_from_source_is_linear_lead_field():
    """Class: EEG forward — with noise off, sensors are exactly the lead-field mixing of the source; a zero
    lead-field column (silent-to-sensors parcel) contributes nothing."""
    rng = np.random.default_rng(0)
    source = rng.standard_normal((2, 3, 40)).astype(np.float32)
    lead = rng.standard_normal((5, 3)).astype(np.float32)
    lead[:, 2] = 0.0                                                  # parcel 2 invisible to sensors
    eeg = JointForward.eeg_from_source(source, lead, JointConfig(eeg_noise=0.0))
    assert eeg.shape == (2, 5, 40)
    expected = np.einsum("cp,npt->nct", lead, source)
    np.testing.assert_allclose(eeg, expected, atol=1e-5)
    # dropping the silent parcel leaves the sensors unchanged
    np.testing.assert_allclose(eeg, np.einsum("cp,npt->nct", lead[:, :2], source[:, :2]), atol=1e-5)


def test_fnirs_hbo_hbr_anticorrelated_on_neural():
    """Class: fNIRS forward — with systemic + measurement noise off, HbR is exactly the anti-correlated
    -hbr_ratio·HbO neural response (what CBSI must keep); a silent source yields a flat-zero response."""
    scfg = SynthConfig(systemic_amp=0.0, noise_std=0.0, hbr_ratio=0.4)
    cfg = JointConfig(synth=scfg)
    lat = JointForward.plant_latent(Grid(n_trials=3, n_parcels=4, n_times=200, sfreq=10.0), cfg=cfg, seed=2)
    sens = np.abs(np.random.default_rng(3).standard_normal((2, 4)).astype(np.float32))
    hbo, hbr = JointForward.fnirs_from_source(lat.source, sens, fs=10.0, cfg=cfg, seed=0)
    assert hbo.shape == (3, 2, 200) and hbr.shape == (3, 2, 200)
    np.testing.assert_allclose(hbr, -scfg.hbr_ratio * hbo, rtol=1e-4, atol=1e-5)
    # a silent latent -> no hemodynamic response at all (systemic/noise are off)
    silent = np.zeros((1, 4, 200), dtype=np.float32)
    hbo0, _ = JointForward.fnirs_from_source(silent, sens, fs=10.0, cfg=cfg, seed=0)
    np.testing.assert_allclose(hbo0, 0.0, atol=1e-6)
