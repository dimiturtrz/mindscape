"""Unit tests for the brain-camera fusion feature extractor (`core.features.fusion`).

Equivalence-class coverage of the genuinely-new math: CBSI common-mode cancellation (both chromophores), the
data-derived hemodynamic coupling (does `estimate_coupling` recover a PLANTED lag?), and the tensor contract.
"""
import numpy as np

from core.features import fusion as bc


def test_cbsi_cancels_common_mode_keeps_anticorrelated():
    """CBSI = 0.5(HbO − α·HbR): a common-mode (systemic) signal in BOTH chromophores cancels; a truly neural
    (anti-correlated) signal survives. This is the whole point of using both wavelengths."""
    rng = np.random.default_rng(0)
    s = rng.standard_normal((2, 4, 300))
    common = bc.cbsi_neural(s, s.copy())                 # HbO == HbR (motion/systemic) -> ~0
    anti = bc.cbsi_neural(s, -s.copy())                  # HbO == −HbR (neural activation) -> preserved
    assert np.abs(common).max() < 1e-6
    assert np.abs(anti).mean() > 0.5 * np.abs(s).mean()


def test_estimate_coupling_recovers_planted_lag():
    """Plant a known hemodynamic delay: resp = drive ⊛ gamma(peak=6). `estimate_coupling` should recover a lag
    near 6 s (within the 0.5 s grid + width-floor smoothing), not rail to a boundary."""
    fs, T, n = 10.0, 300, 12
    rng = np.random.default_rng(1)
    tk = np.arange(0, 30, 1.0 / fs)
    g = bc._gamma_kernel(tk, peak=6.0, width=3.0)
    drive = np.array([np.convolve(rng.standard_normal(T), np.ones(8) / 8, "same") for _ in range(n)])
    resp = np.array([np.convolve(d, g)[:T] for d in drive]) + 0.01 * rng.standard_normal((n, T))
    lag, decay, beta = bc.estimate_coupling(drive, resp, fs)
    assert 4.5 <= lag <= 7.5, f"planted 6 s, recovered {lag}"
    assert bc._HRF_WIDTH[0] - 0.1 <= (decay * lag) ** 0.5 <= bc._HRF_WIDTH[1] + 0.1   # width stayed in physio range
    assert beta > 0                                       # positive coupling recovered (drive -> resp)


def test_build_tensor_contract():
    """Paired EEG+fNIRS -> `[n, C=5, grid, grid, T]` with the principled channels [θ, α, β, CBSI-neural,
    coverage], finite everywhere."""
    rng = np.random.default_rng(2)
    n, grid = 3, 12
    Xe = rng.standard_normal((n, 8, 4000))               # 8 EEG ch @100 Hz, 40 s
    Xf = rng.standard_normal((n, 12, 320))               # 6 HbO + 6 HbR @10 Hz, 32 s
    pos_e = rng.uniform(-0.8, 0.8, (8, 2))
    pos_f = rng.uniform(-0.8, 0.8, (6, 2))
    X = bc.build_tensor(bc.PairedModalities(Xe, Xf, pos_e, pos_f), grid=grid,
                        series=bc.SeriesConfig(fps=10.0, t_end=20.0))
    assert X.shape == (n, 5, grid, grid, 200)
    assert np.isfinite(X).all()


def test_fused_node_series_eeg_format():
    """The fusion-only signal collapses to EEG channel format `[n, n_e_finite, T]` (drops non-finite-position
    EEG nodes), finite, ready for spatial-covariance -> the EEG Riemann decoder."""
    rng = np.random.default_rng(3)
    n = 4
    Xe = rng.standard_normal((n, 8, 4000))
    Xf = rng.standard_normal((n, 12, 320))
    pos_e = rng.uniform(-0.8, 0.8, (8, 2)); pos_e[7] = np.nan          # one dropped node
    pos_f = rng.uniform(-0.8, 0.8, (6, 2))
    joint, coupling = bc.fused_node_series(bc.PairedModalities(Xe, Xf, pos_e, pos_f),
                                           series=bc.SeriesConfig(fps=10.0, t_end=20.0))
    assert joint.shape == (n, 7, 200)                                  # 8 EEG nodes − 1 non-finite = 7
    assert np.isfinite(joint).all()
    assert set(coupling) == {"lag", "decay", "beta"}


def test_channel_series_fixed_lag_override_skips_estimation():
    """`SeriesConfig.lag_s` overrides the derived hemodynamic offset: the returned coupling reports exactly
    that lag and NaN decay/beta (no estimation ran)."""
    rng = np.random.default_rng(5)
    n = 3
    Xe = rng.standard_normal((n, 8, 4000))               # 8 EEG ch @100 Hz
    Xf = rng.standard_normal((n, 12, 320))               # 6 HbO + 6 HbR @10 Hz
    eeg, neural, t_dst, coupling = bc.channel_series(
        Xe, Xf, bc.SeriesConfig(fps=10.0, t_end=20.0, lag_s=3.0))
    assert coupling["lag"] == 3.0
    assert np.isnan(coupling["decay"]) and np.isnan(coupling["beta"])
    assert set(eeg) == {"theta", "alpha", "beta"}
    assert neural.shape[0] == n and t_dst.shape == (200,)


def test_csd_transform_preserves_shape():
    """CSD (surface Laplacian) is a spatial filter — same [n, ch, t] shape, finite, and actually CHANGES the
    data (it's a high-pass, not identity). Real 10-20 names so the montage resolves."""
    rng = np.random.default_rng(4)
    ch = ["Fz", "Cz", "Pz", "Oz", "C3", "C4", "F3", "F4", "P3", "P4"]
    Xe = rng.standard_normal((3, len(ch), 500))
    out = bc.csd_transform(Xe, ch, 100.0)
    assert out.shape == Xe.shape
    assert np.isfinite(out).all()
    assert not np.allclose(out, Xe)                       # it did something (spatial high-pass)


def test_coverage_is_local_and_bounded():
    """Coverage ∈ [0,1], and is ~1 where an EEG and fNIRS sensor coincide, ~0 far from both."""
    pos_e = np.array([[0.0, 0.0], [0.5, 0.5]])
    pos_f = np.array([[0.0, 0.0], [-0.5, -0.5]])
    cov = bc.coverage_map(pos_e, pos_f, grid=16)
    assert cov.min() >= 0.0 and cov.max() <= 1.0
    assert cov[8, 8] > 0.5                                # centre: both modalities have a co-located sensor
