"""Pre-raster cross-modal series (`core.features.fusion.series.Series`).

`channel_series` puts paired EEG/fNIRS on one lag-aligned display grid. Contracts: the display grid is
`t_end*fps` samples; EEG yields the three band envelopes (>= 0, they are analytic magnitudes); a fixed
`lag_s` override skips the derived-coupling estimate (NaN decay/beta), while the default derives a finite lag.
"""
import numpy as np

from core.features.fusion.series import Series, SeriesConfig


def _epochs(rng, n=3, n_e=8, n_f=6):
    return rng.standard_normal((n, n_e, 4000)), rng.standard_normal((n, 2 * n_f, 320))


def test_channel_series_grid_and_envelope_contract():
    """Returns (eeg{theta,alpha,beta}, neural, t_dst, coupling); the grid has `t_end*fps` samples and the
    band envelopes are non-negative (|analytic signal|)."""
    rng = np.random.default_rng(0)
    Xe, Xf = _epochs(rng)
    eeg, neural, t_dst, coupling = Series.channel_series(Xe, Xf, SeriesConfig(fps=10.0, t_end=20.0))
    assert set(eeg) == {"theta", "alpha", "beta"}
    assert t_dst.shape == (200,)
    assert eeg["alpha"].shape == (3, 8, 200)
    assert neural.shape == (3, 6, 200)
    assert all((eeg[b] >= 0).all() for b in eeg)         # envelopes are magnitudes
    assert set(coupling) == {"lag", "decay", "beta"}


def test_fixed_lag_override_skips_estimation():
    """`lag_s` given -> coupling reports exactly that lag with NaN decay/beta (no estimation ran)."""
    rng = np.random.default_rng(1)
    Xe, Xf = _epochs(rng)
    _, _, _, coupling = Series.channel_series(Xe, Xf, SeriesConfig(fps=10.0, t_end=20.0, lag_s=3.0))
    assert coupling["lag"] == 3.0
    assert np.isnan(coupling["decay"]) and np.isnan(coupling["beta"])


def test_derived_lag_is_finite_and_default_grid_length():
    """Default (lag_s=None) derives a finite hemodynamic lag; default SeriesConfig grid is t_end*fps = 200."""
    rng = np.random.default_rng(2)
    Xe, Xf = _epochs(rng)
    _, neural, t_dst, coupling = Series.channel_series(Xe, Xf)
    assert t_dst.shape == (200,)                          # 20 s * 10 fps
    assert np.isfinite(coupling["lag"])
    assert np.isfinite(neural).all()
