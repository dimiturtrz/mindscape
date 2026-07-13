"""EEG band-power feature (`core.features.eeg.bandpower.BandPower`).

Equivalence classes over the spectral content: a channel dominated by one rhythm must show the most power in
that rhythm's band; the output must carry one log-power value per (band, channel).
"""
import numpy as np

from core.features.eeg.bandpower import CANONICAL_BANDS, BandPower


def _sine(freq, fs, t, rng):
    return np.sin(2 * np.pi * freq * np.arange(t) / fs) + 0.01 * rng.standard_normal(t)


def test_band_powers_shape_is_bands_times_channels():
    """`[n, ch, t]` -> `[n, ch*len(bands)]` (one log band-power per band per channel)."""
    rng = np.random.default_rng(0)
    n, ch, t, fs = 4, 5, 400, 100.0
    X = rng.standard_normal((n, ch, t))
    out = BandPower.band_powers(X, fs)
    assert out.shape == (n, ch * len(CANONICAL_BANDS))
    assert np.isfinite(out).all()


def test_dominant_rhythm_has_the_most_power():
    """A 10 Hz (alpha) channel must read more power in alpha than in theta or beta. Layout is
    [theta(ch..), alpha(ch..), beta(ch..)] so a channel's three band values are ch columns apart."""
    rng = np.random.default_rng(1)
    fs, t, ch = 100.0, 600, 3
    X = np.stack([np.stack([_sine(10.0, fs, t, rng) for _ in range(ch)])])   # [1, ch, t], all alpha
    out = BandPower.band_powers(X, fs)                                       # [1, 3*ch]
    theta, alpha, beta = out[0, :ch], out[0, ch:2 * ch], out[0, 2 * ch:]
    assert (alpha > theta).all()
    assert (alpha > beta).all()


def test_relative_is_scale_free_absolute_is_not():
    """`relative=True` divides out a per-channel amplitude gain (fraction of total power), so scaling the
    signal leaves the relative features unchanged while the absolute (log-power) features shift by ~log(g^2)."""
    rng = np.random.default_rng(2)
    fs, t = 100.0, 400
    X = np.stack([np.stack([_sine(10.0, fs, t, rng), _sine(20.0, fs, t, rng)])])   # [1, 2, t]
    g = 5.0
    rel = BandPower.band_powers(X, fs, relative=True)
    rel_scaled = BandPower.band_powers(g * X, fs, relative=True)
    assert np.allclose(rel, rel_scaled, atol=1e-4)
    absol = BandPower.band_powers(X, fs)
    absol_scaled = BandPower.band_powers(g * X, fs)
    assert not np.allclose(absol, absol_scaled, atol=1e-2)
