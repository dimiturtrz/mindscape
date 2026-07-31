"""fNIRS amplitude features (`core.features.fnirs.amplitude.Amplitude`).

The canonical mean+slope+peak triple. Equivalence classes: a flat channel (mean set, slope ~0), a rising
ramp (positive slope, positive peak), a negative dip (negative signed peak); plus a monotonic amplitude
response — scaling the epoch scales mean and peak.
"""
import numpy as np

from core.features.fnirs.amplitude import Amplitude


def test_amplitude_features_shape_and_known_values():
    """`[n, ch, t]` -> `[n, 3*ch]` laid out [mean(ch..), slope(ch..), peak(ch..)]; each block recovers the
    hand-computable statistic for a flat, a rising, and a dipping channel."""
    t = 50
    ramp = np.linspace(0.0, 1.0, t)                      # rising HbO-like response
    flat = np.full(t, 2.0)
    dip = -np.linspace(0.0, 3.0, t)                      # HbR-like negative deflection
    X = np.stack([np.stack([ramp, flat, dip])])          # [1, 3, t]
    ch = 3
    feat = Amplitude.amplitude_features(X)
    assert feat.shape == (1, 3 * ch)
    mean, slope, peak = feat[0, :ch], feat[0, ch:2 * ch], feat[0, 2 * ch:]
    assert np.isclose(mean[1], 2.0)                      # flat channel mean
    assert slope[0] > 0 and abs(slope[1]) < 1e-6         # ramp rises, flat is flat
    assert peak[0] > 0 and peak[2] < 0                   # signed extreme: HbO up, HbR down


def test_amplitude_response_scales_monotonically():
    """Doubling the epoch amplitude doubles the mean-amplitude and peak features (the linear stats), i.e. a
    stronger hemodynamic response reads as a proportionally larger feature."""
    rng = np.random.default_rng(0)
    x = np.abs(rng.standard_normal((1, 4, 60)))
    X = np.concatenate([x, 2.0 * x], axis=0)             # sample 1 = 2x amplitude of sample 0
    feat = Amplitude.amplitude_features(X)
    ch = 4
    mean, peak = feat[:, :ch], feat[:, 2 * ch:]
    assert np.allclose(mean[1], 2.0 * mean[0], rtol=1e-4)
    assert np.allclose(peak[1], 2.0 * peak[0], rtol=1e-4)
