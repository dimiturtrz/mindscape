"""fNIRS pipeline units: preprocessing (bandpass/epoch), the feature baseline, and cfg keying.
Pure/synthetic — no dataset download needed."""
import numpy as np

from baselines import fnirs_features
from core.features import amplitude_features
from core.data.fnirs.base import FnirsCfg, bandpass, epoch_blocks


def test_cfg_key_is_recipe_unique():
    a = FnirsCfg().key()
    b = FnirsCfg(l_freq=0.02).key()
    assert a != b and "b" in a and "native" in a          # default resample=None -> 'native'


def test_bandpass_removes_dc_and_drift():
    fs = 10.0
    t = np.arange(2000)
    x = 5.0 + 0.01 * t + np.sin(2 * np.pi * 0.1 * t / fs)  # DC + linear drift + a 0.1 Hz component
    y = bandpass(x[None, :], 0.01, 0.2, fs)[0]
    assert abs(y.mean()) < 0.1                              # DC + drift gone
    assert y.std() > 0.1                                    # the in-band oscillation survives


def test_epoch_blocks_windows_and_baseline_corrects():
    fs = 10.0
    cont = np.ones((4, 1000)) * 3.0                         # constant 3.0 everywhere
    onsets = np.array([200, 500])
    y = np.array([0, 1])
    cfg = FnirsCfg(tmin=-2.0, tmax=8.0, baseline_s=2.0)
    X, ye = epoch_blocks(cont, onsets, y, fs, cfg)
    assert X.shape == (2, 4, 100)                           # (tmax-tmin)*fs = 10s*10 = 100
    assert np.allclose(X, 0.0)                              # constant signal -> baseline-corrected to 0
    assert list(ye) == [0, 1]


def test_epoch_blocks_extracts_correct_samples():
    # ramp signal -> exact window + baseline are known, so this catches an off-by-window indexing bug
    fs, Tn = 10.0, 1000
    cont = np.tile(np.arange(Tn, dtype=float), (3, 1))     # each channel = 0,1,2,... sample index
    cfg = FnirsCfg(tmin=-2.0, tmax=8.0, baseline_s=2.0)     # a=-20, b=80, nb=20
    a, b, nb = int(round(-2.0 * fs)), int(round(8.0 * fs)), int(round(2.0 * fs))
    X, ye = epoch_blocks(cont, np.array([300]), np.array([0]), fs, cfg)
    seg = np.arange(300 + a, 300 + b, dtype=float)         # the samples that should be extracted
    assert X.shape == (1, 3, b - a)
    assert np.allclose(X[0, 0], seg - seg[:nb].mean())     # right window, baseline-subtracted
    assert list(ye) == [0]


def test_epoch_blocks_drops_out_of_range():
    cont = np.zeros((2, 300))
    # onset near the end -> window overruns -> dropped
    X, ye = epoch_blocks(cont, np.array([290]), np.array([1]), 10.0, FnirsCfg(tmin=-2, tmax=8))
    assert len(ye) == 0


def test_fnirs_features_shape_and_decodes_amplitude_signal():
    rng = np.random.default_rng(0)
    n, ch, t = 60, 6, 100
    y = np.tile([0, 1, 2], n // 3)
    # class encoded in per-channel MEAN LEVEL (what covariance discards, features must catch)
    X = rng.normal(scale=0.1, size=(n, ch, t)) + (y[:, None, None] * 1.0)
    feats = amplitude_features(X)
    assert feats.shape == (n, 3 * ch)                       # mean + slope + peak
    clf = fnirs_features.fit(X, y)
    acc = (fnirs_features.score(clf, X).argmax(1) == y).mean()
    assert acc > 0.9                                        # amplitude-coded classes must decode
