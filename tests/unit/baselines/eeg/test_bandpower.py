"""EegBandpower — the workload-native EEG decode: per-channel θ/α/β log band-power → shrinkage-LDA.

Builds trials whose CLASS is in the ABSOLUTE power of one band (an added alpha oscillation), the signal
band-power reads and covariance would normalize away, and checks the feature shape, the relative-mode
scale-invariance, and that the decoder beats chance through the fit/predict_proba contract.
"""
import numpy as np
import pytest

from baselines.eeg import bandpower as bp_module
from baselines.eeg.bandpower import EegBandpower
from core.features import band_powers as _bandpower

FS = 100.0
N_CH = 4
N_T = 400          # 4 s at 100 Hz — enough for a 2 s Welch segment


def _band_dataset(n_per_class=40, seed=0):
    """class 1 = class 0 + an alpha (10 Hz) oscillation on every channel → higher absolute alpha power."""
    rng = np.random.default_rng(seed)
    t = np.arange(N_T) / FS
    X, y = [], []
    for cls in (0, 1):
        for _ in range(n_per_class):
            sig = rng.normal(scale=1.0, size=(N_CH, N_T))
            if cls == 1:
                sig = sig + 1.5 * np.sin(2 * np.pi * 10.0 * t)[None, :]   # inject alpha
            X.append(sig)
            y.append(cls)
    return np.asarray(X, dtype=np.float64), np.asarray(y)


def test_bandpower_shape_is_three_bands_per_channel():
    X, _ = _band_dataset(n_per_class=3)
    F = _bandpower(X, FS)
    assert F.shape == (6, N_CH * 3)           # [n, ch * (theta,alpha,beta)]


def test_relative_mode_is_scale_invariant():
    """relative=True divides each band by the epoch's total band-power, so scaling the signal leaves the
    features (near) unchanged — unlike absolute log-power, which shifts by log(scale**2)."""
    X, _ = _band_dataset(n_per_class=5)
    abs_1, abs_2 = _bandpower(X, FS, relative=False), _bandpower(2 * X, FS, relative=False)
    rel_1, rel_2 = _bandpower(X, FS, relative=True), _bandpower(2 * X, FS, relative=True)
    assert not np.allclose(abs_1, abs_2, atol=0.1)            # absolute power moves with scale
    assert np.allclose(rel_1, rel_2, atol=1e-6)               # relative power does not


def test_decodes_absolute_band_power_signal():
    Xtr, ytr = _band_dataset(seed=1)
    Xte, yte = _band_dataset(seed=2)
    clf = EegBandpower(fs=FS).fit(Xtr, ytr)
    proba = clf.predict_proba(Xte)
    assert proba.shape == (len(yte), 2)
    assert np.allclose(proba.sum(1), 1.0, atol=1e-6)
    acc = (proba.argmax(1) == yte).mean()
    assert acc > 0.8                                          # a clean band-power signal must decode


@pytest.mark.parametrize("relative", [False, True])
def test_fit_returns_self_and_proba_contract(relative):
    X, y = _band_dataset(n_per_class=15)
    clf = EegBandpower(fs=FS, relative=relative)
    assert clf.fit(X, y) is clf                              # fit -> self (Decoder contract)
    assert clf.predict_proba(X).shape == (len(y), 2)


def test_module_shims_delegate_to_class():
    X, y = _band_dataset(n_per_class=15)
    clf = bp_module.fit(X, y)
    probs = bp_module.score(clf, X)
    assert probs.shape == (len(y), 2)
