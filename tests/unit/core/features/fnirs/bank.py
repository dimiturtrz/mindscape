"""fNIRS descriptor bank: shape/column-map contract, per-descriptor sanity on known signals."""
import numpy as np
import pytest

from core.features import DescriptorBank
from core.features.fnirs.bank import FNIRS_FEATURE_FNS


def test_family_names():
    """family_names() returns the family list in column order."""
    names = DescriptorBank.family_names()
    assert isinstance(names, list)
    assert len(names) > 0
    expected_families = {"mean", "slope", "peak", "variance", "skew", "kurtosis", "auc", "time_to_peak",
                        "min", "max", "range", "final", "early_slope", "late_slope", "zero_crossings"}
    assert set(names) == expected_families


def test_extract_bank():
    """extract_bank() concatenates all descriptor families and maps each column to its family."""
    n, ch, t = 5, 72, 220
    X = np.random.default_rng(0).standard_normal((n, ch, t))
    F, fam = DescriptorBank.extract_bank(X)
    K = len(DescriptorBank.family_names())
    assert F.shape == (n, ch * K)
    assert fam.shape == (ch * K,)
    # each family owns a contiguous ch-wide block, in DescriptorBank.family_names() order
    for i, name in enumerate(DescriptorBank.family_names()):
        assert (fam[i * ch:(i + 1) * ch] == name).all()
    assert np.isfinite(F).all()


def test_descriptors_on_known_signals():
    # channel 0: constant 3.0; channel 1: linear ramp 0..t-1
    t = 100
    X = np.zeros((1, 2, t))
    X[0, 0, :] = 3.0
    X[0, 1, :] = np.arange(t)
    f = {name: fn(X)[0] for name, fn in FNIRS_FEATURE_FNS.items()}         # name -> [ch]

    assert f["mean"][0] == pytest.approx(3.0)                              # constant
    assert f["slope"][0] == pytest.approx(0.0, abs=1e-9)
    assert f["variance"][0] == pytest.approx(0.0, abs=1e-9)
    assert f["range"][0] == pytest.approx(0.0, abs=1e-9)
    assert f["peak"][0] == pytest.approx(3.0)

    assert f["slope"][1] > 0                                               # ramp rises
    assert f["peak"][1] == pytest.approx(t - 1)                            # signed extreme = the max
    assert f["time_to_peak"][1] == pytest.approx((t - 1) / t)             # peak at the end
    assert f["min"][1] == pytest.approx(0.0)
    assert f["max"][1] == pytest.approx(t - 1)




def test_f_time_to_peak():
    """f_time_to_peak: latency (in [0,1)) of each channel's signed extreme — a ramp peaks at the end."""
    t = 100
    X = np.zeros((1, 2, t))
    X[0, 1, :] = np.arange(t)                              # channel 1 ramps up -> extreme at the last sample
    ttp = DescriptorBank.f_time_to_peak(X)
    assert ttp.shape == (1, 2)
    assert ttp[0, 1] == pytest.approx((t - 1) / t)


def test_f_zero_crossings():
    """f_zero_crossings: number of sign changes over time — an alternating signal crosses every step."""
    alt = np.tile([1.0, -1.0], 50)                        # 100 samples, sign flips every step -> 99 crossings
    X = np.stack([np.full(100, 3.0), alt])[None]          # channel 0 constant (0 crossings), channel 1 alternating
    zc = DescriptorBank.f_zero_crossings(X)
    assert zc.shape == (1, 2)
    assert zc[0, 0] == pytest.approx(0.0)                  # constant never crosses
    assert zc[0, 1] == pytest.approx(99.0)
