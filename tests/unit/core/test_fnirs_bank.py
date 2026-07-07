"""fNIRS descriptor bank: shape/column-map contract, per-descriptor sanity on known signals, and the
one thing that's easy to get wrong — the weight must land AFTER standardisation."""
import numpy as np
import pytest

from core.features import extract_bank, family_names
from core.features.fnirs.bank import FNIRS_FEATURE_FNS, WeightedFamilyScaler


def test_extract_bank_shape_and_column_map():
    n, ch, t = 5, 72, 220
    X = np.random.default_rng(0).standard_normal((n, ch, t))
    F, fam = extract_bank(X)
    K = len(family_names())
    assert F.shape == (n, ch * K)
    assert fam.shape == (ch * K,)
    # each family owns a contiguous ch-wide block, in family_names() order
    for i, name in enumerate(family_names()):
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


def test_weighted_scaler_standardises_then_weights():
    rng = np.random.default_rng(1)
    X = rng.normal(loc=5.0, scale=2.0, size=(200, 2))                      # 2 columns, non-unit mean/scale
    fam = np.array(["a", "b"])
    ws = WeightedFamilyScaler(fam, {"a": 1.0, "b": 2.0}).fit(X)
    Z = ws.transform(X)
    # column a: standardised -> ~unit std; column b: standardised THEN ×2 -> ~2× std
    assert np.std(Z[:, 0]) == pytest.approx(1.0, abs=0.05)
    assert np.std(Z[:, 1]) == pytest.approx(2.0, abs=0.1)
    assert np.mean(Z[:, 0]) == pytest.approx(0.0, abs=0.05)


def test_weight_zero_drops_family():
    X = np.random.default_rng(2).standard_normal((50, 3))
    fam = np.array(["keep", "drop", "keep"])
    Z = WeightedFamilyScaler(fam, {"drop": 0.0}).fit(X).transform(X)
    assert np.allclose(Z[:, 1], 0.0)                                       # dropped family is zeroed out
    assert not np.allclose(Z[:, 0], 0.0)


def test_scaler_fits_on_train_only():
    # std_ comes from the fitted (train) data, not the transformed set — no leakage
    rng = np.random.default_rng(3)
    Xtr = rng.normal(0, 1, (100, 1))
    Xte = rng.normal(0, 5, (100, 1))                                       # test has 5× the spread
    ws = WeightedFamilyScaler(np.array(["a"]), {}).fit(Xtr)
    Zte = ws.transform(Xte)
    assert np.std(Zte[:, 0]) == pytest.approx(5.0, abs=0.6)               # scaled by TRAIN std (~1), so ~5
