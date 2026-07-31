"""Per-family weighted standardiser for the fNIRS descriptor bank: fit/transform contract, weight ordering,
and train/test isolation (no leakage)."""
import numpy as np
import pytest

from core.features.fnirs.weighted_family_scaler import WeightedFamilyScaler


def test_fit():
    """fit() captures per-column mean and std, and builds per-family weight vector."""
    rng = np.random.default_rng(3)
    Xtr = rng.normal(0, 1, (100, 1))
    Xte = rng.normal(0, 5, (100, 1))                                       # test has 5× the spread
    ws = WeightedFamilyScaler(np.array(["a"]), {}).fit(Xtr)
    # After fit, scaler holds train statistics
    assert hasattr(ws, "mean_")
    assert hasattr(ws, "std_")
    assert hasattr(ws, "w_")
    # transform() applies those statistics, not test statistics — no leakage
    Zte = ws.transform(Xte)
    assert np.std(Zte[:, 0]) == pytest.approx(5.0, abs=0.6)               # scaled by TRAIN std (~1), so ~5


def test_transform():
    """transform() standardises using fit statistics, THEN applies per-family weights."""
    rng = np.random.default_rng(1)
    X = rng.normal(loc=5.0, scale=2.0, size=(200, 2))                      # 2 columns, non-unit mean/scale
    fam = np.array(["a", "b"])
    ws = WeightedFamilyScaler(fam, {"a": 1.0, "b": 2.0}).fit(X)
    Z = ws.transform(X)
    # column a: standardised -> ~unit std; column b: standardised THEN ×2 -> ~2× std
    assert np.std(Z[:, 0]) == pytest.approx(1.0, abs=0.05)
    assert np.std(Z[:, 1]) == pytest.approx(2.0, abs=0.1)
    assert np.mean(Z[:, 0]) == pytest.approx(0.0, abs=0.05)
    # Weight 0 drops the family entirely
    X2 = np.random.default_rng(2).standard_normal((50, 3))
    fam2 = np.array(["keep", "drop", "keep"])
    Z2 = WeightedFamilyScaler(fam2, {"drop": 0.0}).fit(X2).transform(X2)
    assert np.allclose(Z2[:, 1], 0.0)                                      # dropped family is zeroed out
    assert not np.allclose(Z2[:, 0], 0.0)
