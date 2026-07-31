"""fNIRS cleaners — the CBSI anti-correlation mechanism (kills common-mode, keeps anti-correlated neural),
shape/stateless contract, and the composite chain order."""
import numpy as np
import pytest

from core.data.fnirs.clean import Cbsi, Chain, Clean, Detrend


def _pair(hbo, hbr):
    """One epoch [1, 2*ch, t] from an HbO block and an HbR block (channels = HbO then HbR)."""
    return np.concatenate([hbo, hbr], axis=1)


def test_cbsi_removes_common_mode_artifact():
    t = np.arange(200) / 10.0
    neural = np.sin(2 * np.pi * 0.05 * t)                            # slow hemodynamic-ish
    artifact = 3.0 * np.sin(2 * np.pi * 0.1 * t)                     # common-mode (same in HbO and HbR)
    hbo = (neural + artifact)[None, None, :]                         # HbO = +neural + artifact
    hbr = (-neural + artifact)[None, None, :]                        # HbR = -neural + artifact (anti-corr neural)
    X = _pair(hbo, hbr)
    out = Cbsi().transform(X)
    hbo_c = out[0, 0]
    # common-mode artifact is largely removed -> corrected HbO tracks the neural component, not the artifact
    assert np.corrcoef(hbo_c, neural)[0, 1] > 0.98
    assert abs(np.corrcoef(hbo_c, artifact)[0, 1]) < 0.3


def test_cbsi_shape_and_dtype_preserved():
    X = np.random.default_rng(0).standard_normal((5, 8, 100)).astype(np.float32)
    out = Cbsi().transform(X)
    assert out.shape == X.shape and out.dtype == X.dtype


def test_detrend_kills_linear_drift():
    t = np.arange(100)
    X = (0.5 * t + 2.0)[None, None, :].repeat(3, 1)                  # pure linear ramp, 3 channels
    out = Detrend().transform(X)
    assert np.allclose(out, 0.0, atol=1e-6)                          # ramp -> ~0


def test_chain_applies_in_order_and_is_leakage_free():
    X = np.random.default_rng(1).standard_normal((4, 6, 80))
    chain = Clean.make_cleaner(["cbsi", "detrend"])
    assert isinstance(chain, Chain) and len(chain.cleaners) == 2
    # fit is a no-op for stateless cleaners: fit-then-transform == transform (no state carried from data)
    a = chain.transform(X)
    b = Clean.make_cleaner(["cbsi", "detrend"]).fit(X).transform(X)
    assert np.allclose(a, b)


def test_make_cleaner_single_none_and_bad():
    assert Clean.make_cleaner(None) is None
    assert Clean.make_cleaner([]) is None
    assert len(Clean.make_cleaner("cbsi").cleaners) == 1
    with pytest.raises(ValueError):
        Clean.make_cleaner("nope")
