"""core.data.fnirs.base — the FnirsCfg hemodynamic recipe key + FnirsEpochs windowing delegation.

Pure logic: FnirsCfg.key must encode the recipe (band, window, resample, cleaner) so two recipes never collide,
and FnirsEpochs.epoch_blocks must be a thin FnirsCfg adapter over the shared modality-agnostic
Signal.block_epochs (same cutting/edge-drop, driven by the cfg window). CANONICAL_NBACK is re-exported here for
back-compat call sites, so that alias is asserted too.
"""
import numpy as np

from core.data.fnirs.base import CANONICAL_NBACK, FnirsCfg, FnirsEpochs


def test_canonical_nback_reexported():
    assert CANONICAL_NBACK == {"0-back": 0, "2-back": 1, "3-back": 2}


def test_fnirscfg_key_encodes_recipe_and_native_resample():
    base = FnirsCfg()                                    # 0.01-0.2 Hz, t-2..20, native rate
    k = base.key()
    assert k.startswith("b0p01-0p2_t")                   # band encoded, dots -> 'p'
    assert "_rnative_" in k                              # resample=None renders as 'native'
    assert FnirsCfg(l_freq=0.02).key() != k              # a changed band param -> a different cache dir
    assert FnirsCfg(resample=10.0).key() != k            # a concrete rate is distinguishable from native


def test_epoch_blocks_delegates_to_shared_windowing():
    cont = np.arange(100.0).reshape(2, 50)               # 2 ch, 50 samples
    cfg = FnirsCfg(tmin=0.0, tmax=3.0, baseline_s=0.0)
    X, y = FnirsEpochs.epoch_blocks(cont, onsets=np.array([5, 20]), y=np.array([0, 1]), fs=1.0, cfg=cfg)
    assert X.shape == (2, 2, 3)                           # 2 epochs, 2 ch, 3-sample window
    assert y.tolist() == [0, 1]
    assert np.allclose(X[0, 0], cont[0, 5:8])             # first epoch is the raw cfg window
