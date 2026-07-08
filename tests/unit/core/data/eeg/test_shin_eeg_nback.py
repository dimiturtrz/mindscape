"""Shin EEG n-back adapter — pure epoching logic + a data-gated smoke.

The parse (_load_continuous) needs the gitignored .mat, so the equivalence-class test targets `_epoch`
(pure); the end-to-end smoke skips when the raw data isn't present.
"""
import numpy as np
import pytest

from core.data.eeg import shin2017_nback_eeg as m
from core.data.signal import BlockedRecording, block_epochs


def test_block_epochs_windows_and_drops_edges():
    # the EEG adapter's epoching (shared core/data/signal.block_epochs, no baseline)
    ch, T, fs = 28, 1000, 100.0
    cont = np.tile(np.arange(T, dtype=float), (ch, 1))     # ramp per channel = sample index
    onsets = np.array([50, 500, 990])                      # last overruns a [0,2s)=200-sample window
    y = np.array([0, 1, 2])
    X, ye = block_epochs(BlockedRecording(cont, onsets, y), fs, tmin=0.0, tmax=2.0)   # 2s * 100Hz = 200 samples
    assert X.shape == (2, ch, 200)                         # third onset dropped (990+200 > 1000)
    assert list(ye) == [0, 1]
    assert np.allclose(X[0, 0], np.arange(50, 250))        # exact window extracted
    assert np.allclose(X[1, 0], np.arange(500, 700))


def test_block_epochs_all_dropped_returns_empty():
    cont = np.zeros((28, 100))
    X, ye = block_epochs(BlockedRecording(cont, np.array([90]), np.array([1])), 100.0, tmin=0.0, tmax=2.0)  # 200 > 100
    assert X.shape == (0, 28, 200) and len(ye) == 0


def test_adapter_metadata():
    a = m.adapter()
    assert a.name == "shin2017_nback_eeg" and a.n_classes == 3
    assert a.label_map == {"0-back": 0, "2-back": 1, "3-back": 2}


def test_adapter_smoke_if_data_present():
    from core.data.eeg.base import EpochCfg
    a = m.adapter()
    subs = a.subjects()
    if not subs:
        pytest.skip("Shin EEG raw data not present (<data>/raw/shin2017_eeg)")
    X, y, meta = a.get_data([subs[0]], EpochCfg(fmin=4, fmax=30, tmin=0.0, tmax=40.0, resample=100.0))
    assert X.shape[1] == 28 and X.ndim == 3                # 28 EEG channels
    assert sorted(set(y.tolist())) == [0, 1, 2]            # 3 workload classes
    assert len(y) == len(meta) == X.shape[0]
