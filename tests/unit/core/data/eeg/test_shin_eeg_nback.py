"""Shin EEG n-back adapter — pure epoching logic (equivalence-class on the shared block_epochs) + adapter
metadata. The BBCI `.mat` parse (_load_continuous) is disk IO — its parse logic wants a synthetic loadmat
fixture (bd), not a data-gated skip; the adapter file itself is omitted from coverage (disk read).
"""
import numpy as np

from core.data.eeg import shin2017_nback_eeg as m
from core.data.signal import BlockedRecording, Signal


def test_block_epochs_windows_and_drops_edges():
    # the EEG adapter's epoching (shared core/data/signal.block_epochs, no baseline)
    ch, T, fs = 28, 1000, 100.0
    cont = np.tile(np.arange(T, dtype=float), (ch, 1))     # ramp per channel = sample index
    onsets = np.array([50, 500, 990])                      # last overruns a [0,2s)=200-sample window
    y = np.array([0, 1, 2])
    X, ye = Signal.block_epochs(BlockedRecording(cont, onsets, y), fs, tmin=0.0, tmax=2.0)   # 2s * 100Hz = 200 samples
    assert X.shape == (2, ch, 200)                         # third onset dropped (990+200 > 1000)
    assert list(ye) == [0, 1]
    assert np.allclose(X[0, 0], np.arange(50, 250))        # exact window extracted
    assert np.allclose(X[1, 0], np.arange(500, 700))


def test_block_epochs_all_dropped_returns_empty():
    cont = np.zeros((28, 100))
    X, ye = Signal.block_epochs(BlockedRecording(cont, np.array([90]), np.array([1])), 100.0, tmin=0.0, tmax=2.0)  # 200 > 100
    assert X.shape == (0, 28, 200) and len(ye) == 0


def test_adapter_metadata():
    a = m.Shin2017NbackEegAdapter.adapter()
    assert a.name == "shin2017_nback_eeg" and a.n_classes == 3
    assert a.label_map == {"0-back": 0, "2-back": 1, "3-back": 2}
