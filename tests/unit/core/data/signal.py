"""core.data.signal — the cross-modality primitives (Butterworth bandpass, block epoching).

Pure numpy signal ops, no I/O: the bandpass must pass an in-band tone and kill an out-of-band one; block
epoching must cut the right windows, drop epochs whose window runs off the edge, and subtract the pre-onset
baseline when asked. Boundary case: an onset whose window exactly reaches T is kept; one sample past is dropped.
"""
import numpy as np

from core.data.signal import BlockedRecording, Signal


def _tone(freq, fs, n):
    t = np.arange(n) / fs
    return np.sin(2 * np.pi * freq * t)[None, :]        # [1, n]


def test_bandpass_passes_in_band_kills_out_of_band():
    fs, n = 128.0, 2048
    x = _tone(15.0, fs, n) + _tone(2.0, fs, n)          # 15 Hz in [8,32]; 2 Hz below the band
    y = Signal.bandpass(x, 8.0, 32.0, fs)
    mid = slice(256, -256)                              # drop filtfilt edge transients
    # the pure 2 Hz reference loses almost all its energy; the 15 Hz component survives
    r2 = np.sqrt((Signal.bandpass(_tone(2.0, fs, n), 8.0, 32.0, fs)[0, mid] ** 2).mean())
    r15 = np.sqrt((Signal.bandpass(_tone(15.0, fs, n), 8.0, 32.0, fs)[0, mid] ** 2).mean())
    assert r2 < 0.05                                    # out-of-band attenuated to near zero
    assert r15 > 0.5                                    # in-band tone preserved (unit-amplitude sine)
    assert y.shape == x.shape


def _rec(cont, onsets, labels):
    return BlockedRecording(np.asarray(cont, float), np.asarray(onsets), np.asarray(labels))


def test_block_epochs_cuts_windows_and_drops_edge():
    cont = np.arange(200.0).reshape(2, 100)             # 2 ch, 100 samples
    rec = _rec(cont, onsets=[10, 50, 98], labels=[0, 1, 2])   # 98+5=103 > 100 -> dropped
    X, y = Signal.block_epochs(rec, fs=1.0, tmin=0.0, tmax=5.0)
    assert X.shape == (2, 2, 5)                         # 2 kept epochs, 2 ch, 5 samples
    assert y.tolist() == [0, 1]                         # third (edge) epoch dropped with its label
    assert np.allclose(X[0, 0], cont[0, 10:15])         # first epoch is the raw window (no baseline)


def test_block_epochs_baseline_subtracts_pre_onset_mean():
    cont = np.zeros((1, 50))
    cont[0, 10:20] = 5.0                                # flat level inside the window
    rec = _rec(cont, onsets=[10], labels=[0])
    X, _ = Signal.block_epochs(rec, fs=1.0, tmin=0.0, tmax=6.0, baseline_s=3.0)
    assert np.allclose(X[0, 0, :3], 0.0)               # first 3 samples are the baseline -> zeroed


def test_block_epochs_all_invalid_returns_empty():
    rec = _rec(np.zeros((3, 10)), onsets=[9], labels=[1])     # 9+5 > 10 -> nothing valid
    X, y = Signal.block_epochs(rec, fs=1.0, tmin=0.0, tmax=5.0)
    assert X.shape == (0, 3, 5) and y.shape == (0,)
