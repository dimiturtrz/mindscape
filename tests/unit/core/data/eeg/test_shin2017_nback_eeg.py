"""Shin2017 n-back EEG BBCI `.mat` parse (`_load_continuous`) — the parse logic, no disk (bd 0s4).

`scipy.io.loadmat` is monkeypatched to return synthetic BBCI-shaped structs (squeeze_me style: attribute
access), so the channel-select / EOG-drop / marker->block-onset logic is exercised without the ~GB raw files.
Equivalence classes: EOG channels dropped (30 -> 28), only '*-back session' markers kept (non-session
filtered), onsets converted ms -> samples, canonical labels mapped.
"""
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from core.data.eeg import shin2017_nback_eeg as mod
from core.data.eeg.base import EpochCfg


@pytest.fixture
def synthetic_mat(monkeypatch):
    """cnt.x [T,30] (28 EEG + HEOG/VEOG), mrk with 3 kept '*-back session' markers + 1 filtered 'instruction'."""
    fs, t = 200.0, 800
    cnt = SimpleNamespace(x=np.arange(t * 30, dtype=float).reshape(t, 30), fs=fs,
                          clab=[f"E{i}" for i in range(28)] + ["HEOG", "VEOG"])
    classes = ["0-back session", "2-back session", "3-back session", "instruction"]
    mrk = SimpleNamespace(className=np.array(classes, dtype=object),
                          y=np.eye(4)[[0, 1, 2, 3]].T,                 # one-hot [4 classes, 4 markers]
                          time=np.array([1000.0, 2000.0, 3000.0, 3500.0]))   # ms
    monkeypatch.setattr(mod.sio, "loadmat", lambda *a, **k: {"cnt_nback": cnt, "mrk_nback": mrk})
    return fs


def test_load_continuous_parses_channels_onsets_labels(synthetic_mat):
    cont, fs, onsets, y = mod.Shin2017NbackEegAdapter.adapter()._load_continuous(Path("ignored/path"))

    assert cont.shape[0] == mod._N_EEG == 28                          # 30 -> 28: HEOG/VEOG dropped
    assert cont.shape[1] == 800 and fs == synthetic_mat               # [28, T], fs carried
    np.testing.assert_array_equal(onsets, [200, 400, 600])            # ms/1000*fs, only the 3 'session' markers
    np.testing.assert_array_equal(y, [0, 1, 2])                       # 0-/2-/3-back -> canonical 0/1/2


def test_load_continuous_drops_non_session_markers(synthetic_mat):
    _, _, onsets, y = mod.Shin2017NbackEegAdapter.adapter()._load_continuous(Path("ignored/path"))
    assert len(onsets) == len(y) == 3                                 # 'instruction' marker filtered out


def test_get_data_epochs_end_to_end(synthetic_mat, monkeypatch):
    """get_data over the mocked parse: bandpass -> block-epoch -> resample -> (X[n,28,t], y, meta)."""
    monkeypatch.setattr(mod.Shin2017NbackEegAdapter, "_index", lambda self: {1: Path("dummy")})

    x, y, meta = mod.Shin2017NbackEegAdapter.adapter().get_data([1], EpochCfg(fmin=1.0, fmax=40.0, tmin=0.0, tmax=0.5, resample=100.0))

    assert x.shape[:2] == (3, 28) and x.shape[2] == 50                # 3 blocks, 28 EEG ch, 0.5 s @ 100 Hz
    np.testing.assert_array_equal(sorted(y), [0, 1, 2])              # the 3 back-levels epoched
    assert meta["subject"].to_list() == ["1", "1", "1"]             # per-block meta carried
