"""THINGS-EEG1 adapter — the pure epoching logic (epochs_from_events / _row_mask) on synthetic arrays."""
import numpy as np
import polars as pl
import pytest

from core.data.eeg.things_eeg1 import ThingsEeg1EpochCfg, _row_mask, epochs_from_events


def _events(onsets_s, concepts, files, istarget=None, isteststim=None):
    cols = {"onset": onsets_s, "object": concepts, "stimname": files}
    if istarget is not None:
        cols["istarget"] = istarget
    if isteststim is not None:
        cols["isteststim"] = isteststim
    return pl.DataFrame(cols)


def _ramp_eeg(ch=64, t=3000):
    # each channel = its own scaled sample-index ramp, so an extracted window is exactly identifiable
    return (np.arange(t, dtype=float)[None, :] * (1 + np.arange(ch)[:, None])) * 1e-5   # volts-scale


def test_epochs_window_extraction_and_labels():
    eeg = _ramp_eeg()
    ev = _events([0.1, 0.5, 1.0], ["aardvark", "antelope", "axe"], ["a.jpg", "b.jpg", "c.jpg"])
    cfg = ThingsEeg1EpochCfg(tmin=0.0, tmax=0.2, resample=0)      # 0.2s @1000Hz = 200 samples, no resample
    epochs, concept, files = epochs_from_events(eeg, 1000.0, ev, cfg)
    assert epochs.shape == (3, 64, 200)
    assert list(concept) == ["aardvark", "antelope", "axe"] and list(files) == ["a.jpg", "b.jpg", "c.jpg"]
    # per-channel z-score: ~0 mean, ~unit std within each epoch-channel
    assert np.allclose(epochs.mean(axis=2), 0, atol=1e-4) and np.allclose(epochs.std(axis=2), 1, atol=1e-2)


def test_drop_targets_and_validation_filtering():
    eeg = _ramp_eeg()
    ev = _events([0.1, 0.2, 0.3, 0.4], ["a", "b", "c", "d"], ["1", "2", "3", "4"],
                 istarget=[0, 1, 0, 0], isteststim=[0, 0, 1, 0])
    # default: drop_targets -> row1 gone; exclude validation -> row2 gone; keeps rows 0 and 3
    _, concept, _ = epochs_from_events(eeg, 1000.0, ev, ThingsEeg1EpochCfg(tmax=0.1, resample=0))
    assert list(concept) == ["a", "d"]
    # include_validation keeps the test stim; keep_targets keeps the fixation-colour trial
    _, concept2, _ = epochs_from_events(eeg, 1000.0, ev,
                                        ThingsEeg1EpochCfg(tmax=0.1, resample=0, include_validation=True,
                                                           drop_targets=False))
    assert list(concept2) == ["a", "b", "c", "d"]


def test_row_mask_absent_flag_columns_keep_all():
    ev = _events([0.1, 0.2], ["a", "b"], ["1", "2"])            # no istarget/isteststim columns
    assert _row_mask(ev, ThingsEeg1EpochCfg()).all()


def test_overrunning_window_dropped():
    eeg = _ramp_eeg(t=1000)
    ev = _events([0.1, 0.95], ["a", "b"], ["1", "2"])           # 0.95s + 0.2s window overruns 1.0s recording
    epochs, concept, _ = epochs_from_events(eeg, 1000.0, ev, ThingsEeg1EpochCfg(tmax=0.2, resample=0))
    assert epochs.shape[0] == 1 and list(concept) == ["a"]


def test_onset_units_guard_raises_on_out_of_range():
    eeg = _ramp_eeg(t=1000)
    ev = _events([5000.0], ["a"], ["1"])                        # 5000s * 1000Hz way past a 1000-sample recording
    with pytest.raises(ValueError, match="onset units"):
        epochs_from_events(eeg, 1000.0, ev, ThingsEeg1EpochCfg())


def test_resample_changes_time_length():
    eeg = _ramp_eeg()
    ev = _events([0.5], ["a"], ["1"])
    epochs, _, _ = epochs_from_events(eeg, 1000.0, ev, ThingsEeg1EpochCfg(tmax=0.2, resample=250.0))
    assert epochs.shape == (1, 64, 50)                          # 200 @1000Hz -> 50 @250Hz
