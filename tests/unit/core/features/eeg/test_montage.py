"""EEG scalp geometry (`eeg_positions` + `_to_unit_disk`) — pure lookup/geometry, no disk.

Reads the in-memory MNE standard 10-05 montage and normalizes to the unit head disk: known channels land at
finite positions inside the disk, unknown channels become NaN rows.
"""
import numpy as np

from core.features.eeg.montage import _to_unit_disk, eeg_positions


def test_eeg_positions_shape_and_unit_disk():
    ch = ["Cz", "Fz", "Pz", "Oz", "C3", "C4"]
    pos = eeg_positions(ch)
    assert pos.shape == (len(ch), 2)
    r = np.hypot(pos[:, 0], pos[:, 1])
    assert np.all(r <= 1.0 + 1e-6)                          # normalized into the unit disk
    assert np.isfinite(pos).all()


def test_unknown_channel_is_nan_row():
    pos = eeg_positions(["Cz", "NOT_A_CHANNEL"])
    assert np.isnan(pos[1]).all()                           # missing channel -> NaN, not a fabricated position


def test_to_unit_disk_centers_and_scales():
    raw = np.array([[10.0, 10.0], [10.0, 12.0], [12.0, 10.0], [8.0, 8.0]])
    out = _to_unit_disk(raw)
    assert np.allclose(out.mean(0), 0.0, atol=1e-6)        # centred on the mean
    assert np.hypot(out[:, 0], out[:, 1]).max() <= 1.0 + 1e-6
