"""EEG scalp geometry — 2D electrode positions on the unit head disk (single-modality; the fusion pipeline and
any EEG-alone spatial method reuse this)."""
from __future__ import annotations

import numpy as np


def eeg_positions(ch_names: list[str]) -> np.ndarray:
    """2D scalp positions for EEG channels via the MNE standard 10-05 montage, normalized to the unit disk.
    Returns `[n_ch, 2]` in the same head-disk convention as `fnirs_positions`."""
    import mne
    m = mne.channels.make_standard_montage("standard_1005").get_positions()["ch_pos"]
    pos = np.array([m[c][:2] if c in m else (np.nan, np.nan) for c in ch_names], dtype=float)  # x,y (drop z)
    return _to_unit_disk(pos)


def _to_unit_disk(pos: np.ndarray) -> np.ndarray:
    """Center + scale a 2D montage so its channels fit the unit disk (radius ≤ 1) — a common head frame."""
    p = pos - np.nanmean(pos, axis=0)
    r = np.nanmax(np.hypot(p[:, 0], p[:, 1]))
    return p / (r + 1e-9)
