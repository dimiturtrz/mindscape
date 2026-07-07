"""EEG surface-Laplacian / Current Source Density — a spatial deblur (single-modality EEG prep)."""
from __future__ import annotations

import numpy as np


def csd_transform(Xe: np.ndarray, ch_names, fs: float, *, montage: str = "standard_1005") -> np.ndarray:
    """Current Source Density / surface Laplacian (spherical spline, Perrin 1989 — via MNE). A reference-free
    spatial HIGH-PASS that undoes most of the volume-conduction / skull blur, sharpening each EEG channel toward
    the cortex beneath it. Stays in scalp space (no head model, no inverse) — the cheap PARTIAL fix for EEG's
    spatial smear before fusion: it de-blurs, but does NOT relocate tangential sources or add depth (that needs
    source localization). `Xe` [n, ch, t] -> [n, ch, t] (CSD units, spatially sharpened)."""
    import mne
    info = mne.create_info(list(ch_names), float(fs), "eeg")
    ep = mne.EpochsArray(np.asarray(Xe, dtype=float), info, verbose="ERROR")
    ep.set_montage(montage, match_case=False, on_missing="ignore", verbose="ERROR")
    ep = mne.preprocessing.compute_current_source_density(ep, verbose="ERROR")
    return ep.get_data().astype(np.float32)
