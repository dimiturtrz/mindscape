"""Load one subject's paired EEG+fNIRS brain-camera inputs — the shared input stage of the fusion viz
(`viz.py`, the animated GIF) and the fusion export (`export.py`, the web JSON).

Both need the same thing: block-aligned EEG+fNIRS for one subject, EEG surface-Laplacian (CSD) deblurred,
plus both montages' 2D positions. One home so the "what the brain-camera reads" contract lives in one place.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from core.data import store
from core.data.eeg import shin2017_nback_eeg as eegmod
from core.data.eeg.base import EpochCfg
from core.data.fnirs import shin2017 as fnmod
from core.data.fnirs.base import FnirsCfg
from core.features import fusion as bc


@dataclass
class BrainCameraInputs:
    """One subject's paired, montage-located brain-camera inputs."""
    Xe: np.ndarray                   # [n, ch_e, t] CSD-deblurred EEG
    Xf: np.ndarray                   # [n, ch_f, t] fNIRS HbO/HbR
    y: np.ndarray                    # [n] block labels (shared, alignment-guarded)
    pos_e: np.ndarray                # [ch_e, 2] EEG unit-disk positions
    pos_f: np.ndarray                # [ch_f, 2] fNIRS unit-disk positions


class PairedInputs:
    """Loader for the brain-camera's paired EEG+fNIRS input stage."""

    @staticmethod
    def load(subject: int, fs_e: float = 100.0, fn_tmax: float = 32.0) -> BrainCameraInputs:
        """Block-aligned EEG+fNIRS for one subject + montage positions, EEG CSD-deblurred. `fn_tmax` epochs
        fNIRS past the 20 s window so the read-forward (τ+lag) tail still has blood."""
        me = store.Store.load("shin2017_nback_eeg", EpochCfg(fmin=4, fmax=30, tmin=0.0, tmax=40.0, resample=fs_e))
        mf = store.Store.load("shin2017_nback", FnirsCfg(tmax=fn_tmax))
        xe, xf, y = store.Store.gather_aligned(me, mf, subject)
        channels = eegmod.Shin2017NbackEegAdapter.adapter().channels()
        xe = bc.CSD.csd_transform(xe, channels, fs_e)               # surface-Laplacian deblur before fusion
        pos_e = bc.EegMontage.eeg_positions(channels)
        pos_f = bc.FnirsMontage.fnirs_positions(fnmod.Shin2017NirsAdapter.adapter().subject_dir(subject))
        return BrainCameraInputs(xe, xf, y, pos_e, pos_f)
