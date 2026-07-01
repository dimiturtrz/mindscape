"""Shin 2018 — the EEG half of the n-back workload task (same subjects/trials as the fNIRS n-back).

Simultaneous EEG+fNIRS recording, so identical 0/2/3-back load blocks — decoding BOTH on this one task is
the honest same-task modality comparison (Table B, alongside `core/data/fnirs/shin2017.py`) and the basis
for EEG+fNIRS fusion. Unlike fNIRS (workload lives in HbO amplitude, which covariance methods discard), EEG
carries workload as band-power (frontal theta / parietal alpha) — covariance-based CSP/Riemann CAN read it.

Parsed from the BBCI `.mat` (MOABB carries only Shin's MI + mental-arith sets, not this cognitive one):
    <data>/raw/shin2017_eeg/VP<NNN>-EEG/{cnt,mrk}_nback.mat
`cnt.x` is [T, 30] @ 200 Hz — 28 EEG + HEOG/VEOG; the 2 EOG channels are dropped. `mrk` has 8 event
classes; we keep the 3 BLOCK-level `*-back session` onsets (27/subject, 9 per load), which match the
fNIRS block-level workload. (The `target`/`non-target` events are the per-stimulus ERP task — a different
problem, not decoded here.)
Download: EEG_01-26_MATLAB.zip, doc.ml.tu-berlin.de/simultaneous_EEG_NIRS (DOI 10.14279/depositonce-5830).
"""
from __future__ import annotations

import re

import numpy as np
import polars as pl

from core.config import raw_dir
from core.data.eeg.base import EpochCfg
from core.data.fnirs.base import CANONICAL_NBACK, bandpass

_ROOT = "shin2017_eeg"
_N_EEG = 28          # first 28 clab entries are EEG; the last two (HEOG, VEOG) are EOG — dropped
_BLOCK_TMAX = 40.0   # default block window (s) when cfg.tmax is None — the n-back task period


class Shin2017NbackEegAdapter:
    """EEG n-back workload adapter over the Shin-2018 BBCI `.mat`. Implements the DatasetAdapter contract;
    block-level 0/2/3-back load (3-class) — the same task as the fNIRS n-back, so results share Table B."""

    def __init__(self):
        self.name = "shin2017_nback_eeg"
        self.n_classes = 3
        self.label_map = dict(CANONICAL_NBACK)          # '0-back'/'2-back'/'3-back' -> 0/1/2

    def _index(self) -> dict[int, object]:
        """{subject int -> VP dir holding cnt_nback.mat}, discovered on disk (naming-robust)."""
        out: dict[int, object] = {}
        for f in sorted((raw_dir() / _ROOT).glob("**/cnt_nback.mat")):
            m = re.search(r"(\d{1,3})", f.parent.name)
            if m:
                out[int(m.group(1))] = f.parent
        return out

    def subjects(self) -> list[int]:
        return sorted(self._index())

    def _load_continuous(self, d):
        """(cont [28, T] EEG, fs, block onsets[samples], canonical labels[27]) — block-level workload only."""
        import scipy.io as sio

        cnt = sio.loadmat(d / "cnt_nback.mat", struct_as_record=False, squeeze_me=True)["cnt_nback"]
        mrk = sio.loadmat(d / "mrk_nback.mat", struct_as_record=False, squeeze_me=True)["mrk_nback"]
        cont = np.asarray(cnt.x).T[:_N_EEG]                             # [28, T], drop HEOG/VEOG
        fs = float(cnt.fs)
        names = [str(c) for c in np.asarray(mrk.className)]
        idx = np.asarray(mrk.y).argmax(0)                              # event-class id per marker
        keep = np.array(["session" in names[i] for i in idx])          # block-level '*-back session' only
        onsets = np.round(np.asarray(mrk.time)[keep] / 1000.0 * fs).astype(int)
        y = np.array([self.label_map[names[i].split()[0]] for i in idx[keep]], dtype=np.int64)
        return cont, fs, onsets, y

    def get_data(self, subjects: list[int] | None, cfg: EpochCfg
                 ) -> tuple[np.ndarray, np.ndarray, pl.DataFrame]:
        """Epoch requested subjects -> (X [n,28,t] float32, y [n] canonical int, meta polars frame).
        meta: subject, session (chronological thirds ≈ the 3 recording series of 9 blocks), run."""
        idx = self._index()
        subs = subjects or sorted(idx)
        tmax = cfg.tmax if cfg.tmax is not None else _BLOCK_TMAX
        Xs, ys, subj, sess, run = [], [], [], [], []
        for sub in subs:
            cont, fs, onsets, y = self._load_continuous(idx[sub])
            cont = bandpass(cont, cfg.fmin, cfg.fmax, fs)
            order = np.argsort(onsets)                                  # chronological
            onsets, y = onsets[order], y[order]
            X, ye = _epoch(cont, onsets, y, fs, cfg.tmin, tmax)
            if cfg.resample and cfg.resample != fs:
                from scipy.signal import resample as _rs
                X = _rs(X, int(round(X.shape[2] * cfg.resample / fs)), axis=2).astype(np.float32)
            n = len(ye)
            Xs.append(X)
            ys.append(ye)
            subj += [str(sub)] * n
            sess += [str(i // 9) for i in range(n)]                     # 27 blocks -> 3 series of 9
            run += ["0"] * n
        X = np.concatenate(Xs).astype(np.float32)
        y = np.concatenate(ys).astype(np.int64)
        return X, y, pl.DataFrame({"subject": subj, "session": sess, "run": run})


def _epoch(cont, onsets, y, fs, tmin, tmax):
    """Slice [onset+tmin, onset+tmax) per block onset -> (X [n,ch,L] f32, y). No baseline subtraction:
    CSP/Riemann read the covariance, which is mean-invariant. Windows off the recording edge are dropped."""
    a, b = int(round(tmin * fs)), int(round(tmax * fs))
    T = cont.shape[1]
    valid = (onsets + a >= 0) & (onsets + b <= T)
    if not valid.any():
        return np.empty((0, cont.shape[0], b - a), np.float32), np.empty(0, np.int64)
    ix = onsets[valid][:, None] + np.arange(a, b)
    X = cont[:, ix].transpose(1, 0, 2).astype(np.float32)              # [n_valid, ch, L]
    return X, y[valid].astype(np.int64)


def adapter() -> Shin2017NbackEegAdapter:
    return Shin2017NbackEegAdapter()
