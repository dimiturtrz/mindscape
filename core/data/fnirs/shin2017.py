"""Shin et al. 2017 hybrid EEG-fNIRS — the fNIRS half, parsed direct from the TU Berlin `.mat`.

MOABB loads only the EEG half (fNIRS raises NotImplementedError), so this adapter goes to the raw `.mat`:
the NIRS files already store oxy/deoxy hemoglobin (mmol/L, 10 Hz, 36 ch), so preprocessing enters AFTER
Beer-Lambert — bandpass + baseline-corrected epoching, no optical-density / MBLL step.

    <data>/raw/shin2017/VP<NNN>-NIRS/{cnt,mrk,mnt}_<task>.mat   (task ∈ {nback, dsr, wg})

Download: TU Berlin DepositOnce, DOI 10.14279/depositonce-5830.2 (per-subject VP<NNN>-NIRS.zip, GPL-3.0).
This adapter targets the n-back workload task (0/2/3-back = the canonical fNIRS load-level decode).
Channels are HbO then HbR (72 total): [<36 oxy>, <36 deoxy>].
"""
from __future__ import annotations

import numpy as np
import polars as pl

from core.config import raw_dir
from core.data.fnirs.base import CANONICAL_NBACK, FnirsCfg, bandpass, epoch_blocks


class Shin2017NirsAdapter:
    """fNIRS n-back workload adapter over the Shin-2017 `.mat`. Implements the DatasetAdapter contract."""

    def __init__(self, task: str = "nback"):
        self.name = f"shin2017_{task}"
        self.task = task
        self.n_classes = 3
        self.label_map = dict(CANONICAL_NBACK)          # source class name -> canonical id

    def subjects(self) -> list[int]:
        """Subjects actually present on disk (this cognitive set ships 26; robust to partial downloads)."""
        root = raw_dir() / "shin2017"
        subs = []
        for d in sorted(root.glob("VP*-NIRS")):
            if (d / f"cnt_{self.task}.mat").exists():
                subs.append(int(d.name[2:5]))
        return subs

    def _subject_dir(self, sub: int):
        return raw_dir() / "shin2017" / f"VP{sub:03d}-NIRS"

    def _load_continuous(self, sub: int):
        """Return (cont [72, T] HbO|HbR, fs, onsets[samples], canonical_labels[n]) for one subject."""
        import scipy.io as sio

        d = self._subject_dir(sub)
        cnt = sio.loadmat(d / f"cnt_{self.task}.mat", struct_as_record=False, squeeze_me=True)[f"cnt_{self.task}"]
        mrk = sio.loadmat(d / f"mrk_{self.task}.mat", struct_as_record=False, squeeze_me=True)[f"mrk_{self.task}"]
        oxy, deoxy = cnt.oxy, cnt.deoxy
        fs = float(oxy.fs)
        cont = np.concatenate([np.asarray(oxy.x).T, np.asarray(deoxy.x).T], axis=0)   # [72, T]
        onsets = np.round(np.asarray(mrk.time) / 1000.0 * fs).astype(int)             # ms -> samples
        src_idx = np.asarray(mrk.y).argmax(0)                                          # class per event (className order)
        names = [str(c).split()[0] for c in np.asarray(mrk.className)]                # '0-back session' -> '0-back'
        y = np.array([self.label_map[names[i]] for i in src_idx], dtype=np.int64)
        return cont, fs, onsets, y

    def get_data(self, subjects: list[int] | None, cfg: FnirsCfg
                 ) -> tuple[np.ndarray, np.ndarray, pl.DataFrame]:
        """Epoch requested subjects -> (X [n,72,t] float32, y [n] canonical int, meta polars frame).
        meta: subject, session (chronological thirds ≈ the 3 recording series), run."""
        subs = subjects or self.subjects()
        Xs, ys, subj, sess, run = [], [], [], [], []
        for sub in subs:
            cont, fs, onsets, y = self._load_continuous(sub)
            cont = bandpass(cont, cfg.l_freq, cfg.h_freq, fs)
            order = np.argsort(onsets)                                                # chronological
            onsets, y = onsets[order], y[order]
            X, ye = epoch_blocks(cont, onsets, y, fs, cfg)
            if cfg.clean is not None:                                                  # physiological-noise stage
                from core.data.fnirs.clean import make_cleaner
                X = make_cleaner(cfg.clean).transform(X).astype(np.float32)            # stateless -> leakage-free
            if cfg.resample and cfg.resample != fs:
                from scipy.signal import resample as _rs
                X = _rs(X, int(round(X.shape[2] * cfg.resample / fs)), axis=2).astype(np.float32)
            n = len(ye)
            Xs.append(X)
            ys.append(ye)
            subj += [str(sub)] * n
            sess += [str(i // 9) for i in range(n)]                                   # 27 blocks -> 3 sessions of 9
            run += ["0"] * n
        X = np.concatenate(Xs).astype(np.float32)
        y = np.concatenate(ys).astype(np.int64)
        meta = pl.DataFrame({"subject": subj, "session": sess, "run": run})
        return X, y, meta


def adapter(task: str = "nback") -> Shin2017NirsAdapter:
    return Shin2017NirsAdapter(task=task)
