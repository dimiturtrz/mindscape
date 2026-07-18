"""Cross-modality data primitives shared by the EEG and fNIRS adapters.

A Butterworth bandpass and a block-onset windowing op don't care whether the channels are electrodes or
optodes, and the n-back workload labels are the same task in both modalities — so these live here, in the
neutral data layer, rather than inside one modality's base (which would force the other to import "up" into
it). `core/data/eeg/*` and `core/data/fnirs/*` both depend on this; it depends on neither.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import numpy as np
import polars as pl
from jaxtyping import Float, Int
from scipy.signal import butter, filtfilt

# canonical n-back workload classes (load level) — shared by the fNIRS and EEG n-back adapters
CANONICAL_NBACK: dict[str, int] = {"0-back": 0, "2-back": 1, "3-back": 2}


@dataclass
class BlockedRecording:
    """A continuous [ch, T] recording plus the per-block onset samples and labels that carve it into epochs —
    the three things `block_epochs` always needs together."""
    cont: np.ndarray
    onsets: np.ndarray
    labels: np.ndarray


class EpochBatch:
    """Accumulate per-subject epochs + their meta columns, then stack into the common
    (X[n,ch,t] float32, y[n] int64, subject/session/run frame) triple every dataset adapter returns — the
    one home for the assemble-and-frame contract (the Store schema), shared across the EEG + fNIRS adapters."""

    def __init__(self):
        self.Xs: list[np.ndarray] = []
        self.ys: list[np.ndarray] = []
        self.subj: list[str] = []
        self.sess: list[str] = []
        self.run: list[str] = []

    def add(self, X: Float[np.ndarray, "n ch t"], y: Int[np.ndarray, "n"],
            subj: list[str], sess: list[str], run: list[str]) -> None:
        """Append one subject's epochs and its per-epoch subject/session/run meta columns."""
        self.Xs.append(X)
        self.ys.append(y)
        self.subj += subj
        self.sess += sess
        self.run += run

    def stack(self) -> tuple[Float[np.ndarray, "n ch t"], Int[np.ndarray, "n"], pl.DataFrame]:
        """Concatenate to the canonical float32/int64 arrays + the subject/session/run meta frame."""
        X = np.concatenate(self.Xs).astype(np.float32)
        y = np.concatenate(self.ys).astype(np.int64)
        meta = pl.DataFrame({"subject": self.subj, "session": self.sess, "run": self.run})
        return X, y, meta


class Signal:
    """Cross-modality data primitives (bandpass, block epoching) as staticmethods (public names kept) — the
    neutral data layer both EEG and fNIRS adapters ride on."""

    @staticmethod
    def bandpass(X: Float[np.ndarray, "*batch t"], l_freq: float, h_freq: float, fs: float, order: int = 4
                 ) -> Float[np.ndarray, "*batch t"]:
        """Zero-phase Butterworth bandpass, filtering the LAST (time) axis (filtfilt — no phase shift). Rank-
        agnostic: `*batch` leading axes pass through, so continuous [ch, T] and epoched [n, ch, T] both work."""
        nyq = fs / 2.0
        b, a = cast(tuple[np.ndarray, np.ndarray],
                    butter(order, [l_freq / nyq, min(h_freq, nyq * 0.99) / nyq], btype="band"))
        return filtfilt(b, a, X, axis=-1)

    @staticmethod
    def block_epochs(rec: BlockedRecording, fs: float, tmin: float, tmax: float, baseline_s: float = 0.0
                     ) -> tuple[np.ndarray, np.ndarray]:
        """Cut a continuous [ch, T] recording into epochs [onset+tmin, onset+tmax) at each onset (samples).

        Optionally subtract the pre-onset `baseline_s` mean per channel (fNIRS uses it; for EEG covariance
        methods it's left 0 — the covariance is mean-invariant). Epochs whose window falls off the recording
        edge are dropped. Returns (X [n, ch, t] float32, y [n]). Vectorized: one fancy-index, no per-epoch loop.
        """
        cont = rec.cont
        start, stop = round(tmin * fs), round(tmax * fs)
        n_baseline = round(baseline_s * fs)
        T = cont.shape[1]
        onsets, y = np.asarray(rec.onsets), np.asarray(rec.labels)
        valid = (onsets + start >= 0) & (onsets + stop <= T)               # window fully on the recording
        if not valid.any():
            return np.empty((0, cont.shape[0], stop - start), np.float32), np.empty(0, np.int64)
        idx = onsets[valid][:, None] + np.arange(start, stop)              # [n_valid, stop-start] sample indices
        segs = cont[:, idx].transpose(1, 0, 2).astype(np.float32)          # [n_valid, ch, stop-start]
        base = segs[:, :, :n_baseline].mean(axis=2, keepdims=True) if n_baseline > 0 else 0.0
        return segs - base, y[valid].astype(np.int64)
