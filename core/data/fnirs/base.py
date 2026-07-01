"""Shared fNIRS primitives + the FnirsCfg recipe — the hemodynamic sibling of core/data/eeg/base.py.

fNIRS decodes a SLOW hemodynamic signal (ΔHbO/ΔHbR, ~10 Hz), so the recipe differs from EEG: a very-low
bandpass (drift + heartbeat/Mayer removal) and a long, hemodynamically-delayed window (the response peaks
~8-12 s post-onset). Same downstream contract though — an epoch tensor is [n, ch, t] float32, labels are
canonical ints, meta is one row/epoch (subject, session, run) — so the SAME store/splits/harness ride on it.

Canonical n-back workload labels (fixed so a decoder's classes mean the same everywhere):
    0 nback0   1 nback2   2 nback3
"""
from __future__ import annotations

import numpy as np
from pydantic import BaseModel

# canonical workload classes (n-back load level)
CANONICAL_NBACK: dict[str, int] = {"0-back": 0, "2-back": 1, "3-back": 2}
CANONICAL_NBACK_NAMES: dict[int, str] = {v: k for k, v in CANONICAL_NBACK.items()}


class FnirsCfg(BaseModel):
    """Preprocessing params that define an epoched fNIRS cache. Two recipes never collide (see `key`).

    Defaults = a standard hemodynamic block-design recipe: 0.01-0.2 Hz band (kill drift + pulse/Mayer),
    a window from -2 s (baseline) to +20 s (capture the delayed HbO peak), baseline-corrected on the
    pre-onset 2 s. Native 10 Hz kept (resample=None)."""
    l_freq: float = 0.01
    h_freq: float = 0.2
    tmin: float = -2.0
    tmax: float = 20.0
    baseline_s: float = 2.0
    resample: float | None = None

    def key(self) -> str:
        def f(x):
            return str(x).replace(".", "p").replace("-", "m")
        rs = "native" if self.resample is None else f(self.resample)
        return f"b{f(self.l_freq)}-{f(self.h_freq)}_t{f(self.tmin)}-{f(self.tmax)}_r{rs}"


def bandpass(X: np.ndarray, l_freq: float, h_freq: float, fs: float, order: int = 4) -> np.ndarray:
    """Zero-phase Butterworth bandpass on continuous [ch, T] (filtfilt — no phase shift on the slow HRF)."""
    from scipy.signal import butter, filtfilt
    nyq = fs / 2.0
    b, a = butter(order, [l_freq / nyq, min(h_freq, nyq * 0.99) / nyq], btype="band")
    return filtfilt(b, a, X, axis=-1)


def epoch_blocks(cont: np.ndarray, onsets: np.ndarray, y: np.ndarray, fs: float, cfg: FnirsCfg
                 ) -> tuple[np.ndarray, np.ndarray]:
    """Cut a continuous [ch, T] recording into baseline-corrected epochs at `onsets` (samples).
    Returns (X [n, ch, t] float32, y [n]) — epochs whose window falls off the recording are dropped."""
    a, b = int(round(cfg.tmin * fs)), int(round(cfg.tmax * fs))
    nb = int(round(cfg.baseline_s * fs))
    T = cont.shape[1]
    Xs, ys = [], []
    for o, lab in zip(onsets, y):
        s, e = o + a, o + b
        if s < 0 or e > T:
            continue
        seg = cont[:, s:e].astype(np.float32)
        base = seg[:, :nb].mean(axis=1, keepdims=True) if nb > 0 else 0.0   # pre-onset baseline
        Xs.append(seg - base)
        ys.append(lab)
    if not Xs:                                                              # all epochs fell off the edge
        return np.empty((0, cont.shape[0], b - a), np.float32), np.empty(0, np.int64)
    return np.stack(Xs), np.asarray(ys, dtype=np.int64)
