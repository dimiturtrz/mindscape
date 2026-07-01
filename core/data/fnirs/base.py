"""Shared fNIRS primitives + the FnirsCfg recipe — the hemodynamic sibling of core/data/eeg/base.py.

fNIRS decodes a SLOW hemodynamic signal (ΔHbO/ΔHbR, ~10 Hz), so the recipe differs from EEG: a very-low
bandpass (drift + heartbeat/Mayer removal) and a long, hemodynamically-delayed window (the response peaks
~8-12 s post-onset). Same downstream contract though — an epoch tensor is [n, ch, t] float32, labels are
canonical ints, meta is one row/epoch (subject, session, run) — so the SAME store/splits/harness ride on it.

Canonical n-back workload labels (fixed so a decoder's classes mean the same everywhere):
    0 nback0   1 nback2   2 nback3
"""
from __future__ import annotations

from pydantic import BaseModel

# Cross-modality primitives (bandpass, block epoching, n-back labels) live in the neutral data layer
# (core/data/signal) so the EEG adapter doesn't import "up" into fNIRS. Re-exported here so existing
# `from core.data.fnirs.base import bandpass / epoch_blocks / CANONICAL_NBACK` call sites keep working.
from core.data.signal import (  # noqa: F401
    CANONICAL_NBACK, CANONICAL_NBACK_NAMES, bandpass, block_epochs)


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


def epoch_blocks(cont, onsets, y, fs: float, cfg: FnirsCfg) -> tuple:
    """Baseline-corrected block epoching per the fNIRS recipe — a thin FnirsCfg adapter over the shared
    `signal.block_epochs` (the modality-agnostic windowing op)."""
    return block_epochs(cont, onsets, y, fs, cfg.tmin, cfg.tmax, cfg.baseline_s)
