"""Shared fNIRS primitives + the FnirsCfg recipe — the hemodynamic sibling of core/data/eeg/base.py.

fNIRS decodes a SLOW hemodynamic signal (ΔHbO/ΔHbR, ~10 Hz), so the recipe differs from EEG: a very-low
bandpass (drift + heartbeat/Mayer removal) and a long, hemodynamically-delayed window (the response peaks
~8-12 s post-onset). Same downstream contract though — an epoch tensor is [n, ch, t] float32, labels are
canonical ints, meta is one row/epoch (subject, session, run) — so the SAME store/splits/harness ride on it.

Canonical n-back workload labels (fixed so a decoder's classes mean the same everywhere):
    0 nback0   1 nback2   2 nback3
"""
from __future__ import annotations

from typing import override

import numpy as np
from jaxtyping import Float, Shaped

# Cross-modality primitives (block epoching, n-back labels, the Recipe base) live in the neutral data layer
# (core/data/signal) so the EEG adapter doesn't import "up" into fNIRS. CANONICAL_NBACK is re-exported here
# so existing `from core.data.fnirs.base import CANONICAL_NBACK` call sites keep working.
from core.data.fnirs.clean import Chain
from core.data.signal import (  # noqa: F401
    CANONICAL_NBACK,
    BlockedRecording,
    Recipe,
    Signal,
)


class FnirsCfg(Recipe):
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
    # physiological/motion-noise cleaner applied to epochs after bandpass+baseline: None | "cbsi" | "detrend"
    # | a list (composite chain). Stateless cleaners only here (leakage-free at load); see fnirs/clean.py.
    clean: str | list[str] | None = None

    @override
    def key(self) -> str:
        rs = "native" if self.resample is None else self._fmt(self.resample)
        return (f"b{self._fmt(self.l_freq)}-{self._fmt(self.h_freq)}"
                f"_t{self._fmt(self.tmin)}-{self._fmt(self.tmax)}_r{rs}_c{Chain.clean_key(self.clean)}")


class FnirsEpochs:
    """fNIRS-recipe epoching over the shared modality-agnostic windowing op (public name kept)."""

    @staticmethod
    def epoch_blocks(
        cont: Float[np.ndarray, "ch t"],
        onsets: Shaped[np.ndarray, "n"],
        y: Shaped[np.ndarray, "n"],
        fs: float,
        cfg: FnirsCfg,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Baseline-corrected block epoching per the fNIRS recipe — a thin FnirsCfg adapter over the shared
        `Signal.block_epochs` (the modality-agnostic windowing op)."""
        return Signal.block_epochs(BlockedRecording(cont, onsets, y), fs, cfg.tmin, cfg.tmax, cfg.baseline_s)
