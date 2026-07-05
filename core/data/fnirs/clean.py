"""fNIRS signal cleaners — a composable preprocessing stage for physiological/motion-noise removal.

fNIRS light samples cortex AND the scalp it passes through, so the measured ΔHbO/ΔHbR carries systemic
hemodynamics (cardiac, respiration, ~0.1 Hz Mayer waves, motion) on top of the neural response. The bandpass
kills the fast systemic (cardiac ~1 Hz, respiration ~0.3 Hz); the in-band remainder (Mayer, motion, common-mode
scalp flow) needs a dedicated cleaning step. This module is that stage.

A `Cleaner` is a small `fit`/`transform` object on an epoch tensor `X[n, ch, t]` (channels = 36 HbO then
36 HbR, paired by index). `fit`/`transform` — not a bare function — because some cleaners estimate parameters
and MUST fit on train only to avoid leakage; the stateless ones here (`Cbsi`, `Detrend`) no-op `fit`, so they
are safe to apply at load. `Chain` composes cleaners in order (order matters — these transforms don't commute).
Data-estimated cleaners (PCA/GLM systemic) would use the same interface but must be fit INSIDE the CV fold,
not here. Idempotency is NOT assumed — CBSI recomputes its mix from the signal's own stats, so f(f(x)) != f(x).

    make_cleaner("cbsi")            # single
    make_cleaner(["cbsi", "detrend"])  # composite, applied left-to-right
"""
from __future__ import annotations

import numpy as np


class Cbsi:
    """Correlation-Based Signal Improvement (Cui 2010). True cortical activation drives ΔHbO and ΔHbR
    ANTI-correlated (oxy in, deoxy out); motion + common-mode systemic drive them TOGETHER. Keep the
    anti-correlated part, drop the common-mode: with α = std(HbO)/std(HbR) per channel per epoch,

        HbO' = ½(HbO − α·HbR),   HbR' = −HbO'/α

    Stateless (α is per-epoch, per-channel → no cross-trial fit → leakage-free at load)."""

    def fit(self, X):
        return self

    def transform(self, X):
        dt = np.asarray(X).dtype
        X = np.asarray(X, dtype=np.float64)
        ch = X.shape[1] // 2
        hbo, hbr = X[:, :ch, :], X[:, ch:, :]                        # [n, ch, t] each
        a = hbo.std(axis=2, keepdims=True) / (hbr.std(axis=2, keepdims=True) + 1e-9)   # [n, ch, 1]
        hbo_c = 0.5 * (hbo - a * hbr)
        hbr_c = -hbo_c / a
        return np.concatenate([hbo_c, hbr_c], axis=1).astype(dt)


class Detrend:
    """Per-channel per-epoch linear detrend — remove a straight-line drift over the window (residual slow
    trend the 0.01 Hz highpass leaves). Stateless; a control more than a fix (largely redundant with the
    highpass, so a near-null ablation result is the expected, honest outcome)."""

    def fit(self, X):
        return self

    def transform(self, X):
        dt = np.asarray(X).dtype
        X = np.asarray(X, dtype=np.float64)
        t = X.shape[2]
        tc = np.arange(t) - (t - 1) / 2.0                           # centred time axis
        denom = (tc * tc).sum()
        slope = (X * tc).sum(axis=2, keepdims=True) / denom         # OLS slope per [n, ch]
        mean = X.mean(axis=2, keepdims=True)
        return (X - mean - slope * tc).astype(dt)


class Chain:
    """Ordered composite — fit/transform each cleaner on the previous one's output (like sklearn Pipeline).
    Non-commutative, so the list order is the applied order."""

    def __init__(self, cleaners: list):
        self.cleaners = cleaners

    def fit(self, X):
        for c in self.cleaners:
            c.fit(X)
            X = c.transform(X)
        return self

    def transform(self, X):
        for c in self.cleaners:
            X = c.transform(X)
        return X


_CLEANERS = {"cbsi": Cbsi, "detrend": Detrend}


def make_cleaner(spec):
    """`str | list[str] | None` -> a `Cleaner` (single = one-element `Chain`) or `None`. Unknown name errors."""
    if spec is None:
        return None
    names = [spec] if isinstance(spec, str) else list(spec)
    if not names:
        return None
    bad = [n for n in names if n not in _CLEANERS]
    if bad:
        raise ValueError(f"unknown cleaner(s) {bad}; known: {sorted(_CLEANERS)}")
    return Chain([_CLEANERS[n]() for n in names])


def clean_key(spec) -> str:
    """Cache-key fragment for a clean spec (part of FnirsCfg.key so a cleaned cache never collides)."""
    if spec is None:
        return "none"
    names = [spec] if isinstance(spec, str) else list(spec)
    return "+".join(names) if names else "none"
