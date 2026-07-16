"""Physics-forward synthetic fNIRS (bd 7jn, lighter path) — an INDEPENDENT ground-truth generator to validate
our coupling-lag + CBSI extraction (bd uqw), without the circularity of testing our single-gamma estimator
against its own model.

Independence by construction: the forward hemodynamics here are a **double-gamma** HRF (SPM canonical: a
positive lobe + an undershoot) — a different shape than `estimate_coupling`'s single-gamma fit — driven by a
known neural signal with a known neurovascular delay. Systemic physiology (Mayer ~0.1 Hz, respiration, cardiac)
is injected as **common-mode** in both chromophores (what CBSI must cancel), and neural activation as
**anti-correlated** HbO↑ / HbR↓ (what CBSI must keep). So recovering the planted lag from the synthetic is a
genuine test of the estimator, and recovering the neural drive is a genuine test of CBSI.

The full published mesh-Monte-Carlo simulators (arXiv 2605.30552 / 2405.11242) are MATLAB/Docker-heavy with no
public repo (see research/deep_dives/2026-07-09_fnirs_simulator_2605.md); this analytic forward covers the
operator-validation use they were wanted for, minus photon transport / spatial montage realism.
"""
from __future__ import annotations

import numpy as np
from jaxtyping import Float
from pydantic import BaseModel
from scipy.signal import fftconvolve
from scipy.stats import gamma


class SynthConfig(BaseModel):
    """Forward-model knobs. HRF defaults = SPM canonical double-gamma (deliberately NOT our estimator's shape)."""
    hrf_len_s: float = 32.0
    hrf_peak: float = 6.0          # positive-lobe peak time (s) — sets the neurovascular delay
    hrf_under: float = 16.0        # undershoot peak time (s)
    hrf_ratio: float = 1.0 / 6.0   # undershoot weight
    hbr_ratio: float = 0.4         # HbR anti-correlation magnitude vs HbO (activation: HbO↑, HbR↓)
    systemic_amp: float = 0.6      # common-mode systemic amplitude (relative to unit neural response)
    mayer_hz: float = 0.1          # Mayer wave; + respiration + cardiac below
    resp_hz: float = 0.25
    cardiac_hz: float = 1.1
    noise_std: float = 0.08        # independent per-channel measurement noise


class Synthetic:
    """Physics-forward synthetic fNIRS generator — the free helpers folded in as staticmethods (public names
    kept), so the independent double-gamma forward model has one home."""

    @staticmethod
    def double_gamma_hrf(fs: float, cfg: SynthConfig | None = None) -> Float[np.ndarray, "t"]:
        """SPM canonical double-gamma HRF sampled at `fs`, peak-normalized. Positive lobe minus a weighted
        undershoot — the independent forward shape (cf. our single-gamma *estimator* in fusion.coupling)."""
        cfg = cfg or SynthConfig()
        t = np.arange(0, cfg.hrf_len_s, 1.0 / fs)
        h = gamma.pdf(t, cfg.hrf_peak) - cfg.hrf_ratio * gamma.pdf(t, cfg.hrf_under)
        return h / np.abs(h).max()

    @staticmethod
    def systemic(n: int, length: int, fs: float, cfg: SynthConfig,
                 rng: np.random.Generator) -> Float[np.ndarray, "n t"]:
        """Common-mode systemic physiology [n, T] — Mayer/respiration/cardiac oscillations at random phase, plus
        a slow drift. Added identically to HbO and HbR so CBSI cancels it."""
        t = np.arange(length) / fs
        out = np.zeros((n, length))
        for hz in (cfg.mayer_hz, cfg.resp_hz, cfg.cardiac_hz):
            phase = rng.uniform(0, 2 * np.pi, (n, 1))
            out += np.sin(2 * np.pi * hz * t[None, :] + phase)
        out += rng.standard_normal((n, 1)) * t[None, :] / t[-1]        # slow linear drift, per-trial
        return cfg.systemic_amp * out / 3.0

    @staticmethod
    def synthesize_paired(neural_drive: Float[np.ndarray, "n t"], fs: float, cfg: SynthConfig | None = None,
                          seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
        """Forward-generate paired (HbO, HbR) `[n, T]` from a neural drive `[n, T]`.

        `HbO = (neural ⊛ HRF) + systemic + noise`, `HbR = -hbr_ratio·(neural ⊛ HRF) + systemic + noise`. The
        neurovascular delay is the HRF's center of mass (~`hrf_peak`), the ground-truth lag `estimate_coupling`
        should recover; the anti-correlated neural + common-mode systemic are the ground truth CBSI separates."""
        cfg = cfg or SynthConfig()
        rng = np.random.default_rng(seed)
        drive = np.asarray(neural_drive, dtype=np.float64)
        hrf = Synthetic.double_gamma_hrf(fs, cfg)
        response = fftconvolve(drive, hrf[None, :], axes=1)[:, :drive.shape[1]]
        n, length = drive.shape
        sys_o = Synthetic.systemic(n, length, fs, cfg, rng)
        sys_r = sys_o + Synthetic.systemic(n, length, fs, cfg, rng) * 0.15   # near-common-mode (slight de-corr)
        hbo = response + sys_o + rng.standard_normal((n, length)) * cfg.noise_std
        hbr = -cfg.hbr_ratio * response + sys_r + rng.standard_normal((n, length)) * cfg.noise_std
        return hbo.astype(np.float32), hbr.astype(np.float32)
