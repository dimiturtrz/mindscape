"""Feature extraction — the modality-specific signal→feature transforms the decoders sit on top of.

This is where the actual signal-processing substance lives: turning epochs `[n, ch, t]` into the
representation a classifier reads. The `baselines/` methods are thin — they *call* these and bolt a
classifier on. Keeping the extraction here (not inside each method) means one covariance/band-power/amplitude
implementation, reused across methods, modalities, transfer, and the viz.

Grouped by what they produce:
  covariance space (EEG geometric methods)  — `time_delay_embed`, `recenter_covariances`, `scale_to_identity`
  band-power (EEG oscillatory / workload)   — `band_powers`
  amplitude (fNIRS hemodynamic)             — `amplitude_features`
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np

# workload-relevant EEG rhythms (theta rises / alpha suppresses with load); also the MI mu/beta live here
CANONICAL_BANDS = (("theta", 4.0, 7.0), ("alpha", 8.0, 13.0), ("beta", 13.0, 30.0))


# --- covariance-space transforms (EEG Riemannian methods + transfer) -------------------------------------

def time_delay_embed(X: np.ndarray, order: int, lag: int) -> np.ndarray:
    """Augmented Covariance Method embedding: stack `order` lagged copies of each trial so the covariance
    becomes `[ch*order, ch*order]` and encodes temporal dynamics, not just spatial structure.
    `X [n, ch, t] -> [n, ch*order, t-(order-1)*lag]` — folds *time* into the SPD matrix without the
    instability of a short sliding window (Carrara & Papadopoulo)."""
    n, ch, t = X.shape
    L = t - (order - 1) * lag
    if L <= 0:
        raise ValueError(f"order*lag too large for trial length {t} (order={order}, lag={lag})")
    return np.concatenate([X[:, :, k * lag:k * lag + L] for k in range(order)], axis=1)


def recenter_covariances(C: np.ndarray) -> np.ndarray:
    """Congruence-transport one domain's covariances to the identity: `C -> M^{-1/2} C M^{-1/2}`, where M is
    the domain's Riemannian (Fréchet) mean. Removes the per-domain LOCATION shift on the SPD manifold
    (Zanini et al. 2018) while preserving the relative class geometry — the manifold version of whitening,
    applied per subject to kill the between-subject nuisance. Unsupervised → deployment-friendly."""
    from pyriemann.utils.base import invsqrtm
    from pyriemann.utils.mean import mean_riemann

    C = np.asarray(C, dtype=np.float64)
    W = invsqrtm(mean_riemann(C))
    return np.einsum("ij,njk,kl->nil", W, C, W)


def scale_to_identity(C: np.ndarray, target_disp: float = 1.0) -> np.ndarray:
    """Normalize dispersion (RPA step 2): after re-centering to the identity, stretch each covariance so the
    mean squared Riemannian distance to I equals `target_disp` — matches the domains' *spread*, not just
    their location. `C -> C**p` with `p = sqrt(target_disp / current_dispersion)`."""
    from pyriemann.utils.base import powm
    from pyriemann.utils.distance import distance_riemann

    eye = np.eye(C.shape[-1])
    disp = float(np.mean([distance_riemann(c, eye) ** 2 for c in C])) + 1e-12
    p = np.sqrt(target_disp / disp)
    return np.stack([powm(c, p) for c in C])


# --- band-power (EEG oscillatory / workload) -------------------------------------------------------------

def band_powers(X: np.ndarray, fs: float, bands=CANONICAL_BANDS, relative: bool = False) -> np.ndarray:
    """Per-channel log band-power in each band -> `[n, ch*len(bands)]`. One Welch PSD over the time axis
    (vectorized across n and ch), then integrate each band.

    `relative=False` -> log absolute power (best *within*-subject; the absolute scale is subject-specific so
    it transfers poorly). `relative=True` -> each band as a FRACTION of the epoch's total band-power (per
    channel), which divides out the subject/session amplitude offset — the standard cross-subject fix."""
    from scipy.signal import welch

    nperseg = min(X.shape[2], int(round(fs * 2)))            # 2 s segments (or the whole epoch if shorter)
    freqs, psd = welch(X, fs=fs, nperseg=nperseg, axis=2)    # psd: [n, ch, f]
    P = np.stack([psd[:, :, (freqs >= lo) & (freqs < hi)].sum(axis=2) for _n, lo, hi in bands], axis=0)
    if relative:
        P = P / (P.sum(axis=0, keepdims=True) + 1e-12)      # fraction of total -> scale-free
    return np.concatenate([np.log(P[b] + 1e-12) for b in range(P.shape[0])], axis=1)


# --- amplitude (fNIRS hemodynamic) -----------------------------------------------------------------------

@lru_cache(maxsize=8)
def _time_axis(t: int) -> tuple[np.ndarray, float]:
    """Centred time axis + its sum-of-squares (the OLS-slope denominator) — constants for a window length t,
    so cache them. f32 to keep the feature path f32; the sum-of-squares is f64 for a stable denominator."""
    tc = (np.arange(t) - (t - 1) / 2.0).astype(np.float32)
    return tc, float((tc.astype(np.float64) ** 2).sum())


def amplitude_features(X: np.ndarray) -> np.ndarray:
    """Per-channel temporal mean + slope + peak -> `[n, 3*ch]` — the canonical fNIRS feature triple (the
    hemodynamic response's amplitude/shape, exactly what covariance methods discard by centering).
    peak = the extreme deviation (max |value|, signed): HbO rises positive, HbR dips negative."""
    tc, tc_ss = _time_axis(X.shape[2])
    mean = X.mean(axis=2)                                    # response amplitude
    slope = (X * tc).sum(axis=2) / tc_ss                     # response trend (OLS)
    peak = np.take_along_axis(X, np.abs(X).argmax(2)[:, :, None], axis=2)[:, :, 0]   # signed extreme
    return np.concatenate([mean, slope, peak], axis=1)      # native dtype in -> out; LDA needs no f64
