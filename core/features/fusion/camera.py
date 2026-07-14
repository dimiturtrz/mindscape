"""Brain-camera rasterization — the cross-modal series → 2D-surface × time maps (the VIZ representation).

⚠️ This is the LOSSY viz branch: band-power ENVELOPES rasterized onto a head grid. Great for the eye, a poor
decoder representation (it drops phase, merges bands, interpolates between sparse sensors). `build_tensor` and
`fused_node_series` exist as documented DECODE NEGATIVES — decoding the viz branch underperforms raw-covariance
EEG. For real decoding, use raw/multiband EEG covariance + a boundary-aware fNIRS combiner, not this.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from jaxtyping import Float
from scipy.spatial import Delaunay

from core.features.fusion.series import _BANDS, Series, SeriesConfig


@dataclass
class PairedModalities:
    """Block-aligned EEG + fNIRS epochs plus each modality's 2D sensor positions — the four things every
    brain-camera rasterizer needs together (was the loose `Xe, Xf, pos_e, pos_f` argument run)."""
    eeg: np.ndarray
    fnirs: np.ndarray
    pos_eeg: np.ndarray
    pos_fnirs: np.ndarray


class BrainCamera:
    """Brain-camera rasterization — the cross-modal series → 2D-surface × time maps (the VIZ representation;
    free helpers folded in as staticmethods, public names kept). ⚠️ LOSSY viz branch / DECODE NEGATIVE."""

    @staticmethod
    def _bary(pos: np.ndarray, pts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Barycentric linear-interpolation weight matrix `W[len(pts), n_ch]` mapping channel values at `pos` to
        arbitrary query points `pts` (Delaunay triangulation, computed once — every frame is then a matmul), plus
        the distance `d[len(pts)]` from each query point to its nearest sensor. Points outside the hull get zero
        weight. Channels with non-finite positions are dropped from the triangulation (weight column stays zero)."""
        ok = np.isfinite(pos).all(1)
        idx = np.where(ok)[0]
        p = pos[ok]
        tri = Delaunay(p)
        simp = tri.find_simplex(pts)                                        # [-1] outside hull
        W = np.zeros((len(pts), len(pos)), dtype=np.float32)
        inside = simp >= 0
        T = tri.transform[simp[inside]]
        b = np.einsum("nij,nj->ni", T[:, :2, :], pts[inside] - T[:, 2, :])  # first 2 barycentric coords
        bary = np.concatenate([b, 1 - b.sum(1, keepdims=True)], axis=1)     # [Nin, 3]
        verts = idx[tri.simplices[simp[inside]]]                            # original channel indices
        rows = np.where(inside)[0]
        for j in range(3):
            W[rows, verts[:, j]] = bary[:, j]
        d = np.sqrt(((pts[:, None, :] - p[None, :, :]) ** 2).sum(-1)).min(1)
        return W, d

    @staticmethod
    def _interp_weights(pos: np.ndarray, grid: int) -> tuple[np.ndarray, np.ndarray]:
        """Barycentric weights `W[grid², n_ch]` onto the head grid + a validity `mask[grid²]` (pixel within ~1.5
        grid-spacings of a sensor)."""
        gx = np.linspace(-1, 1, grid)
        gxx, gyy = np.meshgrid(gx, gx)
        pts = np.stack([gxx.ravel(), gyy.ravel()], axis=1)                  # [G², 2]
        W, d = BrainCamera._bary(pos, pts)
        mask = (d <= (2.0 / grid) * 1.5).astype(np.float32)                 # [G²]
        return W, mask

    @staticmethod
    def _apply(vals: np.ndarray, W: np.ndarray, grid: int) -> np.ndarray:
        """`vals[n, ch, t]` × `W[grid², ch]` -> `maps[n, grid, grid, t]` (one einsum, all frames at once)."""
        m = np.einsum("nct,gc->ngt", vals, W)                              # [n, G², t]
        n, _, t = m.shape
        return m.reshape(n, grid, grid, t)

    @staticmethod
    def coverage_map(pos_e: Float[np.ndarray, "ch_e 2"], pos_f: Float[np.ndarray, "ch_f 2"],
                     grid: int) -> Float[np.ndarray, "grid grid"]:
        """Locality-coverage confidence `[grid, grid]` — EEG↔fNIRS coupling is LOCAL, so a pixel's joint signal is
        only trustworthy where a co-located pair exists (near BOTH an EEG and an fNIRS sensor). Gaussian falloff to
        the nearest sensor of each modality; product dims where either modality has no nearby sensor."""
        gx = np.linspace(-1, 1, grid)
        gxx, gyy = np.meshgrid(gx, gx)
        pts = np.stack([gxx.ravel(), gyy.ravel()], axis=1)                  # [G², 2]
        s2 = (0.20 ** 2) * 2                                                # co-sampling radius² (unit-disk units)

        def near(pos):
            p = pos[np.isfinite(pos).all(1)]
            return np.exp(-((pts[:, None, :] - p[None, :, :]) ** 2).sum(-1).min(1) / s2)
        return (near(pos_e) * near(pos_f)).reshape(grid, grid).astype(np.float32)

    @staticmethod
    def build_tensor(paired: PairedModalities, *, grid=16, series: SeriesConfig | None = None):
        """Paired EEG+fNIRS -> the brain-camera tensor `X[n, C, grid, grid, T]`, C = [θ, α, β, CBSI-neural,
        coverage]. ⚠️ VIZ representation / DECODE NEGATIVE — the rasterized band-envelope maps underperform raw
        EEG covariance as a decoder input. `series` -> channel_series timing (fs/fps/lag/window)."""
        eeg, neural, _, _ = Series.channel_series(paired.eeg, paired.fnirs, series)
        W_e, _ = BrainCamera._interp_weights(paired.pos_eeg, grid)
        W_f, _ = BrainCamera._interp_weights(paired.pos_fnirs, grid)
        eeg_maps = [BrainCamera._zscore(BrainCamera._apply(eeg[b], W_e, grid)) for b in _BANDS]   # strength layers
        neural_map = BrainCamera._zscore(BrainCamera._apply(neural, W_f, grid))                   # origin + spread
        cov = BrainCamera.coverage_map(paired.pos_eeg, paired.pos_fnirs, grid)                    # locality gate
        stack = [*eeg_maps, neural_map, BrainCamera._broadcast(cov, eeg_maps[0].shape)]
        return np.stack(stack, axis=1).astype(np.float32)                              # [n, C=5, grid, grid, T]

    @staticmethod
    def fused_node_series(paired: PairedModalities, *, band="sum", fnirs=True,
                          series: SeriesConfig | None = None):
        """The FUSION-only signal collapsed to EEG channel format `[n, n_e, T]`. At each EEG node the joint =
        SIGN-ALIGNED EEG envelope × co-located fNIRS CBSI × locality COVERAGE. The EEG term is centered and flipped
        by the derived coupling sign (bd 060) so the physically-correct quadrant — ERD power DROP with an HbO RISE —
        reads as positive activation, instead of the old raw-envelope × CBSI that lit power-up + blood-up.
        ⚠️ Still DECODE NEGATIVE after the sign fix: re-measured 0.386 vs EEG-only 0.580 (Δ −0.194, fair null).
        The ERD sign was not why output-space fusion fails — the MULTIPLICATIVE join is (clean EEG × weak fNIRS,
        phase dropped). Kept physically-correct + documented; source-space/generative fusion is the real lever
        (Stage-4). `fnirs=False` returns the EEG strength alone (isolates the rep from the combiner). `series` ->
        channel_series."""
        eeg, neural, _, coupling = Series.channel_series(paired.eeg, paired.fnirs, series)
        pos_e, pos_f = paired.pos_eeg, paired.pos_fnirs
        ok = np.isfinite(pos_e).all(1)
        strength = sum(eeg[b] for b in _BANDS) if band == "sum" else eeg[band]      # [n, n_e, T], envelopes ≥ 0
        strength = strength[:, ok, :]
        Wq, _ = BrainCamera._bary(pos_f, pos_e[ok])                                # fNIRS -> EEG-node interp [n_e, n_f]
        neural_e = np.einsum("nct,gc->ngt", neural, Wq)                             # CBSI at the EEG nodes [n, n_e, T]
        if not fnirs:                                                              # EEG-representation-ONLY ablation:
            return strength.astype(np.float32), coupling                          # isolate the rep from the combiner
        s2 = (0.20 ** 2) * 2
        pf = pos_f[np.isfinite(pos_f).all(1)]
        # dist² each EEG node -> nearest fNIRS
        d2 = ((pos_e[ok][:, None, :] - pf[None, :, :]) ** 2).sum(-1).min(1)
        cov_e = np.exp(-d2 / s2)                                                    # locality coverage per node [n_e]
        # ERD physics (bd 060): EEG band-power ANTI-correlates with HbO — cortical activation is power DOWN +
        # blood UP, so the raw envelope (≥0) × CBSI lights the WRONG quadrant (power-up + blood-up). Align the EEG
        # term to the DERIVED coupling sign: center the envelope in time and flip by sign(beta) so a power DROP
        # coinciding with an HbO RISE reads as positive fused activation. beta < 0 for ERD; NaN (fixed-lag
        # override — no derived sign) falls back to +1.
        sign = float(np.sign(coupling["beta"])) if np.isfinite(coupling["beta"]) else 1.0
        strength = sign * (strength - strength.mean(axis=-1, keepdims=True))
        joint = strength * neural_e * cov_e[None, :, None]                         # [n, n_e, T]
        return joint.astype(np.float32), coupling

    @staticmethod
    def _zscore(m: np.ndarray) -> np.ndarray:
        """Z-score a map block over space+time per sample (comparability across bands/modalities)."""
        mu = m.mean(axis=(1, 2, 3), keepdims=True)
        sd = m.std(axis=(1, 2, 3), keepdims=True) + 1e-6
        return (m - mu) / sd

    @staticmethod
    def _broadcast(mask2d: np.ndarray, like_shape) -> np.ndarray:
        """A `[grid,grid]` mask -> `[n, grid, grid, T]` matching a map block."""
        n, g, _, t = like_shape
        return np.broadcast_to(mask2d[None, :, :, None], (n, g, g, t)).astype(np.float32)
