"""EEG scalp geometry — 2D electrode positions on the unit head disk (single-modality; the fusion pipeline and
any EEG-alone spatial method reuse this)."""
from __future__ import annotations

import mne
import numpy as np
from jaxtyping import Float


class EegMontage:
    """EEG scalp geometry — 2D electrode positions on the unit head disk (helpers folded in as staticmethods)."""

    @staticmethod
    def eeg_positions(ch_names: list[str]) -> Float[np.ndarray, "ch 2"]:
        """2D scalp positions for EEG channels via the MNE standard 10-05 montage, normalized to the unit disk.
        Returns `[n_ch, 2]` in the same head-disk convention as `fnirs_positions`."""
        m = mne.channels.make_standard_montage("standard_1005").get_positions()["ch_pos"]
        pos = np.array([m[c][:2] if c in m else (np.nan, np.nan) for c in ch_names], dtype=float)  # x,y (drop z)
        return EegMontage.to_unit_disk(pos)

    @staticmethod
    def to_unit_disk(pos: Float[np.ndarray, "ch 2"]) -> Float[np.ndarray, "ch 2"]:
        """Center + scale a 2D montage so its channels fit the unit disk (radius ≤ 1) — a common head frame."""
        p = pos - np.nanmean(pos, axis=0)
        r = np.nanmax(np.hypot(p[:, 0], p[:, 1]))
        return p / (r + 1e-9)

    @staticmethod
    def channel_laplacian(positions: Float[np.ndarray, "ch 2"], sigma: float = 0.2) -> Float[np.ndarray, "ch ch"]:
        """Graph Laplacian L = D − A over channels — the spatial-smoothness prior substrate (bd 1x0). Edge weight
        A_ij = exp(−‖p_i − p_j‖² / 2σ²) (Gaussian RBF on the unit-disk montage, zero self-loop), so the quadratic
        form fᵀLf = ½ Σ_ij A_ij (f_i − f_j)² penalizes differences between neighbouring electrodes. σ is the
        neighbourhood width in unit-disk radii (matches the frozen-head topo RBF, bd nm5). Off-montage NaN
        positions are pushed far so they pick up ~zero edge weight — excluded from the prior, not mis-placed."""
        p = np.nan_to_num(positions, nan=1e6)
        d2 = ((p[:, None, :] - p[None, :, :]) ** 2).sum(-1)
        adj = np.exp(-d2 / (2 * sigma ** 2))
        np.fill_diagonal(adj, 0.0)
        return (np.diag(adj.sum(1)) - adj).astype(np.float32)
