"""Input transforms for the decoders — standardizer construction + sliding-window crops.

Extracted from decoders.py so the trainer stays about *training*. The standardizers themselves live in
`standardizers.py` (one interface: `fit(X) -> self`, `__call__(X) -> X'`); this builds one by name and cuts
crops.
"""
from __future__ import annotations

import numpy as np
from jaxtyping import Float

from neuroscan.models.standardizers import STANDARDIZERS, ZScore


class Transforms:
    @staticmethod
    def standardizer(kind: str):
        """Build a standardizer by name; unknown -> z-score."""
        return STANDARDIZERS.get(kind, ZScore)()

    @staticmethod
    def crops(X: Float[np.ndarray, "n ch t"], crop_len: int, n_crops: int):
        """Cut each trial [.,ch,T] into `n_crops` evenly-spaced windows of length `crop_len`.
        Returns (Xc [N*n_crops, ch, crop_len], trial_index [N*n_crops])."""
        T = X.shape[2]
        starts = np.linspace(0, T - crop_len, n_crops).round().astype(int)
        idx = starts[:, None] + np.arange(crop_len)             # [n_crops, crop_len] sample indices per crop
        Xc = X[:, :, idx].transpose(2, 0, 1, 3).reshape(-1, X.shape[1], crop_len)   # [n_crops*N, ch, crop_len]
        tidx = np.tile(np.arange(len(X)), n_crops)              # trial each crop-row came from
        return Xc, tidx
