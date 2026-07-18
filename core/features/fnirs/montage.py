"""fNIRS optode geometry — 2D channel positions on the unit head disk (single-modality)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import scipy.io as sio
from jaxtyping import Float

from core.features.eeg.montage import EegMontage


class FnirsMontage:
    """fNIRS optode geometry — 2D channel positions on the unit head disk (helper folded in as a staticmethod)."""

    @classmethod
    def fnirs_positions(cls, subject_dir: Path) -> Float[np.ndarray, "ch 2"]:
        """2D positions of the 36 fNIRS channels from `mnt_nback.mat` (already a head-normalized 2D projection),
        normalized to the unit disk to match the EEG frame."""
        mnt = sio.loadmat(subject_dir / "mnt_nback.mat", struct_as_record=False, squeeze_me=True)["mnt_nback"]
        pos = np.stack([np.asarray(mnt.x, dtype=float), np.asarray(mnt.y, dtype=float)], axis=1)  # [36, 2]
        return EegMontage.to_unit_disk(pos)
