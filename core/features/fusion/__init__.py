"""EEG+fNIRS fusion features (cross-modal).

Split by role:
  coupling.py  the derived neurovascular offset+decay (EEG↔fNIRS timing bridge)
  series.py    the pre-raster lag-aligned envelope + CBSI representation
  camera.py    ⚠️ the LOSSY viz rasterization (grid maps) + decode-negatives (build_tensor, fused_node_series)

This `__init__` is a convenience facade re-exporting the pipeline API (incl. the single-modality prep it uses —
`EegMontage`/`CSD` live in `core.features.eeg`, `FnirsMontage`/`Chromophore` in `core.features.fnirs`;
re-exported here for the fusion workflow). Import as `from core.features import fusion`.
"""
from core.features.eeg.csd import CSD
from core.features.eeg.montage import EegMontage
from core.features.fnirs.chromophore import Chromophore
from core.features.fnirs.montage import FnirsMontage
from core.features.fusion.camera import BrainCamera, PairedModalities
from core.features.fusion.coupling import _HRF_WIDTH, Coupling
from core.features.fusion.series import _BANDS, Series, SeriesConfig

__all__ = ["EegMontage", "FnirsMontage", "CSD", "Chromophore", "Coupling",
           "Series", "BrainCamera", "PairedModalities", "SeriesConfig"]
