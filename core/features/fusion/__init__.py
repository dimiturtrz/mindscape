"""EEG+fNIRS fusion features (cross-modal).

Split by role:
  coupling.py  the derived neurovascular offset+decay (EEG↔fNIRS timing bridge)
  series.py    the pre-raster lag-aligned envelope + CBSI representation
  camera.py    ⚠️ the LOSSY viz rasterization (grid maps) + decode-negatives (build_tensor, fused_node_series)

This `__init__` is a convenience facade re-exporting the pipeline API (incl. the single-modality prep it uses —
`eeg_positions`/`csd_transform` live in `core.features.eeg`, `fnirs_positions`/`cbsi_neural` in
`core.features.fnirs`; re-exported here for the fusion workflow). Import as `from core.features import fusion`.
"""
from core.features.eeg.csd import csd_transform
from core.features.eeg.montage import eeg_positions
from core.features.fnirs.chromophore import cbsi_neural
from core.features.fnirs.montage import fnirs_positions
from core.features.fusion.camera import (
    build_tensor, coverage_map, fused_node_series, _apply, _bary, _broadcast, _interp_weights, _zscore)
from core.features.fusion.coupling import estimate_coupling, _HRF_WIDTH, _gamma_kernel
from core.features.fusion.series import channel_series, _band_env, _BANDS, _resample_time

__all__ = ["eeg_positions", "fnirs_positions", "csd_transform", "cbsi_neural", "estimate_coupling",
           "channel_series", "coverage_map", "build_tensor", "fused_node_series"]
