"""Feature extraction — the signal→feature substance the decoders sit on, grouped by feature family so no
one file carries everything. `baselines/` methods are thin: they *call* these + bolt on a classifier, so
there's one extractor implementation reused across methods, modalities, transfer, and the viz.

  covariance.py  EEG geometric — `time_delay_embed` + the manifold transfer transforms
                 (`recenter_covariances`, `scale_to_identity`)
  bandpower.py   EEG oscillatory — `band_powers` (θ/α/β) + `CANONICAL_BANDS`
  amplitude.py   fNIRS hemodynamic — `amplitude_features` (mean/slope/peak)
  fnirs_bank.py  fNIRS wide descriptor bank — `extract_bank` + `family_names` + `WeightedFamilyScaler`
                 (the weighted-feature importance search)

Import from the package (`from core.features import band_powers`); the submodules are the grouping.
"""
from core.features.amplitude import amplitude_features
from core.features.bandpower import CANONICAL_BANDS, band_powers
from core.features.covariance import recenter_covariances, scale_to_identity, time_delay_embed
from core.features.fnirs_bank import (
    FNIRS_FEATURE_FNS, WeightedFamilyScaler, extract_bank, family_names)

__all__ = ["time_delay_embed", "recenter_covariances", "scale_to_identity",
           "band_powers", "CANONICAL_BANDS", "amplitude_features",
           "extract_bank", "family_names", "FNIRS_FEATURE_FNS", "WeightedFamilyScaler"]
