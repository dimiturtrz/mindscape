"""Feature extraction — the signal→feature substance the decoders sit on, organized by MODALITY (matching
`core/data/{eeg,fnirs}` and `baselines/{eeg,fnirs,fusion}`), then by feature family within.

  eeg/     bandpower (θ/α/β `band_powers`) · covariance (`time_delay_embed`, transfer transforms) · csd · montage
  fnirs/   amplitude (`amplitude_features`) · bank (`extract_bank` descriptor bank) · chromophore (CBSI) · montage
  fusion/  coupling (derived neurovascular lag) · series (envelope+lag rep) · camera (⚠️ lossy viz raster)

Import the flat feature functions from the package (`from core.features import band_powers`); the cross-modal
fusion API lives under the subpackage (`from core.features import fusion`).
"""
from core.features.eeg.bandpower import CANONICAL_BANDS, band_powers
from core.features.eeg.covariance import recenter_covariances, scale_to_identity, time_delay_embed
from core.features.fnirs.amplitude import amplitude_features
from core.features.fnirs.bank import (
    FNIRS_FEATURE_FNS, WeightedFamilyScaler, extract_bank, family_names)

__all__ = ["time_delay_embed", "recenter_covariances", "scale_to_identity",
           "band_powers", "CANONICAL_BANDS", "amplitude_features",
           "extract_bank", "family_names", "FNIRS_FEATURE_FNS", "WeightedFamilyScaler"]
