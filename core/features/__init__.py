"""Feature extraction — the signal→feature substance the decoders sit on, organized by MODALITY (matching
`core/data/{eeg,fnirs}` and `baselines/{eeg,fnirs,fusion}`), then by feature family within.

  eeg/     bandpower (θ/α/β `BandPower`) · covariance (`Covariance`, transfer transforms) · csd · montage
  fnirs/   amplitude (`Amplitude`) · bank (`DescriptorBank` descriptor bank) · chromophore (CBSI) · montage
  fusion/  coupling (derived neurovascular lag) · series (envelope+lag rep) · camera (⚠️ lossy viz raster)

Import the feature classes from the package (`from core.features import BandPower`); the cross-modal fusion
API lives under the subpackage (`from core.features import fusion`).
"""
import numpy as np

from core.features.eeg.bandpower import CANONICAL_BANDS, BandPower
from core.features.eeg.covariance import Covariance
from core.features.fnirs.amplitude import Amplitude
from core.features.fnirs.bank import FNIRS_FEATURE_FNS, DescriptorBank, WeightedFamilyScaler


class SubjectNorm:
    """Per-subject feature standardization — the unsupervised cross-subject offset/scale fix (helper folded in
    as a staticmethod, public name kept)."""

    @staticmethod
    def zscore_per_subject(F: np.ndarray, groups: np.ndarray) -> np.ndarray:
        """Standardize each feature within each subject by ITS OWN mean/std — unsupervised, so it applies to a
        held-out test subject too (it standardizes by its own stats). Removes the subject-specific offset/scale
        that sinks cross-subject band-power. `F[n, d]` features, `groups[n]` subject id per row -> z-scored `F`."""
        out = np.empty_like(F)
        for subject in np.unique(groups):
            mask = groups == subject
            mu, sd = F[mask].mean(0), F[mask].std(0)
            out[mask] = (F[mask] - mu) / (sd + 1e-6)
        return out


__all__ = [
    "CANONICAL_BANDS",
    "FNIRS_FEATURE_FNS",
    "Amplitude",
    "BandPower",
    "Covariance",
    "DescriptorBank",
    "SubjectNorm",
    "WeightedFamilyScaler",
]
