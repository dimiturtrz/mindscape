"""Feature extraction — the signal→feature substance the decoders sit on, grouped by feature family so no
one file carries everything. `baselines/` methods are thin: they *call* these + bolt on a classifier, so
there's one extractor implementation reused across methods, modalities, transfer, and the viz.

  covariance.py  EEG geometric — `time_delay_embed` + the manifold transfer transforms
                 (`recenter_covariances`, `scale_to_identity`)
  bandpower.py   EEG oscillatory — `band_powers` (θ/α/β) + `CANONICAL_BANDS`
  amplitude.py   fNIRS hemodynamic — `amplitude_features` (mean/slope/peak)

Import from the package (`from core.features import band_powers`); the submodules are the grouping.
"""
from core.features.amplitude import amplitude_features
from core.features.bandpower import CANONICAL_BANDS, band_powers
from core.features.covariance import recenter_covariances, scale_to_identity, time_delay_embed

__all__ = ["time_delay_embed", "recenter_covariances", "scale_to_identity",
           "band_powers", "CANONICAL_BANDS", "amplitude_features"]
