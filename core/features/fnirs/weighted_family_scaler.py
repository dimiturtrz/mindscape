"""Per-family weighted standardiser for the fNIRS descriptor bank — the sklearn transformer the weighted
feature-importance search fits (bd; see `bank.py` for the `extract_bank` column→family map it consumes).
"""
from __future__ import annotations

import numpy as np
from jaxtyping import Float, Shaped


class WeightedFamilyScaler:
    """Per-feature standardisation (fit on TRAIN) followed by a per-family weight. sklearn transformer
    contract (`fit`/`transform`), so it sits in a Pipeline and standardises on train only — no leakage.

    The weight is applied AFTER standardisation on purpose: a weight applied before is exactly divided back
    out by the per-feature std, so it would have no effect. `weights` maps family name -> w (missing = 1.0);
    w≈0 effectively drops the family. `fam` is the column→family map from `extract_bank`."""

    def __init__(self, fam: Shaped[np.ndarray, "d"], weights: dict[str, float]):
        self.fam = fam
        self.weights = weights

    def fit(self, X: Float[np.ndarray, "n d"], y: None = None) -> WeightedFamilyScaler:
        X = np.asarray(X, dtype=np.float64)
        self.mean_ = X.mean(axis=0)
        self.std_ = X.std(axis=0) + 1e-8                                   # guard zero-variance columns
        self.w_ = np.array([float(self.weights.get(f, 1.0)) for f in self.fam])
        return self

    def transform(self, X: Float[np.ndarray, "n d"]) -> Float[np.ndarray, "n d"]:
        X = np.asarray(X, dtype=np.float64)
        return ((X - self.mean_) / self.std_) * self.w_                    # standardise, THEN weight
