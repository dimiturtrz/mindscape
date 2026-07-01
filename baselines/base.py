"""Classical baseline decoders — one small object per method, owning its hyperparameters.

Each baseline is a plain object: construct it with its hyperparameters, `.fit(X, y)` returns self (so the
fitted object IS the model the harness carries), and `.predict_proba(X)` returns class probabilities
`[n, C]`. This replaces the earlier module-level fit/score function pairs + method-string dispatch: the
method IS the type, its hyperparameters are constructor arguments, and its pipeline lives inside it.

`Baseline` is the shared ABC for the *classical* side (it lets variants share fit/predict_proba — e.g.
_RiemannBaseline). It structurally satisfies the general `neuroscan.models.base.Decoder` contract (same
`fit` + `predict_proba`) that the braindecode nets also satisfy, so the harness treats every method the
same way — but we don't import Decoder here, keeping baselines/ standalone. The data configs
(EpochCfg/FnirsCfg) already follow this "config/logic lives with the object" shape; this is the decoder
side in line.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class Baseline(ABC):
    """A classical decoder: hyperparameters in __init__, fit returns self, predict_proba returns [n, C]."""

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> "Baseline":
        ...

    @abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Class probabilities, shape [n_trials, n_classes]."""
        ...
