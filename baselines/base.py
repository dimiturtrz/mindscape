"""Baseline decoder interface — one small object per classical method, owning its hyperparameters.

Each baseline is a plain object: construct it with its hyperparameters, `.fit(X, y)` returns self (so the
fitted object IS the model the harness carries), and `.score(X)` returns class probabilities `[n, C]`.
This replaces the earlier module-level fit/score function pairs + method-string dispatch: the method IS
the type, its hyperparameters are constructor arguments, and its pipeline lives inside it. The data
configs (EpochCfg/FnirsCfg) already follow this "config/logic lives with the object" shape; this brings
the decoder side in line.

The harness stays decoupled via its `(fit_fn, score_fn)` contract — `models.get_method` adapts a class to
it as `(lambda X, y: Method(**hp).fit(X, y), lambda clf, X: clf.score(X))`.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class Baseline(ABC):
    """A classical decoder: hyperparameters in __init__, fit returns self, score returns probs [n, C]."""

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> "Baseline":
        ...

    @abstractmethod
    def score(self, X: np.ndarray) -> np.ndarray:
        """Class probabilities, shape [n_trials, n_classes]."""
        ...
