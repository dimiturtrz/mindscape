"""Composable EEG normalization — the `Normalizer` strategy (fit / apply) + the composite that sequences them.

One home for what used to be scattered across the adapter (inline per-channel z-score), a standalone MVNN op,
and the backbone wrappers (an in-forward z-score baked into each foundation model). A `Normalizer` is a
two-method strategy: `fit(X)` learns any state from data (a no-op for the stateless links), `apply(X)`
transforms — split so a *fitted* transform (a per-subject whitener) can be fit on one set of trials and applied
to another (calibration → deployment). A `CompositeNormalization` is itself a `Normalizer`, built from a list
of normalizer objects and fit/applied in order — constructed directly (`CompositeNormalization([Scale(1e4)])`),
no registry.

The interface is deliberately data-only: `fit(X)`/`apply(X)` take just the epochs. A link that needs extra
structure to fit (MVNN's per-subject, within-condition noise) takes that structure in its **own constructor**,
so it never leaks into the general interface.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from jaxtyping import Float


class Normalizer(ABC):
    """One normalization step. `fit(X)` learns state (no-op for stateless links, so they need not override it);
    `apply(X)` returns the transformed epochs, same shape `[n, ch, t]`."""

    def fit(self, X: Float[np.ndarray, "n ch t"]) -> Normalizer:
        """Learn any state from `X`; stateless links keep this no-op. Returns self."""
        return self

    @abstractmethod
    def apply(self, X: Float[np.ndarray, "n ch t"]) -> Float[np.ndarray, "n ch t"]:
        """Transform the epochs using the fitted state (or none, for a stateless link)."""


class CompositeNormalization(Normalizer):
    """A `Normalizer` built from an ordered list of normalizer objects, fit/applied in sequence (composite) —
    `CompositeNormalization([Scale(1e4), Mvnn(groups, conditions)])`. An empty list is the identity."""

    def __init__(self, links: list[Normalizer]):
        self.links = links

    def fit(self, X: Float[np.ndarray, "n ch t"]) -> CompositeNormalization:
        """Fit each link on the running output of the ones before it (only transforming when a later link still
        needs the intermediate) — the standard pipeline fit."""
        for i, link in enumerate(self.links):
            link.fit(X)
            if i < len(self.links) - 1:
                X = link.apply(X)
        return self

    def apply(self, X: Float[np.ndarray, "n ch t"]) -> Float[np.ndarray, "n ch t"]:
        for link in self.links:
            X = link.apply(X)
        return X
