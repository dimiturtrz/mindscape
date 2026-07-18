"""Composable EEG normalization — the `Normalizer` strategy (fit / apply) + the composite that sequences them.

One home for what used to be scattered across the adapter (inline per-channel z-score), a standalone MVNN op,
and the backbone wrappers (an in-forward z-score baked into each foundation model). A `Normalizer` is a
two-method strategy: `fit(X)` learns any state from data (a no-op for the stateless links), `apply(X)`
transforms — split so a *fitted* transform (a per-subject whitener) can be fit on one set of trials and applied
to another (calibration → deployment). A `CompositeNormalization` is itself a `Normalizer`, built from a list
of normalizer objects and fit/applied in order — constructed directly (`CompositeNormalization([Scale(1e4)])`),
no registry.

The fit STRUCTURE (MVNN's per-subject, within-condition grouping of the FIT epochs) lives in a link's **own
constructor**, aligned with the fit data. `apply` additionally takes an optional `groups` (subject id per row
of the APPLIED epochs): a per-subject whitener is a *different matrix per subject*, so selecting the right one
at apply intrinsically needs each row's identity — the fit-data grouping can't stand in for a held-out set the
constructor never saw. Stateless links (z-score, scale) ignore `groups`, so the common path stays data-only.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import override

import numpy as np
from jaxtyping import Float, Int


class Normalizer(ABC):
    """One normalization step. `fit(X)` learns state (no-op for stateless links, so they need not override it);
    `apply(X, groups)` returns the transformed epochs, same shape `[n, ch, t]`. `groups` (subject id per row)
    is used only by per-subject links (MVNN) to pick each row's whitener; stateless links ignore it."""

    def fit(self, X: Float[np.ndarray, "n ch t"]) -> Normalizer:
        """Learn any state from `X`; stateless links keep this no-op. Returns self."""
        return self

    @abstractmethod
    def apply(self, X: Float[np.ndarray, "n ch t"],
              groups: Int[np.ndarray, "n"] | None = None) -> Float[np.ndarray, "n ch t"]:
        """Transform the epochs using the fitted state (or none, for a stateless link). `groups` = subject id
        per row for per-subject links; `None` for the stateless links that don't need it."""


class CompositeNormalization(Normalizer):
    """A `Normalizer` built from an ordered list of normalizer objects, fit/applied in sequence (composite) —
    `CompositeNormalization([Scale(1e4), Mvnn(groups, conditions)])`. An empty list is the identity."""

    def __init__(self, links: list[Normalizer]):
        self.links = links

    @override
    def fit(self, X: Float[np.ndarray, "n ch t"],
            groups: Int[np.ndarray, "n"] | None = None) -> CompositeNormalization:
        """Fit each link on the running output of the ones before it (only transforming when a later link still
        needs the intermediate) — the standard pipeline fit. `groups` (the fit epochs' subject ids) is threaded
        to the intermediate `apply` so a per-subject link mid-chain whitens the fit data by its own grouping."""
        for i, link in enumerate(self.links):
            link.fit(X)
            if i < len(self.links) - 1:
                X = link.apply(X, groups)
        return self

    @override
    def apply(self, X: Float[np.ndarray, "n ch t"],
              groups: Int[np.ndarray, "n"] | None = None) -> Float[np.ndarray, "n ch t"]:
        for link in self.links:
            X = link.apply(X, groups)
        return X
