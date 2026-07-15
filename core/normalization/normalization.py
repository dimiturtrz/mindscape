"""Composable EEG normalization — the `Normalizer` strategy interface + the composite that sequences them.

One home for what used to be scattered across the adapter (inline per-channel z-score), a standalone MVNN op,
and the backbone wrappers (an in-forward z-score baked into each foundation model). A `Normalizer` is a single
`apply(X, ctx) -> X` step; a `CompositeNormalization` is itself a `Normalizer` built from a list of normalizer
objects, applied in order — constructed directly (`CompositeNormalization([Scale(1e4)])`), no registry. `ctx`
carries the optional per-trial structure a data-fitted link needs — subject id (`groups`, for a per-subject
whitener) and stimulus id (`conditions`, for MVNN's within-condition noise) — which the stateless links
(z-score, scale) simply ignore. There is no separate `fit`: every link fits on the data it is handed (MVNN
estimates a subject's whitener from that subject's own trials, unsupervised), so fit-and-apply is one call —
the honest shape for a pipeline that never carries normalizer state across splits.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
from jaxtyping import Float, Int


@dataclass
class NormContext:
    """Optional per-trial structure a normalizer may use. `groups` = subject id per trial (a per-subject link
    fits one transform per group); `conditions` = stimulus/image id per trial (MVNN's within-condition noise).
    Stateless links (z-score, scale) ignore both."""
    groups: Int[np.ndarray, "n"] | None = None
    conditions: Int[np.ndarray, "n"] | None = None


class Normalizer(ABC):
    """One normalization step: `apply(X, ctx) -> X`, shape-preserving `[n, ch, t]`. Fits on the data it is
    handed (no carried state), so it is safe to apply per split."""

    @abstractmethod
    def apply(self, X: Float[np.ndarray, "n ch t"], ctx: NormContext) -> Float[np.ndarray, "n ch t"]:
        """Return the normalized epochs (same shape). Data-fitted links read `ctx`; stateless links ignore it."""


class CompositeNormalization(Normalizer):
    """A `Normalizer` built from an ordered list of normalizer objects, applied in sequence (composite pattern)
    — `CompositeNormalization([Scale(1e4), Mvnn()])`. An empty list is the identity."""

    def __init__(self, links: list[Normalizer]):
        self.links = links

    def apply(self, X: Float[np.ndarray, "n ch t"], ctx: NormContext) -> Float[np.ndarray, "n ch t"]:
        for link in self.links:
            X = link.apply(X, ctx)
        return X
