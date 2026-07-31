"""The decoder contract — the structural interface every method (classical baseline or braindecode net)
satisfies, so the eval harness can score any of them the same way.

A `Decoder` exposes `predict_proba(X) -> probs[n, C]`; that single scorer is what the harness calls, whether
the object is a fitted sklearn-style baseline or a wrapped torch net. Structural (`Protocol`), so nothing has
to inherit it — a class merely needs the method.
"""
from __future__ import annotations

from typing import Protocol

import numpy as np
from jaxtyping import Float


class Decoder(Protocol):
    """A fitted decoder: `predict_proba(X) -> probs[n, C]`. The one scorer the eval harness calls."""

    def predict_proba(self, X: Float[np.ndarray, "n ..."]) -> Float[np.ndarray, "n k"]: ...
