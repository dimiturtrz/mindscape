"""The one decoder contract the harness runs against — shared by classical baselines AND braindecode nets.

A `Decoder` is anything that fits and returns per-trial class probabilities `[n, C]`. It's a structural
`typing.Protocol`, so implementations don't import or inherit it — they satisfy it by shape:

  - the classical baselines (`baselines/base.Baseline` -> CspLda / TangentSpace / Mdm / Acm / FnirsLda)
  - the deep decoders (`neuroscan/models/decoders.BraindecodeClf`)

both already expose `fit(X, y) -> self` and `predict_proba(X) -> probs`. Keeping the contract structural
also preserves the dependency direction (baselines/ stays standalone; it need not know about neuroscan).
`models.get_method` adapts any Decoder to the harness `(fit_fn, score_fn)` pair with a single
`predict_proba` scorer for every method.
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class Decoder(Protocol):
    def fit(self, X: np.ndarray, y: np.ndarray) -> "Decoder":
        ...

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Per-trial class probabilities, shape [n_trials, n_classes]."""
        ...
