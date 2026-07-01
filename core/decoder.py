"""The one decoder contract the whole pipeline speaks — shared by every model family and the harness.

A `Decoder` is anything that fits and returns per-trial class probabilities `[n, C]`. It's a structural
`typing.Protocol`, so implementations don't import or inherit it — they satisfy it by shape:

  - the classical baselines (`baselines/base.Baseline` -> CspLda / TangentSpace / Mdm / Acm / FnirsLda)
  - the deep decoders (`neuroscan/models/decoders.BraindecodeClf`)

both expose `fit(X, y) -> self` and `predict_proba(X) -> probs`. It lives in `core` (not inside one of the
two implementer trees) because it's the neutral vocabulary both sit above — the same layer as
`core/export_onnx`, which already consumes a trained decoder. Keeping it structural also preserves the
dependency direction: `baselines/` stays standalone, needing to know nothing about `core` or `neuroscan`.
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
