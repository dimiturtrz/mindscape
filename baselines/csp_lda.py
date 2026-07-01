"""CSP + LDA — the standard motor-imagery baseline (the quarantine ceiling to compare against).

Common Spatial Patterns (MNE) -> Linear Discriminant Analysis (sklearn). This is the canonical,
no-deep-learning reference for BCI IV-2a (~65–75% within-subject, 4-class); kept in `baselines/`
and separate from the decoders under test, exactly as the siblings isolate nnU-Net/PatchCore.

A `Baseline` object (see baselines/base.py): hyperparameters in __init__, `.fit(X, y) -> self`,
`.score(X) -> probs`. Module-level `fit`/`score` remain as back-compat shims.
"""
from __future__ import annotations

import numpy as np

from baselines.base import Baseline


class CspLda(Baseline):
    """Common Spatial Patterns -> LDA. CSP expects double-precision [n, ch, t]; LDA gives
    calibrated-ish predict_proba. The fitted pipeline lives on `self.pipe_`."""

    def __init__(self, n_components: int = 6):
        self.n_components = n_components

    def _build(self):
        from mne.decoding import CSP
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
        from sklearn.pipeline import Pipeline
        return Pipeline([
            ("csp", CSP(n_components=self.n_components, reg="ledoit_wolf", log=True)),
            ("lda", LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")),
        ])

    def fit(self, X, y):
        self.pipe_ = self._build()
        self.pipe_.fit(np.asarray(X, dtype=np.float64), y)   # MNE-CSP requires float64
        return self

    def score(self, X):
        return self.pipe_.predict_proba(np.asarray(X, dtype=np.float64))


def fit(X: np.ndarray, y: np.ndarray, n_components: int = 6) -> Baseline:
    """Back-compat shim — prefer `CspLda(...).fit(X, y)`."""
    return CspLda(n_components=n_components).fit(X, y)


def score(clf: Baseline, X: np.ndarray) -> np.ndarray:
    return clf.score(X)
