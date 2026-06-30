"""CSP + LDA — the standard motor-imagery baseline (the quarantine ceiling to compare against).

Common Spatial Patterns (MNE) -> Linear Discriminant Analysis (sklearn). This is the canonical,
no-deep-learning reference for BCI IV-2a (~65–75% within-subject, 4-class); kept in `baselines/`
and separate from the decoders under test, exactly as the siblings isolate nnU-Net/PatchCore.

Interface = the harness contract: `fit(X, y) -> clf`, `score(clf, X) -> probs[n, C]`.
"""
from __future__ import annotations

import numpy as np


def fit(X: np.ndarray, y: np.ndarray, n_components: int = 6):
    from mne.decoding import CSP
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.pipeline import Pipeline

    # CSP expects double-precision [n, ch, t]; LDA gives calibrated-ish predict_proba.
    pipe = Pipeline([
        ("csp", CSP(n_components=n_components, reg="ledoit_wolf", log=True)),
        ("lda", LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")),
    ])
    pipe.fit(X.astype(np.float64), y)
    return pipe


def score(pipe, X: np.ndarray) -> np.ndarray:
    return pipe.predict_proba(X.astype(np.float64))
