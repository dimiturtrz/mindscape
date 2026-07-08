"""Shared within/cross-subject CV scorer for the fNIRS decoder-comparison scripts (windowed / clean / glm
ablations). One home for the repeated-seeded StratifiedKFold (within) vs StratifiedGroupKFold (cross-subject)
loop those scripts each re-implemented — the difference between arms is the decoder or the preprocessing, not
the CV plumbing.
"""
from __future__ import annotations

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

from baselines.fnirs.features import FnirsLda
from neuroscan.evaluation import metrics


def cv_score(build, X, y, groups, *, grouped: bool, seeds=(0, 1, 2), k: int = 5, classes=None):
    """Mean (acc, sd, kappa) over repeated seeded k-fold. `grouped` -> StratifiedGroupKFold by subject
    (cross-subject); else StratifiedKFold (within, subjects in train+test). `build` is a `() -> decoder`
    thunk (None -> `FnirsLda`). `classes=[a, b]` restricts to those two labels and relabels binary
    (b -> 1) for a per-boundary probe."""
    X, y, groups = np.asarray(X), np.asarray(y), np.asarray(groups)
    if classes is not None:
        m = np.isin(y, classes)
        X, y, groups = X[m], (y[m] == classes[1]).astype(int), groups[m]
    accs, kaps = [], []
    for seed in seeds:
        sp = (StratifiedGroupKFold(k, shuffle=True, random_state=seed) if grouped
              else StratifiedKFold(k, shuffle=True, random_state=seed))
        for tr, te in sp.split(X, y, groups if grouped else None):
            clf = (build() if build is not None else FnirsLda()).fit(X[tr], y[tr])
            pred = clf.predict_proba(X[te]).argmax(1)
            accs.append(metrics.accuracy(y[te], pred))
            kaps.append(metrics.kappa(y[te], pred))
    return float(np.mean(accs)), float(np.std(accs)), float(np.mean(kaps))
