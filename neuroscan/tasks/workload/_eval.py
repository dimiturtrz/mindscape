"""Shared within/cross-subject CV scorer for the fNIRS decoder-comparison scripts (windowed / clean / glm
ablations). One home for the repeated-seeded StratifiedKFold (within) vs StratifiedGroupKFold (cross-subject)
loop those scripts each re-implemented — the difference between arms is the decoder or the preprocessing, not
the CV plumbing.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from pydantic import BaseModel
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

from baselines.fnirs.features import FnirsLda
from neuroscan.evaluation import metrics


@dataclass
class CvData:
    """The three arrays a CV run needs together: features/epochs `X[n, ...]`, labels `y[n]`, and the per-block
    subject `groups[n]` (used only when the fold split is subject-grouped)."""
    X: np.ndarray
    y: np.ndarray
    groups: np.ndarray


class CvConfig(BaseModel):
    """CV knobs. `grouped` -> StratifiedGroupKFold by subject (cross-subject) vs StratifiedKFold (within);
    `seeds`/`k` = the repeated seeded k-fold; `classes=[a, b]` restricts to those two labels and relabels
    binary (b -> 1) for a per-boundary probe."""
    model_config = {"arbitrary_types_allowed": True}
    grouped: bool
    seeds: tuple[int, ...] = (0,)     # ONE seed by default — a first pass answers most questions (a clear
                                      # null/win shows at 1 seed × k-fold). Add seeds only to CONFIRM a small Δ
                                      # sitting near the fold noise floor (escalate-on-signal, not by reflex).
    k: int = 5                        # k-fold is for VALIDITY (held-out subjects when grouped), not rigor;
                                      # 5 gives a usable SE-of-mean. Not a knob to inflate.
    classes: tuple[int, ...] | None = None


def cv_score(build, data: CvData, config: CvConfig):
    """Mean (acc, sd, kappa) over repeated seeded k-fold. `build` is a `() -> decoder` thunk (None ->
    `FnirsLda`)."""
    X, y, groups = np.asarray(data.X), np.asarray(data.y), np.asarray(data.groups)
    if config.classes is not None:
        m = np.isin(y, config.classes)
        X, y, groups = X[m], (y[m] == config.classes[1]).astype(int), groups[m]
    accs, kaps = [], []
    for seed in config.seeds:
        sp = (StratifiedGroupKFold(config.k, shuffle=True, random_state=seed) if config.grouped
              else StratifiedKFold(config.k, shuffle=True, random_state=seed))
        for tr, te in sp.split(X, y, groups if config.grouped else None):
            clf = (build() if build is not None else FnirsLda()).fit(X[tr], y[tr])
            pred = clf.predict_proba(X[te]).argmax(1)
            accs.append(metrics.accuracy(y[te], pred))
            kaps.append(metrics.kappa(y[te], pred))
    return float(np.mean(accs)), float(np.std(accs)), float(np.mean(kaps))
