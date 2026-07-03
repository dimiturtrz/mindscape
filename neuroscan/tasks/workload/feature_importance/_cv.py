"""Shared subject-grouped cross-validation plumbing for the fNIRS feature-importance studies.

The one fold-generation contract every study here obeys: repeated seeded StratifiedGroupKFold, grouped by
subject so whole subjects fall on one side of each split (no within-subject leakage). The studies differ in
what they put *inside* the fold (weighted-family LDA / a torch head / a fixed recipe) — that difference is
deliberate, so this consolidates only the fold loop they genuinely share, not the classifier.
"""
from __future__ import annotations

from collections.abc import Iterator

import numpy as np


def grouped_folds(F: np.ndarray, y: np.ndarray, groups: np.ndarray,
                  seeds, k: int) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Yield `(train_idx, test_idx)` for each fold of each seeded StratifiedGroupKFold pass. Subject-grouped;
    repeated over `seeds` to average out the split noise in the CV estimate."""
    from sklearn.model_selection import StratifiedGroupKFold
    for seed in seeds:
        sgkf = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=seed)
        yield from sgkf.split(F, y, groups)
