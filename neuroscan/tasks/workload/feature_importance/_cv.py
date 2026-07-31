"""Shared subject-grouped cross-validation plumbing for the fNIRS feature-importance studies.

The one fold-generation contract every study here obeys: repeated seeded StratifiedGroupKFold, grouped by
subject so whole subjects fall on one side of each split (no within-subject leakage). The studies differ in
what they put *inside* the fold (weighted-family LDA / a torch head / a fixed recipe) — that difference is
deliberate, so this consolidates only the fold loop they genuinely share, not the classifier.
"""
from __future__ import annotations

from collections.abc import Iterator, Sequence

import numpy as np
from jaxtyping import Float, Int
from sklearn.model_selection import StratifiedGroupKFold


class Cv:
    """Shared subject-grouped fold generation for the fNIRS feature-importance studies (free helper folded in
    as a staticmethod, public name kept)."""

    @staticmethod
    def grouped_folds(F: Float[np.ndarray, "n f"], y: Int[np.ndarray, "n"], groups: Int[np.ndarray, "n"],
                      seeds: Sequence[int], k: int) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        """Yield `(train_idx, test_idx)` for each fold of each seeded StratifiedGroupKFold pass. Subject-grouped;
        repeated over `seeds` to average out the split noise in the CV estimate."""
        for seed in seeds:
            sgkf = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=seed)
            yield from sgkf.split(F, y, groups)
