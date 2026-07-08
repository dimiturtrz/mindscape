"""Cross-dataset EEG->image bridge — reconcile THINGS-EEG1 and THINGS-EEG2 for a zero-shot transfer test.

Both datasets show the SAME 1,854 THINGS concepts, so an encoder trained on one can be evaluated on the other
— the hardest honest generalization test (different people, different rig, same semantics). Two reconciliations
make it honest, and both are pure set/label logic (no EEG, no GPU) so they're unit-tested here:

  1. Concept identity is the THINGS NAME (EEG1 carries it inline; EEG2 derives it from its concept order). The
     retrieval target for a trial is that concept's shared CLIP prototype — looked up by name, so the two
     datasets land in one embedding space.
  2. Zero-shot holdout: EEG2's 200 test concepts also occur in EEG1, so training the EEG1 encoder on them would
     leak the eval concepts. `holdout_mask` removes them from the training side; retrieval still runs on the
     200-concept bank. The result is cross-dataset AND concept-zero-shot AND cross-subject at once.
"""
from __future__ import annotations

import numpy as np


def holdout_mask(concept_names: np.ndarray, holdout: set[str]) -> np.ndarray:
    """Boolean mask of trials to KEEP for training: every trial whose concept is NOT in the eval `holdout`
    set. Keeps the cross-dataset encoder from ever seeing the concepts it will be retrieved on."""
    names = np.asarray(concept_names)
    return np.array([name not in holdout for name in names], dtype=bool)


def align_targets(concept_names: np.ndarray, name_to_index: dict[str, int]
                  ) -> tuple[np.ndarray, np.ndarray]:
    """Map each trial's concept NAME to its index in a shared candidate bank. Returns (label_index[m],
    keep_mask[n]) where keep_mask drops trials whose concept isn't in the bank (so the two datasets' differing
    concept coverage is reconciled explicitly, not silently mis-indexed). label_index is aligned to the kept
    trials, in order."""
    names = np.asarray(concept_names)
    keep = np.array([name in name_to_index for name in names], dtype=bool)
    labels = np.array([name_to_index[name] for name in names[keep]], dtype=np.int64)
    return labels, keep


def name_to_bank_index(candidate_names: list[str]) -> dict[str, int]:
    """{concept name -> row index} for a candidate bank whose rows are ordered by `candidate_names`. The
    inverse of the sorted-concept order EEG2's prototypes use, so an EEG1 concept name resolves to the right
    bank row."""
    return {name: index for index, name in enumerate(candidate_names)}
