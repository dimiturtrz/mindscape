"""Shared data structures for the EEG↔fNIRS fusion path.

The fusion functions kept sprouting long argument lists because the same things travel together: the two
modalities' block-aligned inputs + labels (+ subject groups), and the pooled per-block probabilities the
oracle/aggregation analysis consumes. Naming those bundles once removes the argument sprawl and makes the
pairing explicit (no more six loose `Xe_tr, Xf_tr, y_tr, Xe_va, ...` arrays).
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np


@dataclass
class FusionData:
    """Block-aligned EEG + fNIRS inputs for the *same* trials, with labels (and optionally subject groups).

    `eeg`/`fnirs` are whatever the modality consumes (raw epochs or features), aligned block-for-block;
    `groups` is the per-block subject id, needed by the re-centered EEG decoder and grouped inner splits."""
    eeg: np.ndarray
    fnirs: np.ndarray
    y: np.ndarray
    groups: np.ndarray | None = None


@dataclass
class ModalityModels:
    """The two decoders as callables, the shape the fusion combiners need them in.

    `eeg_probs(eeg_tr, y_tr, groups_tr, eeg_te, groups_te) -> probs` (groups so re-centering flows into the
    inner OOF); `fnirs_fit(fnirs_tr, y_tr) -> model` and `fnirs_score(model, fnirs_te) -> probs`."""
    eeg_probs: Callable
    fnirs_fit: Callable
    fnirs_score: Callable


@dataclass
class PooledProbs:
    """Per-block probabilities + correctness, pooled over folds — the input to the oracle-headroom and the
    output-space aggregation sweep. One field per (was `Pe/Pf/Stk/Pce/Pcf/ce/cf` in the old dict)."""
    eeg: np.ndarray
    fnirs: np.ndarray
    stacking: np.ndarray
    cal_eeg: np.ndarray
    cal_fnirs: np.ndarray
    y: np.ndarray
    eeg_correct: np.ndarray
    fnirs_correct: np.ndarray
