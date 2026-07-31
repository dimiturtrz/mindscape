"""Numpy-backed torch Dataset for the retrieval trainer — converts per sample so the (large) training array
is never copied into one torch tensor up front (see `train_nice.py`).
"""
from __future__ import annotations

from typing import override

import numpy as np
import torch
from jaxtyping import Float, Int
from torch.utils.data import Dataset


class EpochDataset(Dataset[tuple[torch.Tensor, torch.Tensor, int]]):
    """Numpy-backed — converts per sample, so the (large) training array is never copied into one torch
    tensor up front. `indices` optionally views a subset of the arrays without copying them (the fit split
    of a much larger epoch pile). Together these keep a full 9-subject LOSO (~38 GB of epochs) in RAM instead
    of OOM-ing on the doubled copies (torch tensor + boolean-mask slice)."""

    def __init__(self, eeg: Float[np.ndarray, "n ch t"], targets: Float[np.ndarray, "n d"],
                 indices: Int[np.ndarray, "m"] | None = None, subject: Int[np.ndarray, "n"] | None = None):
        self.eeg, self.targets, self.subject = eeg, targets, subject
        self.indices = indices

    def __len__(self) -> int:
        return len(self.indices) if self.indices is not None else len(self.eeg)

    @override
    def __getitem__(self, index: int):
        row = int(self.indices[index]) if self.indices is not None else index
        subj = 0 if self.subject is None else int(self.subject[row])
        return torch.from_numpy(self.eeg[row]), torch.from_numpy(self.targets[row]), subj
