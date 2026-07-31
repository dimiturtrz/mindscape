"""The differentiable feature-weighting model — softmax per-group weights × standardised features → linear
head (the torch model the differentiable feature-importance search optimizes; see `differentiable.py`).
"""
from __future__ import annotations

from typing import override

import torch
from jaxtyping import Float, Int
from torch import Tensor


class WeightedLinear(torch.nn.Module):
    """Softmax feature weights (per group) × standardised features → linear head. `group_idx[j]` = the
    weight-group of column j (per-family: 0..14; per-channel: j itself), so one learnable logit per group
    broadcasts to its columns."""

    group_idx: torch.Tensor   # registered buffer (declared so indexing sees Tensor, not Module|Tensor)

    def __init__(self, group_idx: Int[Tensor, "f"], n_groups: int, d: int, n_classes: int):
        super().__init__()
        self.logits = torch.nn.Parameter(torch.zeros(n_groups))
        self.head = torch.nn.Linear(d, n_classes)
        self.register_buffer("group_idx", group_idx)

    def weights(self) -> Float[Tensor, "f"]:
        return torch.softmax(self.logits, dim=0)

    def entropy(self) -> Float[Tensor, ""]:
        w = self.weights()
        return -(w * (w + 1e-12).log()).sum()

    @override
    def forward(self, x: Float[Tensor, "n f"]) -> Float[Tensor, "n c"]:
        return self.head(x * self.weights()[self.group_idx])
