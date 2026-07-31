"""The `Backbone` stage of the EEG→CLIP composite — raw epoch to token grid.

A `Backbone` turns `[B, C, T]` raw eeg into a `[B, C, S, d]` token grid; the concrete backbones (a NICE conv
stem, the pretrained CBraMod / EEGPT wrappers) subclass it. Freeze policy is `Model`'s, not the backbone's.
"""
from __future__ import annotations

from typing import override

from jaxtyping import Float
from torch import Tensor, nn


class Backbone(nn.Module):
    """[B, C, T] raw eeg -> [B, C, S, d] token grid. Subclasses wrap a pretrained/learned feature extractor;
    `d_model` is the token width the heads build against. Freeze policy is `Model`'s, not the backbone's."""

    d_model: int

    @override
    def forward(self, x: Float[Tensor, "n ch t"]) -> Float[Tensor, "n ch s d"]:  # [B, C, T] -> [B, C, S, d]
        raise NotImplementedError
