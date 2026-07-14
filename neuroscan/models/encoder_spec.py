"""The EEG→CLIP encoder contract — a leaf module (bd ylq / 3nn cycle-break).

`EncoderSpec` (data-derived shape) + `ImageEncoder` (the forward Protocol) live here, NOT in `encoders.py`,
so both the registry (`encoders.py`, which imports the concrete `foundation.Foundation` builders) and the
backbones (`foundation.py`, which only needs the spec type) can import them WITHOUT an import cycle. Extraction,
not a lazy import (the house rule for breaking circulars).
"""
from __future__ import annotations

from typing import Protocol

from jaxtyping import Float
from pydantic import BaseModel
from torch import Tensor


class EncoderSpec(BaseModel):
    """The data-derived shape every encoder needs: EEG channel count, epoch length, and the CLIP target dim
    the embedding must match. Passed to a builder so the encoder's shape is fixed by the data, not hardcoded."""
    n_channels: int
    n_times: int
    embed_dim: int


class ImageEncoder(Protocol):
    """The EEG→image encoder contract: `forward([B, C, T]) -> L2-normalized [B, embed_dim]`. Documents what a
    new backbone must satisfy to drop into `train_nice` (NICE already does) — the return type of
    `build_encoder`, so the trainer programs against the contract, not a concrete class."""

    def forward(self, x: Float[Tensor, "n ch t"]) -> Float[Tensor, "n d"]: ...
