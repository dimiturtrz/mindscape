"""Encoder registry for the visual retrieval trainer (bd bji) — one place mapping a model name to a builder
that produces an EEG→CLIP encoder.

The contract every encoder honours: `forward(x [B, C, T]) -> L2-normalized [B, embed_dim]` in CLIP space
(what `train_nice` contrasts against the viewed image's CLIP embedding). The data-derived shape lives in
`EncoderSpec`; the builder turns (name, spec) into an `nn.Module`. This is the seam that lets the trainer swap
the from-scratch NICE baseline for a pretrained foundation-model backbone behind one interface — add a model =
one builder + one `register` line, no trainer change.
"""
from __future__ import annotations

from collections.abc import Callable
from typing import Protocol

from pydantic import BaseModel
from torch import Tensor, nn

from neuroscan.models.nice import NiceConfig, NiceEncoder


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

    def forward(self, x: Tensor) -> Tensor: ...


_BUILDERS: dict[str, Callable[[EncoderSpec], nn.Module]] = {}


def register(name: str, builder: Callable[[EncoderSpec], nn.Module]) -> None:
    _BUILDERS[name] = builder


def build_encoder(name: str, spec: EncoderSpec) -> ImageEncoder:
    """The encoder for a model name, shaped by `spec`. `KeyError` (with the known names) on an unknown model."""
    if name not in _BUILDERS:
        raise KeyError(f"unknown encoder {name!r}; have {sorted(_BUILDERS)}")
    return _BUILDERS[name](spec)


def _build_nice(spec: EncoderSpec) -> nn.Module:
    return NiceEncoder(NiceConfig(n_channels=spec.n_channels, n_times=spec.n_times, embed_dim=spec.embed_dim))


register("nice", _build_nice)   # the from-scratch conv baseline (Song et al. ICLR 2024)
