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

from torch import nn

from neuroscan.models.encoder_spec import EncoderSpec, ImageEncoder
from neuroscan.models.foundation import Foundation
from neuroscan.models.nice import NiceConfig, NiceEncoder

_BUILDERS: dict[str, Callable[[EncoderSpec], nn.Module]] = {}


class EncoderRegistry:
    """Name -> encoder-builder registry — the free helpers folded in as staticmethods (public names kept).
    The built-in builders register lazily on first `build_encoder` (kept out of import time — no side effects
    on import), so importing this module builds nothing. NICE lives here; the pretrained CBraMod builders live
    in `foundation.py` and are registered here too, so registration has one home (mirrors core.data.registry)."""

    _populated = False

    @staticmethod
    def register(name: str, builder: Callable[[EncoderSpec], nn.Module]) -> None:
        _BUILDERS[name] = builder

    @staticmethod
    def build_encoder(name: str, spec: EncoderSpec) -> ImageEncoder:
        """The encoder for a model name, shaped by `spec`. `KeyError` (with the known names) on an unknown model."""
        EncoderRegistry._ensure_populated()
        if name not in _BUILDERS:
            raise KeyError(f"unknown encoder {name!r}; have {sorted(_BUILDERS)}")
        return _BUILDERS[name](spec)

    @staticmethod
    def _build_nice(spec: EncoderSpec) -> nn.Module:
        return NiceEncoder(NiceConfig(n_channels=spec.n_channels, n_times=spec.n_times, embed_dim=spec.embed_dim))

    @staticmethod
    def _ensure_populated() -> None:
        """Register the built-in encoder builders exactly once, on first use (not at import)."""
        if EncoderRegistry._populated:
            return
        EncoderRegistry._populated = True
        EncoderRegistry.register("nice", EncoderRegistry._build_nice)   # from-scratch conv baseline (Song ICLR 2024)
        EncoderRegistry.register("cbramod", Foundation._build_cbramod)          # frozen backbone + head (linear probe)
        EncoderRegistry.register("cbramod_ft", Foundation._build_cbramod_ft)    # unfrozen — fine-tune on perception
        EncoderRegistry.register("cbramod_ft_attn", Foundation._build_cbramod_ft_attn)  # unfrozen + attention pool (bd)
        EncoderRegistry.register("cbramod_lora", Foundation._build_cbramod_lora)  # frozen + rank-8 LoRA adapters (29z)
