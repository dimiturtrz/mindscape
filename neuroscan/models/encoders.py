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

import numpy as np
from jaxtyping import Int
from torch import nn

from core.normalization.mvnn import Mvnn
from core.normalization.normalization import CompositeNormalization
from core.normalization.scale import Scale
from core.normalization.zscore import ZScore
from neuroscan.models.encoder_spec import EncoderSpec, ImageEncoder
from neuroscan.models.foundation import Foundation
from neuroscan.models.nice import NiceConfig, NiceEncoder

_BUILDERS: dict[str, Callable[[EncoderSpec], nn.Module]] = {}

# our THINGS-EEG2 raw is Volts; CBraMod pretrained on microvolts/100 (pretrain_trainer.py x/100). V->uV is x1e6,
# then /100 = x1e4 -> our O(10uV) signal lands at the O(1) amplitude the pretrained conv filters saw (bd 7mi4).
_CBRAMOD_SCALE = 1e4
_AUTO = "auto"
NORMALIZE_CHOICES = (_AUTO, "zscore", "mvnn", "scale")   # the --normalize CLI vocab, shared by the perception runners


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
    def normalization(model: str, override: str = _AUTO, groups: Int[np.ndarray, "n"] | None = None,
                      conditions: Int[np.ndarray, "n"] | None = None) -> CompositeNormalization:
        """The input-normalization chain an encoder expects, as directly-constructed objects (no registry).
        `override=_AUTO` is **per-channel z-score for every encoder** — the current evidenced default. Two named
        overrides: `mvnn` (official THINGS-EEG2 per-subject noise whitening — each subject's `Σ^{-1/2}` fit on
        its OWN calibration/training-image trials, applied to that subject; leak-free and unbatched-real. Clean
        it is ≈ z-score on NICE single-top1 (1.66 vs 1.60, within single-seed noise; the b40j +0.35 was a
        transductive LEAK, mild ranking edge + concept-avg cost) — kept as a correct, documented override, NOT
        the default) and `scale` (CBraMod's pretraining microvolts/100; REGRESSED the frozen geometry heads vs
        z-score, bd 7mi4). `mvnn` needs `groups`=subject + `conditions`=image aligned with the CALIBRATION
        epochs it is fit on."""
        if override == "scale":
            return CompositeNormalization([Scale(_CBRAMOD_SCALE)])
        if override == "mvnn":
            return CompositeNormalization([Mvnn(groups, conditions)])
        return CompositeNormalization([ZScore()])   # _AUTO for every encoder + explicit 'zscore'

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
