"""Pretrained EEG foundation-model encoders for the visual retrieval trainer (bd yjd, epic bji).

The `Foundation` op-namespace resolves a frozen backbone by name (`CBraModBackbone` / `EegptBackbone`, each in
its own module) into a `LoadedBackbone`, and builds the `Model = Backbone + Head` composites the encoder
registry serves — so `train_nice` can pit a pretrained encoder against the from-scratch NICE baseline (the
epic's capacity-vs-SNR-floor test). Capacity lives in the frozen pretrained weights; only a small head learns
the CLIP map. The builders are registered lazily by `encoders.EncoderRegistry` (one registration home, no
import-time side effects), so importing this module registers nothing on its own.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

from torch import nn

from neuroscan.models.cbramod_backbone import CBraModBackbone
from neuroscan.models.composite import HeadContext, HeadSpec, Model, TokenHead
from neuroscan.models.eegpt_backbone import _EEGPT_PATCH, _EEGPT_RATE, EegptBackbone
from neuroscan.models.lora import LoraLinear

if TYPE_CHECKING:
    from neuroscan.models.encoder_spec import EncoderSpec

logger = logging.getLogger(__name__)


@dataclass
class LoadedBackbone:
    """A frozen foundation backbone ready for feature extraction: the module (emits a `[B, C, S, d]` token
    grid), plus the checkpoint-fixed geometry a caller needs to feed it — `patch_points`/`sample_rate` set S
    for a 1s epoch (CBraMod: 200/200 -> S=1), `d_model` the token width. `name` keys the on-disk feature cache."""
    module: nn.Module
    patch_points: int
    d_model: int
    sample_rate: float
    name: str


class Foundation:
    """Frozen backbone resolution + the encoder builders — the free helpers folded in as staticmethods (public
    names kept). The builders are registered lazily by `encoders.EncoderRegistry` (one registration home, no
    import-time side effects), so importing this module registers nothing on its own."""

    @classmethod
    def load_backbone(cls, name: str = "cbramod", channel_names: list[str] | None = None) -> LoadedBackbone:
        """Resolve a frozen backbone by name -> a `LoadedBackbone` whose `module` is a `composite.Backbone`
        ([B,C,T] -> [B,C,S,d], owning its own patching + normalization). The seam the frozen-head loop swaps on:
        a new foundation model is one entry here, not a fork of the runner. `channel_names` feeds a montage
        adapter (EEGPT needs it to map channels to its CHANNEL_DICT; CBraMod ignores it)."""
        builders = {"cbramod": cls._loaded_cbramod,
                    "eegpt": lambda: cls._loaded_eegpt(channel_names, 0.0, "eegpt"),
                    "eegpt_ov": lambda: cls._loaded_eegpt(channel_names, 0.5, "eegpt_ov")}
        if name not in builders:
            raise KeyError(f"unknown backbone {name!r} — registered: {sorted(builders)}")
        return builders[name]()

    @classmethod
    def _loaded_cbramod(cls) -> LoadedBackbone:
        return LoadedBackbone(CBraModBackbone(), patch_points=200, d_model=200, sample_rate=200.0, name="cbramod")

    @classmethod
    def _loaded_eegpt(cls, channel_names: list[str] | None, overlap: float, name: str) -> LoadedBackbone:
        """`overlap` = fraction of a patch shared with the next (0.0 = non-overlapping, stride = patch -> N=4 on
        1s; 0.5 = the conventional 50% overlap -> N=7). Stride is DERIVED from the patch size, so it tracks the
        checkpoint's patch if that ever changes. `name` keys the feature cache so the variants never collide."""
        if channel_names is None:
            raise ValueError("eegpt needs channel_names for its montage adapter (map to EEGPT's CHANNEL_DICT)")
        stride = None if overlap == 0.0 else round(_EEGPT_PATCH * (1.0 - overlap))
        return LoadedBackbone(EegptBackbone(channel_names, stride), patch_points=_EEGPT_PATCH, d_model=512,
                              sample_rate=float(_EEGPT_RATE), name=name)

    @classmethod
    def _cbramod_model(cls, spec: EncoderSpec, pool: str, freeze: bool) -> Model:  # noqa: FBT001
        """`Model(CBraModBackbone, TokenHead)` — the frozen probe / fine-tune / attn-pool as one composite."""
        bb = CBraModBackbone()
        # mean/attn pool -> no n_tok
        head = TokenHead(HeadSpec("cbramod", pool), HeadContext(bb.d_model, None, spec.embed_dim), None)
        return Model(bb, head, freeze_backbone=freeze)

    @classmethod
    def build_cbramod(cls, spec: EncoderSpec) -> Model:
        return cls._cbramod_model(spec, pool="mean", freeze=True)

    @classmethod
    def build_cbramod_ft(cls, spec: EncoderSpec) -> Model:
        return cls._cbramod_model(spec, pool="mean", freeze=False)

    @classmethod
    def build_cbramod_ft_attn(cls, spec: EncoderSpec) -> Model:
        return cls._cbramod_model(spec, pool="attn", freeze=False)

    @classmethod
    def build_cbramod_lora(cls, spec: EncoderSpec) -> Model:
        """LoRA fine-tune (bd 29z): a frozen CBraMod with rank-8 adapters injected into its feed-forward linears
        (`linear1`/`linear2`) — the cheap middle between the frozen probe (0.6%, chance) and the full fine-tune
        (2.38%; the fused attention stays frozen — it isn't module-callable, see lora.py). The
        backbone is frozen/eval-locked; `LoraLinear.inject` re-arms only the low-rank A/B, which `Model.param_groups`
        picks up via `requires_grad` as the sole trainable backbone group."""
        model = cls._cbramod_model(spec, pool="mean", freeze=True)
        n_adapted = LoraLinear.inject(cast(nn.Module, model.backbone.module))
        logger.info(f"cbramod_lora: injected rank-8 LoRA into {n_adapted} linear layers")
        return model
