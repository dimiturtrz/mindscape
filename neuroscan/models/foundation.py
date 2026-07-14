"""Pretrained EEG foundation-model encoder for the visual retrieval trainer (bd yjd, epic bji).

Wraps the **CBraMod** backbone (Wang et al., ICLR 2025 — a criss-cross transformer, 4.9M params, pretrained on
27,062 h of Temple University EEG) behind the EEG→CLIP `ImageEncoder` contract, so `train_nice` can pit a
pretrained encoder against the from-scratch NICE baseline — the epic's capacity-vs-SNR-floor test. Capacity
lives in the frozen pretrained weights; only a small head learns the CLIP map (the answer to NICE's "overfits
at n≤17 subjects" — the big net isn't trained on the tiny labelled set).

CBraMod's input is patched: `[B, C, S, P]`, `P=200` points/patch at **200 Hz**. Our sensor epoch `[B, C, T]`
is per-channel z-scored (the deep-dive's normalization note — pretraining saw z-scored signals, not our
pipeline's) then reshaped to `[B, C, T//P, P]`. The backbone's per-token `d_model` features are mean-pooled
over (C, S); a trainable MLP maps `d_model → CLIP dim`. Channel count is flexible (verified: 63 posterior
channels feed straight through), so no montage projection is needed.

Backbone checked out (not vendored) under `external/CBraMod`; pretrained weights live out-of-repo under
`<data_root>/pretrained/CBraMod/pretrained_weights.pth`. Reproduce:

    git clone https://github.com/wjq-learning/CBraMod external/CBraMod
    git -C external/CBraMod checkout 0ff6be918985689e7df679bc731ffb70e6c6224f   # MIT
    # then download to <data_root>/pretrained/CBraMod/pretrained_weights.pth :
    #   https://huggingface.co/weighting666/CBraMod/resolve/main/pretrained_weights.pth
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from torch import nn

from core.config import REPO, Config
from neuroscan.models.composite import Backbone, HeadSpec, Model, TokenHead

if TYPE_CHECKING:
    from neuroscan.models.encoder_spec import EncoderSpec

_CBRAMOD_ROOT = REPO / "external" / "CBraMod"   # checked out @ 0ff6be91 (MIT); see the fetch step above


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


class CBraModBackbone(Backbone):
    """CBraMod as a `composite.Backbone`: per-channel z-scored epoch -> `[B, C, S, d]` token grid. The
    checkpoint fixes `patch_points` (200 pts = 1s at 200 Hz -> S=1 on our stimulus) and `d_model` (200)."""

    def __init__(self):
        super().__init__()
        lb = Foundation.load_backbone("cbramod")
        self.module = lb.module
        self.patch_points = lb.patch_points
        self.d_model = lb.d_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, t = x.shape
        p = self.patch_points
        s = t // p
        x = x[:, :, :s * p]                                        # drop the ragged tail patch
        x = (x - x.mean(-1, keepdim=True)) / (x.std(-1, keepdim=True) + 1e-6)   # per-channel z-score
        return self.module(x.reshape(b, c, s, p))                 # [B, C, S, d_model]


class Foundation:
    """CBraMod backbone loading + the encoder builders — the free helpers folded in as staticmethods (public
    names kept). The builders are registered lazily by `encoders.EncoderRegistry` (one registration home, no
    import-time side effects), so importing this module registers nothing on its own."""

    @staticmethod
    def load_backbone(name: str = "cbramod") -> LoadedBackbone:
        """Resolve a frozen backbone by name -> a `LoadedBackbone` (module + patch/rate/d_model geometry). The
        seam the frozen-head loop swaps on: a new foundation model is one entry here, not a fork of the runner.
        Backbones point DOWN to their loader; the runner stays backbone-agnostic (bd m69x.1)."""
        builders = {"cbramod": Foundation._loaded_cbramod}
        if name not in builders:
            raise KeyError(f"unknown backbone {name!r} — registered: {sorted(builders)}")
        return builders[name]()

    @staticmethod
    def _loaded_cbramod() -> LoadedBackbone:
        return LoadedBackbone(Foundation._load_backbone(), patch_points=200, d_model=200,
                              sample_rate=200.0, name="cbramod")

    @staticmethod
    def _load_backbone() -> nn.Module:
        """Instantiate CBraMod and load the pretrained weights. The backbone lives in a checked-out external repo
        (not a package), so its path is injected here — the one place that reaches into `external/` — rather than
        importing an uninstalled top-level module. `proj_out` (the pretrain reconstruction head) is dropped so the
        encoder exposes the raw `d_model` token features."""
        if str(_CBRAMOD_ROOT) not in sys.path:
            sys.path.insert(0, str(_CBRAMOD_ROOT))
        from models.cbramod import CBraMod  # noqa: PLC0415

        ckpt = Config.data_root("pretrained") / "CBraMod" / "pretrained_weights.pth"
        if not ckpt.exists():
            raise FileNotFoundError(f"CBraMod weights not at {ckpt} — see the fetch step in this module's docstring")
        backbone = CBraMod()
        backbone.load_state_dict(torch.load(ckpt, map_location="cpu"))
        backbone.proj_out = nn.Identity()       # expose d_model token features, not the reconstruction output
        return backbone

    @staticmethod
    def _cbramod_model(spec: EncoderSpec, pool: str, freeze: bool) -> Model:  # noqa: FBT001
        """`Model(CBraModBackbone, TokenHead)` — the frozen probe / fine-tune / attn-pool as one composite."""
        bb = CBraModBackbone()
        head = TokenHead(HeadSpec("cbramod", pool), bb.d_model, None, spec.embed_dim)  # mean/attn -> no n_tok
        return Model(bb, head, freeze_backbone=freeze)

    @staticmethod
    def _build_cbramod(spec: EncoderSpec) -> Model:
        return Foundation._cbramod_model(spec, pool="mean", freeze=True)

    @staticmethod
    def _build_cbramod_ft(spec: EncoderSpec) -> Model:
        return Foundation._cbramod_model(spec, pool="mean", freeze=False)

    @staticmethod
    def _build_cbramod_ft_attn(spec: EncoderSpec) -> Model:
        return Foundation._cbramod_model(spec, pool="attn", freeze=False)
