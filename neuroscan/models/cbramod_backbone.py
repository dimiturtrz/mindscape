"""CBraMod pretrained backbone as a `composite.Backbone` (bd yjd, epic bji).

CBraMod (Wang et al., ICLR 2025 — a criss-cross transformer, 4.9M params, pretrained on 27,062 h of Temple
University EEG). Input is patched `[B, C, S, P]`, `P=200` points/patch at 200 Hz; our pre-normalized sensor
epoch `[B, C, T]` is reshaped to `[B, C, T//P, P]`. Normalization is the upstream `core.normalization` chain's
job (CBraMod's amplitude scale, bd 7mi4), not the backbone's.

Backbone checked out (not vendored) under `external/CBraMod`; pretrained weights live out-of-repo under
`<data_root>/pretrained/CBraMod/pretrained_weights.pth`. Reproduce:

    git clone https://github.com/wjq-learning/CBraMod external/CBraMod
    git -C external/CBraMod checkout 0ff6be918985689e7df679bc731ffb70e6c6224f   # MIT
    # then download to <data_root>/pretrained/CBraMod/pretrained_weights.pth :
    #   https://huggingface.co/weighting666/CBraMod/resolve/main/pretrained_weights.pth
"""
from __future__ import annotations

import sys
from typing import override

import torch
from jaxtyping import Float
from torch import nn

from core.config import REPO, Config
from neuroscan.models.backbone import Backbone

_CBRAMOD_ROOT = REPO / "external" / "CBraMod"   # checked out @ 0ff6be91 (MIT); see the fetch step above


class CBraModBackbone(Backbone):
    """CBraMod as a `composite.Backbone`: pre-normalized epoch -> `[B, C, S, d]` token grid. Input normalization
    is the upstream chain's job (CBraMod's amplitude scale, bd 7mi4), not the backbone's. The checkpoint fixes
    `patch_points` (200 pts = 1s at 200 Hz -> S=1 on our stimulus) and `d_model` (200)."""

    def __init__(self):
        super().__init__()
        self.module = CBraModBackbone.load_cbramod()            # raw CBraMod (consumes pre-patched input)
        self.patch_points = 200
        self.d_model = 200

    @override
    def forward(self, x: Float[torch.Tensor, "n ch t"]) -> Float[torch.Tensor, "n ch s d"]:
        b, c, t = x.shape
        p = self.patch_points
        s = t // p
        x = x[:, :, :s * p]                                        # drop the ragged tail patch (input already scaled)
        return self.module(x.reshape(b, c, s, p))                 # [B, C, S, d_model]

    @staticmethod
    def load_cbramod() -> nn.Module:
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
