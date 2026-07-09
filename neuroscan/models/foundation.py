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

import torch
import torch.nn.functional as F
from pydantic import BaseModel
from torch import nn

from core.config import REPO, data_root
from neuroscan.models.encoders import EncoderSpec, register

_CBRAMOD_ROOT = REPO / "external" / "CBraMod"   # checked out @ 0ff6be91 (MIT); see the fetch step above


class FoundationConfig(BaseModel):
    """CBraMod-encoder knobs. Defaults = the frozen-backbone + small-head recipe (capacity in pretrained
    weights). `patch_points`/`d_model` are fixed by the checkpoint; the head + freeze are ours to tune."""
    patch_points: int = 200        # CBraMod points-per-patch — implies 200 Hz epochs (set resample=200)
    d_model: int = 200             # CBraMod per-token feature width (fixed by the checkpoint)
    freeze_backbone: bool = True   # capacity stays in the frozen pretrained weights; only the head learns
    backbone_lr_scale: float = 0.1  # when unfrozen, backbone fine-tunes at base_lr × this (< head's) so the
                                    # pretrained features aren't washed out — standard discriminative-LR fine-tune
    hidden: int = 512
    dropout: float = 0.5


def _load_backbone() -> nn.Module:
    """Instantiate CBraMod and load the pretrained weights. The backbone lives in a checked-out external repo
    (not a package), so its path is injected here — the one place that reaches into `external/` — rather than
    importing an uninstalled top-level module. `proj_out` (the pretrain reconstruction head) is dropped so the
    encoder exposes the raw `d_model` token features."""
    if str(_CBRAMOD_ROOT) not in sys.path:
        sys.path.insert(0, str(_CBRAMOD_ROOT))
    from models.cbramod import CBraMod  # noqa: PLC0415

    ckpt = data_root("pretrained") / "CBraMod" / "pretrained_weights.pth"
    if not ckpt.exists():
        raise FileNotFoundError(f"CBraMod weights not at {ckpt} — see the fetch step in this module's docstring")
    backbone = CBraMod()
    backbone.load_state_dict(torch.load(ckpt, map_location="cpu"))
    backbone.proj_out = nn.Identity()           # expose d_model token features, not the reconstruction output
    return backbone


class CBraModEncoder(nn.Module):
    """CBraMod backbone + a small CLIP-projection head, honouring the `ImageEncoder` contract
    (`[B, C, T] -> L2-normalized [B, embed_dim]`). Frozen backbone by default — only the head trains."""

    def __init__(self, spec: EncoderSpec, cfg: FoundationConfig | None = None):
        super().__init__()
        self.cfg = cfg or FoundationConfig()
        self.backbone = _load_backbone()
        if self.cfg.freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False
        self.head = nn.Sequential(
            nn.Linear(self.cfg.d_model, self.cfg.hidden), nn.GELU(), nn.Dropout(self.cfg.dropout),
            nn.Linear(self.cfg.hidden, spec.embed_dim),
        )

    def train(self, mode: bool = True):  # noqa: FBT001, FBT002
        """Keep the frozen backbone in eval (no dropout/norm drift) even when the encoder is train()-ed.
        Signature mirrors `nn.Module.train(mode=True)` — the boolean positional is PyTorch's, not ours."""
        super().train(mode)
        if self.cfg.freeze_backbone:
            self.backbone.eval()
        return self

    def param_groups(self, base_lr: float) -> list[dict]:
        """Discriminative-LR optimizer groups: the pretrained backbone fine-tunes at `base_lr ×
        backbone_lr_scale` (gentler than the head, so pretrained features survive), the fresh head at
        `base_lr`. Frozen backbone -> head-only group. Consumed by the trainer's `_build_optim`."""
        if self.cfg.freeze_backbone:
            return [{"params": list(self.head.parameters()), "lr": base_lr}]
        return [{"params": list(self.backbone.parameters()), "lr": base_lr * self.cfg.backbone_lr_scale},
                {"params": list(self.head.parameters()), "lr": base_lr}]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, t = x.shape
        p = self.cfg.patch_points
        s = t // p
        x = x[:, :, :s * p]                                        # drop the ragged tail patch
        x = (x - x.mean(-1, keepdim=True)) / (x.std(-1, keepdim=True) + 1e-6)   # per-channel z-score
        feats = self.backbone(x.reshape(b, c, s, p))              # [B, C, S, d_model]
        z = self.head(feats.mean(dim=(1, 2)))                     # pool tokens -> [B, embed_dim]
        return F.normalize(z, dim=-1)


def _build_cbramod(spec: EncoderSpec) -> nn.Module:
    return CBraModEncoder(spec, FoundationConfig(freeze_backbone=True))


def _build_cbramod_ft(spec: EncoderSpec) -> nn.Module:
    return CBraModEncoder(spec, FoundationConfig(freeze_backbone=False))


register("cbramod", _build_cbramod)       # frozen backbone + head (linear probe of pretrained features)
register("cbramod_ft", _build_cbramod_ft)  # unfrozen — fine-tune the backbone on perception (capacity test)
