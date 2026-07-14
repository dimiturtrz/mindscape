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

**EEGPT** (Wang et al., NeurIPS 2024 — an autoregressive/summary-token transformer, patch 64 pts at **256 Hz**
= 0.25 s/patch, so a 1 s epoch is S=4 time-patches, escaping CBraMod's S=1) is the second frozen backbone
(bd m69x.3). Its encoder FUSES channels into `embed_num=4` summary tokens per time-patch, so it emits
`[B, N_time, embed_num, d=512]` — NOT a per-electrode grid: the geometry heads don't apply to it. Reproduce:

    git clone https://github.com/BINE022/EEGPT external/EEGPT
    git -C external/EEGPT checkout a0e0a8f                                     # Apache-2.0
    # pretrained backbone (figshare, CC BY 4.0) -> <data_root>/pretrained/EEGPT/eegpt_mcae_58chs_4s_large4E.ckpt
    #   article: https://figshare.com/articles/code/EEGPT_checkpoints/25866970  (the 'EEGPT/checkpoint/' file)
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING

import torch
from torch import nn

from core.config import REPO, Config
from neuroscan.models.composite import Backbone, HeadSpec, Model, TokenHead

if TYPE_CHECKING:
    from neuroscan.models.encoder_spec import EncoderSpec

_CBRAMOD_ROOT = REPO / "external" / "CBraMod"   # checked out @ 0ff6be91 (MIT); see the fetch step above
_EEGPT_MODELS = REPO / "external" / "EEGPT" / "downstream" / "Modules" / "models"   # checked out @ a0e0a8f (Apache-2.0)


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
        self.module = Foundation._load_backbone()                 # raw CBraMod (consumes pre-patched input)
        self.patch_points = 200
        self.d_model = 200

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, t = x.shape
        p = self.patch_points
        s = t // p
        x = x[:, :, :s * p]                                        # drop the ragged tail patch
        x = (x - x.mean(-1, keepdim=True)) / (x.std(-1, keepdim=True) + 1e-6)   # per-channel z-score
        return self.module(x.reshape(b, c, s, p))                 # [B, C, S, d_model]


class EegptBackbone(Backbone):
    """EEGPT as a `composite.Backbone`: `[B, C, T]` (256 Hz) -> `[B, N_time, embed_num, d=512]`. Subsets our
    montage to the 58 channels EEGPT knows (its `CHANNEL_DICT`; the 5 missing edge channels are dropped),
    per-channel z-scores, and runs the frozen EEGTransformer. The encoder fuses channels into `embed_num`
    summary tokens per time-patch — so the axes are (time-patch, summary), NOT electrodes: token heads
    (mean/flat/attn) apply, geometry heads do not."""

    _N_TIME = 256          # 1 s @ 256 Hz -> 4 time-patches at patch 64 (stride 64); overlap -> more (grow-N)

    def __init__(self, channel_names: list[str], patch_stride: int | None = None):
        super().__init__()
        module, chan_dict = Foundation._load_eegpt_encoder(self._N_TIME, patch_stride)
        self.module = module
        self.d_model = 512
        keep = [i for i, ch in enumerate(channel_names) if ch.upper().strip(".") in chan_dict]
        self.register_buffer("keep", torch.tensor(keep, dtype=torch.long))
        self.register_buffer("chan_ids", module.prepare_chan_ids([channel_names[i] for i in keep]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x[:, self.keep, :]                                    # -> the 58 EEGPT channels
        x = (x - x.mean(-1, keepdim=True)) / (x.std(-1, keepdim=True) + 1e-6)   # per-channel z-score (assumed norm)
        return self.module(x, chan_ids=self.chan_ids)            # [B, N_time, embed_num, d]


class Foundation:
    """CBraMod backbone loading + the encoder builders — the free helpers folded in as staticmethods (public
    names kept). The builders are registered lazily by `encoders.EncoderRegistry` (one registration home, no
    import-time side effects), so importing this module registers nothing on its own."""

    @staticmethod
    def load_backbone(name: str = "cbramod", channel_names: list[str] | None = None) -> LoadedBackbone:
        """Resolve a frozen backbone by name -> a `LoadedBackbone` whose `module` is a `composite.Backbone`
        ([B,C,T] -> [B,C,S,d], owning its own patching + normalization). The seam the frozen-head loop swaps on:
        a new foundation model is one entry here, not a fork of the runner. `channel_names` feeds a montage
        adapter (EEGPT needs it to map channels to its CHANNEL_DICT; CBraMod ignores it)."""
        builders = {"cbramod": lambda: Foundation._loaded_cbramod(),
                    "eegpt": lambda: Foundation._loaded_eegpt(channel_names, None, "eegpt"),
                    "eegpt_ov": lambda: Foundation._loaded_eegpt(channel_names, 16, "eegpt_ov")}
        if name not in builders:
            raise KeyError(f"unknown backbone {name!r} — registered: {sorted(builders)}")
        return builders[name]()

    @staticmethod
    def _loaded_cbramod() -> LoadedBackbone:
        return LoadedBackbone(CBraModBackbone(), patch_points=200, d_model=200, sample_rate=200.0, name="cbramod")

    @staticmethod
    def _loaded_eegpt(channel_names: list[str] | None, patch_stride: int | None, name: str) -> LoadedBackbone:
        """`eegpt` = non-overlapping patches (stride 64 -> N=4 on 1s); `eegpt_ov` = overlapping (stride 16 ->
        N=13), the grow-N arm. `name` keys the feature cache so the two never collide."""
        if channel_names is None:
            raise ValueError("eegpt needs channel_names for its montage adapter (map to EEGPT's CHANNEL_DICT)")
        return LoadedBackbone(EegptBackbone(channel_names, patch_stride), patch_points=64, d_model=512,
                              sample_rate=256.0, name=name)

    @staticmethod
    def _load_eegpt_encoder(n_time: int, patch_stride: int | None = None):
        """Build the EEGPT EEGTransformer encoder (its downstream config: patch 64, dim 512, embed_num 4,
        depth 8) sized to our epoch length and load the FROZEN pretrained `target_encoder` weights from the
        checkpoint. Returns (encoder, CHANNEL_DICT). Reaches into the external checkout (see the fetch step)."""
        if str(_EEGPT_MODELS) not in sys.path:
            sys.path.insert(0, str(_EEGPT_MODELS))
        from EEGPT_mcae import CHANNEL_DICT, EEGTransformer  # noqa: PLC0415

        ckpt = Config.data_root("pretrained") / "EEGPT" / "eegpt_mcae_58chs_4s_large4E.ckpt"
        if not ckpt.exists():
            raise FileNotFoundError(f"EEGPT weights not at {ckpt} — see the fetch step in this module's docstring")
        encoder = EEGTransformer(img_size=(58, n_time), patch_size=64, patch_stride=patch_stride, embed_dim=512,
                                 embed_num=4, depth=8, num_heads=8, mlp_ratio=4.0,
                                 norm_layer=partial(nn.LayerNorm, eps=1e-6))
        state = torch.load(ckpt, map_location="cpu", weights_only=False)
        state = state.get("state_dict", state)
        enc = {k[len("target_encoder."):]: v for k, v in state.items() if k.startswith("target_encoder.")}
        encoder.load_state_dict(enc, strict=True)
        return encoder, CHANNEL_DICT

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
