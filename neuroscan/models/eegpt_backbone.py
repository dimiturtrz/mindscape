"""EEGPT pretrained backbone as a `composite.Backbone` (bd m69x.3).

EEGPT (Wang et al., NeurIPS 2024 — an autoregressive/summary-token transformer, patch 64 pts at 256 Hz =
0.25 s/patch, so a 1 s epoch is S=4 time-patches, escaping CBraMod's S=1). Its encoder FUSES channels into
`embed_num=4` summary tokens per time-patch, so it emits `[B, N_time, embed_num, d=512]` — NOT a per-electrode
grid: the geometry heads don't apply to it, only the token heads (mean/flat/attn). Reproduce:

    git clone https://github.com/BINE022/EEGPT external/EEGPT
    git -C external/EEGPT checkout a0e0a8f                                     # Apache-2.0
    # pretrained backbone (figshare, CC BY 4.0) -> <data_root>/pretrained/EEGPT/eegpt_mcae_58chs_4s_large4E.ckpt
    #   article: https://figshare.com/articles/code/EEGPT_checkpoints/25866970  (the 'EEGPT/checkpoint/' file)
"""
from __future__ import annotations

import sys
from functools import partial
from typing import override

import torch
from jaxtyping import Float
from torch import nn

from core.config import REPO, Config
from neuroscan.models.composite import Backbone

_EEGPT_MODELS = REPO / "external" / "EEGPT" / "downstream" / "Modules" / "models"   # checked out @ a0e0a8f (Apache-2.0)
_EEGPT_PATCH = 64          # EEGPT points/patch (checkpoint-fixed); 0.25 s at its rate
_EEGPT_RATE = 256          # EEGPT's native sample rate (checkpoint-fixed); the epoch is the 1 s THINGS stimulus
_EEGPT_EPOCH_S = 1.0       # -> feed n_time = rate × epoch seconds; resample data to the same rate (kept in sync)


class EegptBackbone(Backbone):
    """EEGPT as a `composite.Backbone`: `[B, C, T]` (256 Hz) -> `[B, N_time, embed_num, d=512]`. Subsets our
    montage to the 58 channels EEGPT knows (its `CHANNEL_DICT`; the 5 missing edge channels are dropped),
    per-channel z-scores, and runs the frozen EEGTransformer. The encoder fuses channels into `embed_num`
    summary tokens per time-patch — so the axes are (time-patch, summary), NOT electrodes: token heads
    (mean/flat/attn) apply, geometry heads do not."""

    _N_TIME = round(_EEGPT_RATE * _EEGPT_EPOCH_S)   # samples fed to the encoder (-> 4 patches at stride 64)
    keep: torch.Tensor        # registered buffers (declared so the type checker sees Tensor, not Module|Tensor)
    chan_ids: torch.Tensor

    def __init__(self, channel_names: list[str], patch_stride: int | None = None):
        super().__init__()
        module, chan_dict = EegptBackbone.load_eegpt_encoder(self._N_TIME, patch_stride)
        self.module = module
        self.d_model = 512
        keep = [i for i, ch in enumerate(channel_names) if ch.upper().strip(".") in chan_dict]
        self.register_buffer("keep", torch.tensor(keep, dtype=torch.long))
        self.register_buffer("chan_ids", module.prepare_chan_ids([channel_names[i] for i in keep]))

    @override
    def forward(self, x: Float[torch.Tensor, "n ch t"]) -> Float[torch.Tensor, "n n_time embed_num d"]:
        x = x[:, self.keep, :]                                    # -> the 58 EEGPT channels (input pre-normalized)
        return self.module(x, chan_ids=self.chan_ids)            # [B, N_time, embed_num, d]

    @staticmethod
    def load_eegpt_encoder(n_time: int, patch_stride: int | None = None):
        """Build the EEGPT EEGTransformer encoder (its downstream config: patch 64, dim 512, embed_num 4,
        depth 8) sized to our epoch length and load the FROZEN pretrained `target_encoder` weights from the
        checkpoint. Returns (encoder, CHANNEL_DICT). Reaches into the external checkout (see the fetch step)."""
        if str(_EEGPT_MODELS) not in sys.path:
            sys.path.insert(0, str(_EEGPT_MODELS))
        from EEGPT_mcae import CHANNEL_DICT, EEGTransformer  # noqa: PLC0415

        ckpt = Config.data_root("pretrained") / "EEGPT" / "eegpt_mcae_58chs_4s_large4E.ckpt"
        if not ckpt.exists():
            raise FileNotFoundError(f"EEGPT weights not at {ckpt} — see the fetch step in this module's docstring")
        encoder = EEGTransformer(img_size=(58, n_time), patch_size=_EEGPT_PATCH, patch_stride=patch_stride,
                                 embed_dim=512, embed_num=4, depth=8, num_heads=8, mlp_ratio=4.0,
                                 norm_layer=partial(nn.LayerNorm, eps=1e-6))
        state = torch.load(ckpt, map_location="cpu", weights_only=False)
        state = state.get("state_dict", state)
        enc = {k[len("target_encoder."):]: v for k, v in state.items() if k.startswith("target_encoder.")}
        encoder.load_state_dict(enc, strict=True)
        return encoder, CHANNEL_DICT
