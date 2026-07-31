"""NICE-style EEG image encoder (Song et al., ICLR 2024) — the Stage-3 EEG->image baseline.

The correct paradigm here is *retrieval*, not classification: learn an EEG encoder that maps an epoch to the
CLIP image-embedding space, trained contrastively (InfoNCE) against the CLIP embedding of the image the
subject was viewing. At test we do zero-shot retrieval over the 200 held-out concepts (disjoint from train):
cosine-match the EEG embedding to the 200 candidate CLIP embeddings -> top-1/top-5 vs 0.5% chance. Generation
(diffusion reconstruction) is a later head; the measured retrieval gap is the point (the field over-reports
within-subject / averaged / small-candidate retrieval — our contribution is the cross-subject single-trial
number, the MI playbook applied to perception).

This is deliberately compact (the NICE encoder is light): a temporal conv over time, a spatial conv over
channels, then a projection to the CLIP dim. Trained with a symmetric InfoNCE (CLIP's own loss). No image
pixels touch the net — only precomputed CLIP embeddings (see tasks/visual/clip_targets.py).
"""
from __future__ import annotations

from typing import override

import torch
import torch.nn.functional as F
from jaxtyping import Float
from pydantic import BaseModel
from torch import nn

_POOL_TIMES = 16   # fixed temporal budget the encoder pools down to (keeps proj input shape constant)


class NiceConfig(BaseModel):
    """Encoder hyperparameters. `n_channels`/`n_times` come from the data; the rest are the NICE recipe."""
    n_channels: int
    n_times: int
    embed_dim: int = 512
    n_temporal_filters: int = 40
    temporal_kernel: int = 25
    dropout: float = 0.5


class NiceEncoder(nn.Module):
    """EEG epoch [B, C, T] -> L2-normalized embedding [B, D] in CLIP space.

    Temporal depthwise conv (learns per-channel temporal filters) -> spatial conv (mixes channels) ->
    pooled -> MLP projection to `embed_dim`. Small on purpose: EEG->image overfits fast at n<=10 subjects,
    so the encoder stays light and the CLIP target carries the semantic structure.
    """

    def __init__(self, config: NiceConfig):
        super().__init__()
        n_filters = config.n_temporal_filters
        self.temporal = nn.Conv2d(1, n_filters, (1, config.temporal_kernel),
                                  padding=(0, config.temporal_kernel // 2))
        self.spatial = nn.Conv2d(n_filters, n_filters, (config.n_channels, 1))
        self.bn = nn.BatchNorm2d(n_filters)
        self.pool = nn.AdaptiveAvgPool2d((1, _POOL_TIMES))
        self.drop = nn.Dropout(config.dropout)
        self.proj = nn.Sequential(
            nn.Flatten(),
            nn.Linear(n_filters * _POOL_TIMES, config.embed_dim),
            nn.GELU(),
            nn.Linear(config.embed_dim, config.embed_dim),
        )

    @override
    def forward(self, x: Float[torch.Tensor, "n ch t"]) -> Float[torch.Tensor, "n d"]:
        x = x.unsqueeze(1)                                 # [B,1,C,T]
        x = self.temporal(x)
        x = torch.clamp(self.bn(self.spatial(x)), -10, 10)  # [B,F,1,T]
        x = F.elu(x)
        x = self.drop(self.pool(x))                        # [B,F,1,16]
        z = self.proj(x)                                   # [B,D]
        return F.normalize(z, dim=-1)

    def geo_penalty(self, laplacian: Float[torch.Tensor, "ch ch"]) -> Float[torch.Tensor, ""]:
        """Graph-Laplacian spatial-smoothness penalty on the spatial conv (bd 1x0). The spatial conv mixes the C
        channels with a per-channel weight; reshaped to W = [C, F·F], tr(Wᵀ L W) = ½ Σ_ij A_ij ‖w_i − w_j‖²
        pushes neighbouring electrodes toward similar mixing weights — the montage adjacency injected as a
        small-data prior (knowledge > data). `laplacian` from EegMontage.channel_laplacian, on the same device."""
        weight = self.spatial.weight
        w = weight.reshape(-1, weight.shape[2]).t()   # [C, F·F]
        return (w * (laplacian @ w)).sum()
