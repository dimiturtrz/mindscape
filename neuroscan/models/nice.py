"""NICE-style EEG image encoder (Song et al., ICLR 2024) — the Stage-3 EEG->image baseline.

The honest paradigm here is *retrieval*, not classification: learn an EEG encoder that maps an epoch to the
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

import torch
import torch.nn.functional as F
from torch import nn


class NiceEncoder(nn.Module):
    """EEG epoch [B, C, T] -> L2-normalized embedding [B, D] in CLIP space.

    Temporal depthwise conv (learns per-channel temporal filters) -> spatial conv (mixes channels) ->
    pooled -> MLP projection to `embed_dim`. Small on purpose: EEG->image overfits fast at n<=10 subjects,
    so the encoder stays light and the CLIP target carries the semantic structure.
    """

    def __init__(self, n_channels: int, n_times: int, embed_dim: int = 512,
                 f_temporal: int = 40, k_temporal: int = 25, p_drop: float = 0.5):
        super().__init__()
        self.temporal = nn.Conv2d(1, f_temporal, (1, k_temporal), padding=(0, k_temporal // 2))
        self.spatial = nn.Conv2d(f_temporal, f_temporal, (n_channels, 1), groups=1)
        self.bn = nn.BatchNorm2d(f_temporal)
        self.pool = nn.AdaptiveAvgPool2d((1, 16))          # collapse to a fixed temporal budget
        self.drop = nn.Dropout(p_drop)
        self.proj = nn.Sequential(
            nn.Flatten(),
            nn.Linear(f_temporal * 16, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(1)                                 # [B,1,C,T]
        x = self.temporal(x)
        x = torch.clamp(self.bn(self.spatial(x)), -10, 10)  # [B,F,1,T]
        x = F.elu(x)
        x = self.drop(self.pool(x))                        # [B,F,1,16]
        z = self.proj(x)                                   # [B,D]
        return F.normalize(z, dim=-1)


def clip_infonce(eeg: torch.Tensor, img: torch.Tensor, logit_scale: torch.Tensor) -> torch.Tensor:
    """Symmetric InfoNCE (CLIP loss) between L2-normalized EEG and image embeddings in a batch.

    Positives = matched (eeg_i, img_i); negatives = every other image in the batch. Symmetric over both
    directions. `logit_scale` is the learned temperature (exp), clamped by the caller.
    """
    logits = logit_scale * eeg @ img.t()                   # [B,B]
    target = torch.arange(eeg.shape[0], device=eeg.device)
    return 0.5 * (F.cross_entropy(logits, target) + F.cross_entropy(logits.t(), target))


@torch.no_grad()
def retrieval_topk(eeg: torch.Tensor, candidates: torch.Tensor, labels: torch.Tensor,
                   ks: tuple[int, ...] = (1, 5)) -> dict[int, float]:
    """Zero-shot retrieval accuracy: for each EEG embedding, rank the `candidates` (one CLIP embedding per
    class) by cosine; hit@k if the true `labels` index is in the top-k. Chance = k / n_candidates."""
    sims = eeg @ candidates.t()                            # [N, n_cand]
    order = sims.argsort(dim=-1, descending=True)
    out: dict[int, float] = {}
    for k in ks:
        hit = (order[:, :k] == labels[:, None]).any(dim=-1).float().mean().item()
        out[k] = hit
    return out
