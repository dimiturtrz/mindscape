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

import numpy as np
import torch
import torch.nn.functional as F
from jaxtyping import Float, Int
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
        w = self.spatial.weight.reshape(-1, self.spatial.weight.shape[2]).t()   # [C, F·F]
        return (w * (laplacian @ w)).sum()


class Nice:
    @staticmethod
    def clip_infonce(eeg: Float[torch.Tensor, "n d"], img: Float[torch.Tensor, "n d"],
                     logit_scale: Float[torch.Tensor, ""],
                     hard_beta: float = 0.0, soft_tau: float = 0.0) -> Float[torch.Tensor, ""]:
        """Symmetric InfoNCE (CLIP loss) between L2-normalized EEG and image embeddings in a batch.

        Positives = matched (eeg_i, img_i); negatives = every other image in the batch. Symmetric over both
        directions. `logit_scale` is the learned temperature (exp), clamped by the caller.

        `hard_beta` > 0 turns on online hard-negative weighting (bd fww): each OFF-diagonal (negative) logit is
        boosted by `hard_beta ×` its own (detached) similarity, so high-similarity negatives contribute more to
        the softmax denominator and get a stronger push-down gradient. Rides this same forward pass — no extra
        inference — and self-sharpens as the encoder improves. `hard_beta = 0` is the exact standard CLIP loss.

        `soft_tau` > 0 replaces the hard one-hot target with a SOFT one — `softmax(img·imgᵀ / soft_tau)` — so a
        same-concept-different-image pair (CLIP targets ~0.7 similar) is a partial positive, not a false negative
        (bd lbd). Diagonal-dominant (self-sim = 1), tail set by `soft_tau`. Mutually exclusive with `hard_beta`.
        """
        logits = logit_scale * eeg @ img.t()                   # [B,B]
        if hard_beta > 0:
            off_diag = ~torch.eye(eeg.shape[0], dtype=torch.bool, device=eeg.device)
            logits = logits + hard_beta * logits.detach() * off_diag
        if soft_tau > 0:                                        # concept-aware soft targets (bd lbd)
            soft = F.softmax((img @ img.t()).detach() / soft_tau, dim=1)
            return -0.5 * ((soft * F.log_softmax(logits, dim=1)).sum(1).mean()
                           + (soft * F.log_softmax(logits.t(), dim=1)).sum(1).mean())
        target = torch.arange(eeg.shape[0], device=eeg.device)
        return 0.5 * (F.cross_entropy(logits, target) + F.cross_entropy(logits.t(), target))

    @staticmethod
    @torch.no_grad()
    def retrieval_hits(eeg: Float[torch.Tensor, "n d"], candidates: Float[torch.Tensor, "k d"],
                       labels: Int[torch.Tensor, "n"],
                       ks: tuple[int, ...] = (1, 5)) -> dict[int, np.ndarray]:
        """PER-TRIAL hit@k: for each EEG embedding, 1.0 if the true `labels` index is within the top-k
        cosine-ranked `candidates`, else 0.0. The un-averaged vector `retrieval_topk` means over — kept so a
        bootstrap can resample it for an honest CI (bd 5s3l). Returns one float array per k."""
        sims = eeg @ candidates.t()                            # [N, n_cand]
        order = sims.argsort(dim=-1, descending=True)
        return {k: (order[:, :k] == labels[:, None]).any(dim=-1).float().cpu().numpy() for k in ks}

    @staticmethod
    @torch.no_grad()
    def retrieval_topk(eeg: Float[torch.Tensor, "n d"], candidates: Float[torch.Tensor, "k d"],
                       labels: Int[torch.Tensor, "n"],
                       ks: tuple[int, ...] = (1, 5)) -> dict[int, float]:
        """Zero-shot retrieval accuracy: for each EEG embedding, rank the `candidates` (one CLIP embedding per
        class) by cosine; hit@k if the true `labels` index is in the top-k. Chance = k / n_candidates."""
        return {k: float(hits.mean()) for k, hits in Nice.retrieval_hits(eeg, candidates, labels, ks).items()}

    @staticmethod
    @torch.no_grad()
    def retrieval_continuous(eeg: Float[torch.Tensor, "n d"], candidates: Float[torch.Tensor, "k d"],
                             labels: Int[torch.Tensor, "n"]) -> dict[str, float]:
        """Continuous eval extras alongside hit@k (bd 2y7k) — the angular error the hard top-k accuracy discards
        (a rank-2 miss 5° off and a rank-180 miss 120° off score identically under top-1). Both operands are
        L2-normalized here so the dot IS cosine regardless of caller scale. Returns:
          - cos_to_true (mean/std): cosine of each prediction to its TRUE concept vector — the angular error dist.
          - margin (mean/std): cos_to_true − mean(cos to the other candidates) — the continuous analog of "ranked
            #1"; >0 = correctly biased toward the true concept even on a top-1 miss.
          - mean_rank: 1-based rank of the true concept (1 = top-1 hit), degrades gracefully unlike top-k.
        These mirror what the InfoNCE loss optimizes (pull to true, push from negatives), so they double as an
        eval-side consistency check on the loss.

        CLIP concept vectors are not evenly spread (animals cluster near animals), so an absolute cos-to-true is
        not self-interpretable. The candidate bank's own off-diagonal cosines are the reference — the cosine
        between two DIFFERENT concept vectors, i.e. what "random" cos looks like in this space — so `cos_to_true`
        is also reported as `cos_to_true_z`: standard deviations above that random-concept-pair baseline."""
        eeg = F.normalize(eeg, dim=-1)
        candidates = F.normalize(candidates, dim=-1)
        sims = eeg @ candidates.t()                                       # [N, n_cand] cosine
        n_cand = sims.shape[1]
        cos_true = sims[torch.arange(len(labels)), labels]               # [N] cos to the true concept
        mean_other = (sims.sum(dim=1) - cos_true) / max(1, n_cand - 1)   # mean cos to the other candidates
        margin = cos_true - mean_other
        rank = 1 + (sims > cos_true[:, None]).sum(dim=1)                 # candidates strictly closer than true
        pair_cos = candidates @ candidates.t()                           # concept-pair cosines (the reference)
        off_diag = pair_cos[~torch.eye(n_cand, dtype=torch.bool, device=pair_cos.device)]   # two random concepts
        random_mean, random_std = off_diag.mean(), off_diag.std()
        return {"cos_to_true_mean": float(cos_true.mean()), "cos_to_true_std": float(cos_true.std()),
                "cos_to_true_z": float((cos_true.mean() - random_mean) / (random_std + 1e-8)),
                "random_cos_mean": float(random_mean), "random_cos_std": float(random_std),
                "margin_mean": float(margin.mean()), "margin_std": float(margin.std()),
                "mean_rank": float(rank.float().mean())}


class _GradReverse(torch.autograd.Function):
    """Identity forward, sign-flipped (× λ) gradient backward — the DANN gradient-reversal layer. Placed
    before the subject discriminator so that minimizing the total loss trains the discriminator to name the
    subject while pushing the ENCODER to make that impossible (subject-invariant embedding)."""

    @staticmethod
    def forward(ctx, x: Float[torch.Tensor, "n d"], lambd: float) -> Float[torch.Tensor, "n d"]:
        ctx.lambd = lambd
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad: Float[torch.Tensor, "n d"]):
        return -ctx.lambd * grad, None


class SubjectDiscriminator(nn.Module):
    """Adversary that predicts which subject an embedding came from, through a gradient-reversal layer (bd
    36g). If the encoder's cross-subject collapse is because the embedding still encodes *who*, forcing it to
    fool this head should make the EEG->image map subject-invariant and transfer better."""

    def __init__(self, embed_dim: int, n_subjects: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(embed_dim, hidden), nn.ReLU(), nn.Linear(hidden, n_subjects))

    def forward(self, z: Float[torch.Tensor, "n d"], lambd: float) -> Float[torch.Tensor, "n subj"]:
        return self.net(_GradReverse.apply(z, lambd))
