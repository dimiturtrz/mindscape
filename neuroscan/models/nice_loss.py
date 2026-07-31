"""The NICE training loss + zero-shot retrieval eval — the contrastive objective and the metrics that read
the EEG↔CLIP embedding geometry (see `nice.py` for the encoder they train/score).
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from jaxtyping import Float, Int


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
