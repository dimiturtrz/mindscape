"""Domain-adversarial subject discriminator (DANN, bd 36g) — the adversary that pushes the EEG→image encoder
toward a subject-invariant embedding, through a gradient-reversal layer.
"""
from __future__ import annotations

from typing import override

import torch
from jaxtyping import Float
from torch import nn


class SubjectDiscriminator(nn.Module):
    """Adversary that predicts which subject an embedding came from, through a gradient-reversal layer (bd
    36g). If the encoder's cross-subject collapse is because the embedding still encodes *who*, forcing it to
    fool this head should make the EEG->image map subject-invariant and transfer better."""

    class _GradReverse(torch.autograd.Function):
        """Identity forward, sign-flipped (× λ) gradient backward — the DANN gradient-reversal layer. Placed
        before the subject discriminator so that minimizing the total loss trains the discriminator to name the
        subject while pushing the ENCODER to make that impossible (subject-invariant embedding)."""

        @staticmethod
        @override
        def forward(ctx: torch.autograd.function.FunctionCtx,  # type: ignore[override]
                    x: Float[torch.Tensor, "n d"], lambd: float) -> Float[torch.Tensor, "n d"]:
            ctx.lambd = lambd     # pyrefly: ignore[missing-attribute]  — autograd ctx carries per-call state
            return x.view_as(x)

        @staticmethod
        @override
        def backward(ctx: torch.autograd.function.FunctionCtx,  # type: ignore[override]
                     grad: Float[torch.Tensor, "n d"]):
            return -ctx.lambd * grad, None   # pyrefly: ignore[missing-attribute]

    def __init__(self, embed_dim: int, n_subjects: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(embed_dim, hidden), nn.ReLU(), nn.Linear(hidden, n_subjects))

    @override
    def forward(self, z: Float[torch.Tensor, "n d"], lambd: float) -> Float[torch.Tensor, "n subj"]:
        return self.net(SubjectDiscriminator._GradReverse.apply(z, lambd))
