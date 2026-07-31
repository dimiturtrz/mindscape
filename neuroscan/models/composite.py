"""The EEG→CLIP encoder as a `Model = Backbone + Head` composite — one shape for the whole encoder zoo.

Every perception encoder is the same two-stage thing: a feature extractor that turns a raw epoch into a token
grid (`Backbone`, backbone.py), then a head that folds those tokens to a point in CLIP space (`Head`, head.py):

    Backbone : [B, C, T] raw eeg   -> [B, C, S, d] token grid   (CBraMod / EEGPT / a NICE conv stem)
    Head     : [B, C, S, d] tokens -> [B, embed_dim]            (mean / flat / attn / pos_attn / topo / gcn)
    Model    : normalize(head(backbone(x)))                     -> the `ImageEncoder` contract, registry-ready

So the frozen probe is `Model(frozen bb, head)`, the fine-tune is `Model(unfrozen bb, head)`, and swapping the
backbone (a finer-patching EEGPT) is a new `Backbone` with the SAME head zoo — no bespoke encoder. `Model`
freezes/eval-locks the backbone when asked, so a frozen sweep caches the backbone's token grid once and trains
heads alone on the stored features (the frozen-head loop).
"""
from __future__ import annotations

from typing import override

import torch.nn.functional as F
from jaxtyping import Float
from torch import Tensor, nn

from neuroscan.models.backbone import Backbone
from neuroscan.models.head import Head


class Model(nn.Module):
    """The composite: `backbone` extracts tokens, `head` maps them to CLIP space, the output is L2-normalized
    (the `ImageEncoder` contract, so it drops into the registry + trainer unchanged). `freeze_backbone` keeps
    the pretrained weights fixed AND in eval (no dropout/norm drift) even under `.train()` — the frozen-probe
    recipe; unfrozen = fine-tune. A frozen sweep caches the backbone's token grid once (the backbone is the
    module the feature-cache persists) and trains heads on the stored tokens."""

    def __init__(self, backbone: Backbone, head: Head, freeze_backbone: bool = True):  # noqa: FBT001, FBT002
        super().__init__()
        self.backbone = backbone
        self.head = head
        self.freeze_backbone = freeze_backbone
        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

    @override
    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_backbone:
            self.backbone.eval()
        return self

    @override
    def forward(self, x: Float[Tensor, "n ch t"]) -> Float[Tensor, "n d_embed"]:
        return F.normalize(self.head(self.backbone(x)), dim=-1)

    def param_groups(self, base_lr: float, backbone_lr_scale: float = 1.0) -> list[dict[str, object]]:
        """Optimizer groups: the trainable backbone params (if any) at `base_lr × backbone_lr_scale`, then the
        head at `base_lr`. One path covers all three regimes by reading `requires_grad`: frozen -> no backbone
        group (head only); full fine-tune -> the whole backbone; LoRA -> only the injected A/B adapters (the
        base weights stay frozen). `backbone_lr_scale=1.0` is the whole-model single-LR fine-tune recipe."""
        groups: list[dict[str, object]] = []
        trainable_backbone = [p for p in self.backbone.parameters() if p.requires_grad]
        if trainable_backbone:
            groups.append({"params": trainable_backbone, "lr": base_lr * backbone_lr_scale})
        groups.append({"params": list(self.head.parameters()), "lr": base_lr})
        return groups
