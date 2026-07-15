"""Low-rank adaptation (LoRA, Hu et al. 2021) of a frozen backbone's linear layers — the cheap sub-patch
temporal unlock for the CBraMod perception fine-tune (bd 29z).

CBraMod is 200 Hz / 1 s-patch: our 1 s stimulus is a SINGLE patch (S=1), so fine ERP timing is compressed
inside the frozen patch representation and unreachable by any head or by resampling. The only lever is to
adapt the backbone weights. Full fine-tune works (single-seed 2.38%) but trains all 4.9M params; LoRA freezes
the pretrained weight `W` and learns a rank-`r` residual `ΔW = (α/r)·B·A` (`A: d_in→r`, `B: r→d_out`,
`B` zero-init so the adapted forward starts EXACTLY at the pretrained one) — a few % of the params, same
inference path.

Reachable surface: CBraMod's attention is a fused `nn.MultiheadAttention` whose q/k/v live in a single
`in_proj_weight` PARAMETER and whose `out_proj` is read by attribute (`self.out_proj.weight`) inside torch's
attention kernel, not called as a module — so neither can be swapped for a wrapper without breaking the kernel
(and PEFT's `nn.Linear`-targeted LoRA can't reach them either). What CAN be wrapped is the per-layer
feed-forward `linear1`/`linear2` — plain `nn.Linear`s invoked normally, and the bulk of the transformer's
weight (d_model→dim_feedforward→d_model, 4× wider than attention). LoRA adapts the FFN while the pretrained
attention stays frozen. (PEFT would add a dependency for this same `nn.Linear` subset with no access to the
fused attention either, so the 20-line hand-roll here is the lighter equal — no vendored dep, one op-namespace.)
"""
from __future__ import annotations

import math

from jaxtyping import Float
from torch import Tensor, nn

_RANK = 8              # bd 29z: rank-8 residual (α/r scaling below); ~4.7% of the CBraMod backbone trained
_ALPHA = 16.0          # LoRA scaling α (α/r = 2.0) — the conventional α = 2·r
_TARGETS = ("linear1", "linear2")   # the safely-wrappable nn.Linear surface: the transformer FFN


class LoraLinear(nn.Module):
    """A frozen `nn.Linear` plus a trainable rank-`r` residual: `y = W x + (α/r)·B(A(x))`. `A` is
    kaiming-init, `B` is zero-init so the module reproduces the base layer exactly at step 0 (the pretrained
    forward is preserved); only `A`/`B` carry gradient — the base weight/bias stay frozen."""

    def __init__(self, base: nn.Linear, rank: int = _RANK, alpha: float = _ALPHA):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad = False
        self.a = nn.Linear(base.in_features, rank, bias=False)
        self.b = nn.Linear(rank, base.out_features, bias=False)
        nn.init.kaiming_uniform_(self.a.weight, a=math.sqrt(5))
        nn.init.zeros_(self.b.weight)
        self.scaling = alpha / rank

    def forward(self, x: Float[Tensor, "... d_in"]) -> Float[Tensor, "... d_out"]:
        return self.base(x) + self.scaling * self.b(self.a(x))


class Lora:
    """Inject LoRA adapters into a frozen module tree — the op-namespace for the low-rank fine-tune (bd 29z)."""

    @staticmethod
    def inject(module: nn.Module, rank: int = _RANK, alpha: float = _ALPHA,
               targets: tuple[str, ...] = _TARGETS) -> int:
        """Recursively replace every child `nn.Linear` whose attribute name is in `targets` with a `LoraLinear`
        wrapping it (base frozen, A/B trainable). Returns the number of layers adapted. The caller freezes the
        rest of the backbone; the injected A/B are the only trainable backbone params left."""
        n = 0
        for name, child in module.named_children():
            if isinstance(child, nn.Linear) and name in targets:
                setattr(module, name, LoraLinear(child, rank, alpha))
                n += 1
            else:
                n += Lora.inject(child, rank, alpha, targets)
        return n
