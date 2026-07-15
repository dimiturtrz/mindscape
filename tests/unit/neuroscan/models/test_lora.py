"""Equivalence-class tests for the LoRA adapter + injector (bd 29z).

Partition: (1) a fresh `LoraLinear` reproduces its base layer exactly (B zero-init) with the base frozen and
only A/B trainable; (2) a non-zero adapter shifts the output off the base; (3) leading dims pass through; and
(4) `Lora.inject` swaps ONLY the target-named linears in a nested tree, freezing each base and arming its A/B.
No CBraMod checkout needed — a toy module stands in for the backbone."""
import torch
from torch import nn

from neuroscan.models.lora import Lora, LoraLinear


def test_fresh_adapter_reproduces_base_and_freezes_it():
    """Class: zero-init residual — B is zero so `LoraLinear` == base at step 0 (pretrained forward preserved);
    the base weight/bias are frozen, only A/B carry gradient."""
    base = nn.Linear(6, 4)
    lora = LoraLinear(base, rank=8)
    x = torch.randn(3, 6)
    torch.testing.assert_close(lora(x), base(x))                       # ΔW = 0 at init
    trainable = {n for n, p in lora.named_parameters() if p.requires_grad}
    assert trainable == {"a.weight", "b.weight"}                       # base frozen, adapters live


def test_nonzero_adapter_shifts_output():
    """Class: active residual — once B is non-zero the adapter changes the output (the low-rank path is wired,
    not a dead branch)."""
    base = nn.Linear(6, 4)
    lora = LoraLinear(base, rank=8)
    nn.init.normal_(lora.b.weight)                                     # arm the residual
    x = torch.randn(3, 6)
    assert not torch.allclose(lora(x), base(x))


def test_adapter_passes_leading_dims():
    """Class: shape — CBraMod feeds a 4D token grid, so the adapter must map [..., d_in] -> [..., d_out]."""
    lora = LoraLinear(nn.Linear(10, 7), rank=8)
    assert lora(torch.randn(2, 5, 3, 10)).shape == (2, 5, 3, 7)


class _ToyLayer(nn.Module):
    """A transformer-ish layer: two target linears (`linear1`, `out_proj`) + one non-target (`gate`)."""

    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(8, 8)
        self.out_proj = nn.Linear(8, 8)
        self.gate = nn.Linear(8, 8)


def test_inject_replaces_only_targets_and_arms_adapters():
    """Class: selective injection — only the target-named linears become `LoraLinear` (base frozen, A/B
    trainable); a non-target linear is untouched and stays fully trainable; the count is the number swapped."""
    module = nn.Sequential(_ToyLayer(), _ToyLayer())
    n = Lora.inject(module, rank=8, targets=("linear1", "out_proj"))
    assert n == 4                                                      # 2 targets × 2 layers
    layer = module[0]
    assert isinstance(layer.linear1, LoraLinear) and isinstance(layer.out_proj, LoraLinear)
    assert isinstance(layer.gate, nn.Linear) and not isinstance(layer.gate, LoraLinear)
    assert not layer.linear1.base.weight.requires_grad                # injected base frozen
    assert layer.linear1.a.weight.requires_grad                       # adapter armed
    assert layer.gate.weight.requires_grad                            # non-target untouched
