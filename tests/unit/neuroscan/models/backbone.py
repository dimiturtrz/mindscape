"""Test the Backbone base class contract via a concrete test implementation."""
import torch

from neuroscan.models.backbone import Backbone

torch.manual_seed(0)


class _TestBackbone(Backbone):
    """Concrete test backbone: [B, C, T] -> [B, C, S, d] via a learnable projection."""

    def __init__(self, d_model: int = 16, s_out: int = 3):
        super().__init__()
        self.d_model = d_model
        self.s_out = s_out
        self.proj = torch.nn.Linear(4, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, t = x.shape
        windows = x.reshape(b, c, self.s_out, 4)
        return self.proj(windows)


def test_backbone_produces_correct_shape():
    """Backbone subclass shapes [B, C, T] -> [B, C, S, d]."""
    d_model = 16
    s_out = 3
    backbone = _TestBackbone(d_model=d_model, s_out=s_out).eval()
    x = torch.randn(5, 8, 12)  # [B, C, T]
    with torch.no_grad():
        out = backbone(x)
    assert out.shape == (5, 8, s_out, d_model)
    assert backbone.d_model == d_model
