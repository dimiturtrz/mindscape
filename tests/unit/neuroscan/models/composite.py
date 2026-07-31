"""Equivalence-class tests for the Model = Backbone + Head composite (bd m69x.4).

Tests for the Model composite: forward shape/norm, freeze policy, and param groups."""
import numpy as np
import torch

from neuroscan.models.backbone import Backbone
from neuroscan.models.composite import Model
from neuroscan.models.head import Head, HeadContext, HeadSpec

np.random.seed(0)
torch.manual_seed(0)

_C, _S, _D, _EMBED = 8, 3, 16, 32


class _FakeBackbone(Backbone):
    """[B, C, T] -> [B, C, S, d] via a trainable linear on windowed slices (stands in for a real backbone)."""

    def __init__(self):
        super().__init__()
        self.d_model = _D
        self.proj = torch.nn.Linear(4, _D)

    def forward(self, x):
        b, c, _t = x.shape
        windows = x.reshape(b, c, _S, 4)                 # [B, C, S, 4]
        return self.proj(windows)                        # [B, C, S, d]


def _pos():
    return np.random.RandomState(0).randn(_C, 2) * 0.4


def _model(pool="mean", freeze=True):
    head = Head.build(HeadSpec(pool, pool), HeadContext(_D, _pos(), _EMBED), n_tok=_C * _S)
    return Model(_FakeBackbone(), head, freeze_backbone=freeze)


def test_forward():
    m = _model()
    with torch.no_grad():
        z = m.forward(torch.randn(5, _C, _S * 4))
    assert z.shape == (5, _EMBED)
    assert torch.allclose(z.norm(dim=-1), torch.ones(5), atol=1e-5)


def test_train():
    m = _model(freeze=True)
    m.train()
    assert not m.backbone.training                       # eval-locked under .train()
    trainable = {n for n, p in m.named_parameters() if p.requires_grad}
    assert trainable and all(n.startswith("head") for n in trainable)


def test_param_groups():
    m = _model(freeze=False)
    m.train()
    assert m.backbone.training
    groups = m.param_groups(1e-3, backbone_lr_scale=0.1)
    assert len(groups) == 2
    assert groups[0]["lr"] == 1e-4 and groups[1]["lr"] == 1e-3   # backbone scaled, head base


def test_param_groups_frozen():
    groups = _model(freeze=True).param_groups(1e-3)
    assert len(groups) == 1 and groups[0]["lr"] == 1e-3
