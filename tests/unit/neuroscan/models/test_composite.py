"""Equivalence-class tests for the Model = Backbone + Head composite (bd m69x.4).

A fake backbone (no CBraMod checkout needed) exercises the composite contract: forward shape/norm, freeze
policy, the cache hook, param groups, and every head in the zoo mapping a [B, C, S, d] grid to CLIP space."""
import numpy as np
import pytest
import torch

from neuroscan.models.composite import Backbone, Head, Heads, HeadSpec, Model

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
    head = Heads.build(HeadSpec(pool, pool), _D, _pos(), n_tok=_C * _S, embed_dim=_EMBED)
    return Model(_FakeBackbone(), head, freeze_backbone=freeze)


def test_forward_is_normalized_clip_shape():
    z = _model()(torch.randn(5, _C, _S * 4))
    assert z.shape == (5, _EMBED)
    assert torch.allclose(z.norm(dim=-1), torch.ones(5), atol=1e-5)


def test_frozen_backbone_has_no_trainable_params_and_stays_eval():
    m = _model(freeze=True)
    m.train()
    assert not m.backbone.training                       # eval-locked under .train()
    trainable = {n for n, p in m.named_parameters() if p.requires_grad}
    assert trainable and all(n.startswith("head") for n in trainable)


def test_unfrozen_backbone_trains_and_gives_two_param_groups():
    m = _model(freeze=False)
    m.train()
    assert m.backbone.training
    groups = m.param_groups(1e-3, backbone_lr_scale=0.1)
    assert len(groups) == 2
    assert groups[0]["lr"] == 1e-4 and groups[1]["lr"] == 1e-3   # backbone scaled, head base


def test_frozen_param_groups_is_head_only():
    groups = _model(freeze=True).param_groups(1e-3)
    assert len(groups) == 1 and groups[0]["lr"] == 1e-3


def test_encode_tokens_exposes_backbone_grid():
    m = _model()
    tokens = m.encode_tokens(torch.randn(5, _C, _S * 4))
    assert tokens.shape == (5, _C, _S, _D)               # the cacheable [B, C, S, d] grid


def test_every_head_maps_grid_to_clip():
    grid = torch.randn(4, _C, _S, _D)
    for pool in ("mean", "attn", "flat", "pos_attn", "topo", "gcn"):
        head = Heads.build(HeadSpec(pool, pool), _D, _pos(), n_tok=_C * _S, embed_dim=_EMBED)
        out = head(grid)
        assert out.shape == (4, _EMBED), pool


def test_flat_head_sizes_mlp_at_construction():
    """flat's MLP in-dim is n_tok·d — it must exist BEFORE the optimizer is built (a lazy init would leave its
    params out of the optimizer and it would never train). Requires n_tok; errors without it."""
    head = Heads.build(HeadSpec("flat", "flat"), _D, _pos(), n_tok=_C * _S, embed_dim=_EMBED)
    assert isinstance(head, Head)
    trainable = [p for p in head.parameters() if p.requires_grad]
    assert trainable and any(p.shape[-1] == _C * _S * _D for p in trainable)   # first linear sees all tokens
    with pytest.raises(ValueError, match="flat pool needs n_tok"):
        Heads.build(HeadSpec("flat", "flat"), _D, _pos(), embed_dim=_EMBED)
