"""Equivalence-class tests for the Head zoo (bd m69x.4).

Tests for the Head factory, static operators, and all concrete head implementations
mapping a [B, C, S, d] grid to CLIP space."""
import numpy as np
import pytest
import torch

from neuroscan.models.head import Head, HeadContext, HeadSpec

np.random.seed(0)
torch.manual_seed(0)

_C, _S, _D, _EMBED = 8, 3, 16, 32


def _pos():
    return np.random.RandomState(0).randn(_C, 2) * 0.4


def test_build():
    """Head.build returns a Head subclass instance for each pool strategy."""
    spec = HeadSpec("mean", "mean")
    context = HeadContext(_D, _pos(), _EMBED)
    head = Head.build(spec, context, n_tok=_C * _S)
    assert isinstance(head, Head)


def test_every_head_maps_grid_to_clip():
    grid = torch.randn(4, _C, _S, _D)
    for pool in ("mean", "attn", "flat", "pos_attn", "topo", "gcn", "temporal"):
        head = Head.build(HeadSpec(pool, pool), HeadContext(_D, _pos(), _EMBED), n_tok=_C * _S)
        out = head(grid)
        assert out.shape == (4, _EMBED), pool


def test_token_head_forward():
    """TokenHead.forward pools tokens and passes through MLP."""
    from neuroscan.models.head import TokenHead
    spec = HeadSpec("mean", "mean")
    context = HeadContext(_D, _pos(), _EMBED)
    head = TokenHead(spec, context, n_tok=_C * _S)
    grid = torch.randn(4, _C, _S, _D)
    out = head.forward(grid)
    assert out.shape == (4, _EMBED)


def test_pos_attn_head_forward():
    """PosAttnHead.forward applies position embedding and attention."""
    from neuroscan.models.head import PosAttnHead
    spec = HeadSpec("pos_attn", "pos_attn")
    context = HeadContext(_D, _pos(), _EMBED)
    head = PosAttnHead(spec, context)
    grid = torch.randn(4, _C, _S, _D)
    out = head.forward(grid)
    assert out.shape == (4, _EMBED)


def test_topo_head_forward():
    """TopoHead.forward interpolates to scalp image and convolves."""
    from neuroscan.models.head import TopoHead
    spec = HeadSpec("topo", "topo")
    context = HeadContext(_D, _pos(), _EMBED)
    head = TopoHead(spec, context)
    grid = torch.randn(4, _C, _S, _D)
    out = head.forward(grid)
    assert out.shape == (4, _EMBED)


def test_gcn_head_forward():
    """GcnHead.forward applies graph convolution."""
    from neuroscan.models.head import GcnHead
    spec = HeadSpec("gcn", "gcn")
    context = HeadContext(_D, _pos(), _EMBED)
    head = GcnHead(spec, context)
    grid = torch.randn(4, _C, _S, _D)
    out = head.forward(grid)
    assert out.shape == (4, _EMBED)


def test_temporal_conv_head_forward():
    """TemporalConvHead.forward convolves along the token sequence."""
    from neuroscan.models.head import TemporalConvHead
    spec = HeadSpec("temporal", "temporal")
    context = HeadContext(_D, _pos(), _EMBED)
    head = TemporalConvHead(spec, context)
    grid = torch.randn(4, _C, _S, _D)
    out = head.forward(grid)
    assert out.shape == (4, _EMBED)


def test_temporal_head_uses_time_order():
    """The temporal head convolves ALONG the first (time) axis — reversing time changes its output, unlike a
    global-mean/flat head. Guards that it actually processes the sequence the S-token backbones expose."""
    head = Head.build(HeadSpec("temporal", "temporal"), HeadContext(_D, _pos(), _EMBED)).eval()
    grid = torch.randn(2, _C, _S, _D)
    with torch.no_grad():
        forward, reversed_ = head(grid), head(grid.flip(1))
    assert not torch.allclose(forward, reversed_, atol=1e-4)   # order-sensitive (temporal, not a bag)


def test_flat_head_sizes_mlp_at_construction():
    """flat's MLP in-dim is n_tok·d — it must exist BEFORE the optimizer is built (a lazy init would leave its
    params out of the optimizer and it would never train). Requires n_tok; errors without it."""
    head = Head.build(HeadSpec("flat", "flat"), HeadContext(_D, _pos(), _EMBED), n_tok=_C * _S)
    assert isinstance(head, Head)
    trainable = [p for p in head.parameters() if p.requires_grad]
    assert trainable and any(p.shape[-1] == _C * _S * _D for p in trainable)   # first linear sees all tokens
    with pytest.raises(ValueError, match="flat pool needs n_tok"):
        Head.build(HeadSpec("flat", "flat"), HeadContext(_D, _pos(), _EMBED))


def test_build_mlp():
    """Head.mlp with hidden=0 is a bare linear; hidden>0 is a GELU block."""
    bare = Head.build_mlp(32, 0, 0.5, 64)
    assert isinstance(bare, torch.nn.Linear)
    gelu_block = Head.build_mlp(32, 128, 0.5, 64)
    assert isinstance(gelu_block, torch.nn.Sequential)
    assert len(list(gelu_block)) == 4  # Linear, GELU, Dropout, Linear


def test_spatial():
    """Head.spatial collapses the S dimension (mean over tokens)."""
    tokens = torch.randn(4, 8, 3, 16)  # [B, C, S, d]
    spatial = Head.spatial(tokens)
    assert spatial.shape == (4, 8, 16)  # [B, C, d]
    assert torch.allclose(spatial, tokens.mean(dim=2))


def test_require_pos():
    """Head.require_pos raises if positions are None."""
    context = HeadContext(_D, None, _EMBED)
    with pytest.raises(ValueError, match="this head requires electrode positions"):
        Head.require_pos(context)


def test_topo_weights():
    """Head.topo_weights produces an RBF interpolation operator [H·W, C]."""
    pos = _pos()
    grid = 8
    sigma = 0.2
    w = Head.topo_weights(pos, grid, sigma)
    assert w.shape == (grid * grid, _C)
    assert torch.all(w >= 0)                          # RBF weights are non-negative
    # each cell's C weights are row-normalized -> sum to ~1 near electrodes, ~0 for cells far from all of them
    assert torch.all(w.sum(dim=1) <= 1.0 + 1e-6)
    assert w.sum(dim=1).max() > 0.9                    # at least one cell sits on/near an electrode


def test_adjacency():
    """Head.adjacency builds a symmetric-normalized kNN adjacency matrix [C, C]."""
    pos = _pos()
    adj = Head.adjacency(pos, k=3)
    assert adj.shape == (_C, _C)
    assert torch.allclose(adj, adj.t())  # symmetric
    assert torch.all(adj.diag() > 0)  # self-loops
