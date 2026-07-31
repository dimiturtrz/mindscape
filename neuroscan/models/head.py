"""The `Head` stage of the EEG→CLIP composite — token grid to a point in CLIP space.

A `Head` folds a `[B, C, S, d]` token grid to `[B, embed_dim]` (pre-normalization; `Model` L2-normalizes).
`Head` also owns the zoo: `Head.build` constructs an arm from a `HeadSpec`, and the shared operators
(`mlp`/`spatial`/`require_pos`/`topo_weights`/`adjacency`) are staticmethods the concrete heads ride on. The
geometry heads (`pos_attn`/`topo`/`gcn`) are SPATIAL — they collapse S then fold the C channel tokens by
electrode geometry; `temporal` folds the time axis instead.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import override

import numpy as np
import torch
import torch.nn.functional as F
from jaxtyping import Float
from torch import Tensor, nn

_EMBED = 512          # CLIP dim (the retrieval target width)
_NHEAD = 4            # pos_attn: transformer heads
_GRID = 16            # topo: scalp interpolation grid (H=W)
_RBF_SIGMA = 0.2      # topo: RBF interp width on the unit-disk montage
_K_GCN = 8            # gcn: electrode-adjacency kNN degree


@dataclass
class HeadSpec:
    """One head arm: pooling strategy + MLP width. `pool`: mean | attn | flat | pos_attn | topo | gcn.
    `hidden=0` = a bare linear map (the probe floor). `grid`/`rbf_sigma` tune the topo scalp-image resolution
    + RBF width (topo only)."""
    name: str
    pool: str
    hidden: int = 512
    dropout: float = 0.5
    grid: int = _GRID
    rbf_sigma: float = _RBF_SIGMA


@dataclass
class HeadContext:
    """Sizing + geometry context shared by every head constructor: token width `d`, electrode positions
    `pos [C, 2]` on the unit-disk scalp (None for geometry-blind heads), and the CLIP output width."""
    d: int
    pos: np.ndarray | None
    embed_dim: int = _EMBED


class Head(nn.Module):
    """[B, C, S, d] token grid -> [B, embed_dim] embedding (pre-normalization; `Model` L2-normalizes). Also the
    zoo factory + shared geometry operators (montage-derived, computed once), folded in as staticmethods."""

    @override
    def forward(self, tokens: Float[Tensor, "n ch s d"]) -> Float[Tensor, "n d_embed"]:
        raise NotImplementedError

    @staticmethod
    def build(spec: HeadSpec, context: HeadContext, n_tok: int | None = None) -> "Head":
        """A `Head` for this arm, sized to the context (token width, electrode positions, CLIP dim).
        `n_tok` (= C·S) is required only by the flat pool (its MLP in-dim is n_tok·d);
        mean/attn/geometry/temporal ignore it."""
        geometry = {"pos_attn": PosAttnHead, "topo": TopoHead, "gcn": GcnHead}
        if spec.pool in geometry:
            return geometry[spec.pool](spec, context)
        if spec.pool == "temporal":
            return TemporalConvHead(spec, context)
        return TokenHead(spec, context, n_tok)

    @staticmethod
    def build_mlp(in_dim: int, hidden: int, dropout: float, embed_dim: int) -> nn.Module:
        """Shared head tail: bare linear (hidden=0) or one GELU-MLP block -> CLIP dim."""
        if hidden == 0:
            return nn.Linear(in_dim, embed_dim)
        return nn.Sequential(nn.Linear(in_dim, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, embed_dim))

    @staticmethod
    def spatial(tokens: Float[Tensor, "n ch s d"]) -> Float[Tensor, "n ch d"]:
        """Collapse the S temporal tokens -> [B, C, d] for the channel-geometry heads (spatial by design; at
        S=1 this is a squeeze, so the frozen-head-search numbers reproduce exactly)."""
        return tokens.mean(dim=2)

    @staticmethod
    def require_pos(context: HeadContext) -> Float[np.ndarray, "ch 2"]:
        """Geometry heads need electrode positions — one home for the shared invariant."""
        if context.pos is None:
            raise ValueError("this head requires electrode positions")
        return context.pos

    @staticmethod
    def topo_weights(pos: Float[np.ndarray, "ch 2"], grid: int, sigma: float) -> Float[Tensor, "hw ch"]:
        """Fixed RBF interpolation operator `[H·W, C]` mapping C electrode features onto a `grid×grid` scalp
        image (Bashivan 2016), RBF width `sigma`. Computed once from the montage; applied per batch as one einsum."""
        axis = np.linspace(-1.0, 1.0, grid)
        gx, gy = np.meshgrid(axis, axis)
        cells = np.stack([gx.ravel(), gy.ravel()], axis=1)                   # [H·W, 2]
        d2 = ((cells[:, None, :] - pos[None, :, :]) ** 2).sum(-1)            # [H·W, C]
        w = np.exp(-d2 / (2 * sigma ** 2))
        w = w / (w.sum(1, keepdims=True) + 1e-8)
        return torch.tensor(w, dtype=torch.float32)

    @staticmethod
    def adjacency(pos: Float[np.ndarray, "ch 2"], k: int = _K_GCN) -> Float[Tensor, "ch ch"]:
        """Symmetric-normalized adjacency Â = D^-1/2 (A+I) D^-1/2 [C, C] from a kNN graph over the electrode
        positions — the fixed message-passing operator for the GCN head."""
        d2 = ((pos[:, None, :] - pos[None, :, :]) ** 2).sum(-1)               # [C, C] pairwise sq-distance
        knn = np.argsort(d2, axis=1)[:, 1:k + 1]                              # k nearest electrodes (exclude self)
        a = np.zeros_like(d2)
        np.put_along_axis(a, knn, 1.0, axis=1)
        a = np.maximum(a, a.T) + np.eye(len(pos))                            # symmetric + self-loops
        dinv = np.diag(1.0 / np.sqrt(a.sum(1)))
        return torch.tensor(dinv @ a @ dinv, dtype=torch.float32)


class TokenHead(Head):
    """Geometry-blind head: pool the C·S tokens (mean / learned attention / unordered flatten) then MLP. The
    flat pool needs `n_tok` (= C·S) up front to size its MLP in-dim (n_tok·d) at construction — so every param
    exists before the optimizer is built."""

    def __init__(self, spec: HeadSpec, context: HeadContext, n_tok: int | None):
        super().__init__()
        self.pool = spec.pool
        self.attn = nn.Linear(context.d, 1) if spec.pool == "attn" else None
        if spec.pool == "flat":
            if n_tok is None:
                raise ValueError("flat pool needs n_tok (= C·S) to size its MLP")
            in_dim = n_tok * context.d
        else:
            in_dim = context.d                                   # mean/attn collapse the tokens to d
        self.mlp = Head.build_mlp(in_dim, spec.hidden, spec.dropout, context.embed_dim)

    @override
    def forward(self, tokens: Float[Tensor, "n ch s d"]) -> Float[Tensor, "n d_embed"]:
        f = tokens.flatten(1, 2)                          # [B, C·S, d]
        if self.pool == "mean":
            z = f.mean(dim=1)
        elif self.pool == "attn":
            if self.attn is None:
                raise RuntimeError("attn pool head requires an attn module")
            z = (self.attn(f).softmax(dim=1) * f).sum(dim=1)
        else:                                            # flat: unordered bag of all tokens
            z = f.flatten(1)
        return self.mlp(z)


class PosAttnHead(Head):
    """Electrode-position embedding + self-attention over the C channel tokens (geometry via learned pos)."""

    def __init__(self, spec: HeadSpec, context: HeadContext):
        super().__init__()
        self.register_buffer("pos", torch.tensor(context.pos, dtype=torch.float32))          # [C, 2]
        self.pos_proj = nn.Linear(2, context.d)
        self.enc = nn.TransformerEncoderLayer(context.d, _NHEAD, dim_feedforward=spec.hidden or context.d,
                                              dropout=spec.dropout, batch_first=True, activation="gelu")
        self.mlp = Head.build_mlp(context.d, spec.hidden, spec.dropout, context.embed_dim)

    @override
    def forward(self, tokens: Float[Tensor, "n ch s d"]) -> Float[Tensor, "n d_embed"]:
        f = Head.spatial(tokens)                          # [B, C, d]
        return self.mlp(self.enc(f + self.pos_proj(self.pos)).mean(dim=1))


class TopoHead(Head):
    """RBF-interpolate the C electrode features onto a scalp image, then 2D-convolve (Bashivan 2016)."""

    def __init__(self, spec: HeadSpec, context: HeadContext):
        super().__init__()
        pos = Head.require_pos(context)
        self.grid = spec.grid
        self.register_buffer("wtopo", Head.topo_weights(pos, spec.grid, spec.rbf_sigma))   # [H·W, C]
        self.conv = nn.Sequential(
            nn.Conv2d(context.d, 64, 3, padding=1), nn.GELU(),
            nn.Conv2d(64, 64, 3, stride=2, padding=1), nn.GELU(), nn.AdaptiveAvgPool2d(1))
        self.mlp = Head.build_mlp(64, spec.hidden, spec.dropout, context.embed_dim)

    @override
    def forward(self, tokens: Float[Tensor, "n ch s d"]) -> Float[Tensor, "n d_embed"]:
        f = Head.spatial(tokens)                          # [B, C, d]
        b, _, d = f.shape
        interp = torch.einsum("hc,bcd->bhd", self.wtopo, f)                          # [B, H·W, d]
        z = self.conv(interp.transpose(1, 2).reshape(b, d, self.grid, self.grid)).flatten(1)
        return self.mlp(z)


class GcnHead(Head):
    """2-layer graph conv over the electrode-adjacency graph (signal mixes between adjacent electrodes)."""

    def __init__(self, spec: HeadSpec, context: HeadContext):
        super().__init__()
        pos = Head.require_pos(context)
        self.register_buffer("adj", Head.adjacency(pos))                            # [C, C] normalized Â
        self.gcn1 = nn.Linear(context.d, context.d)
        self.gcn2 = nn.Linear(context.d, context.d)
        self.mlp = Head.build_mlp(context.d, spec.hidden, spec.dropout, context.embed_dim)

    @override
    def forward(self, tokens: Float[Tensor, "n ch s d"]) -> Float[Tensor, "n d_embed"]:
        f = Head.spatial(tokens)                          # [B, C, d]
        h = F.gelu(self.gcn1(torch.einsum("ck,bkd->bcd", self.adj, f)))              # Â X W₁
        h = F.gelu(self.gcn2(torch.einsum("ck,bkd->bcd", self.adj, h)))              # second graph-conv layer
        return self.mlp(h.mean(dim=1))                                               # readout over electrodes


class TemporalConvHead(Head):
    """The TEMPORAL analog of the geometry heads: when the grid's first axis is time (a finer-patching backbone
    like EEGPT emits `[B, N_time, embed_num, d]`), process the ordered token sequence with a 1D conv instead of
    collapsing it. Flattens (time, summary) into one ordered `N·embed_num` sequence — keeping the summary
    tokens — then convolves along it (kernel 3, size-agnostic) and reads out — LATE aggregation that preserves
    the temporal order a global mean or flat bag throws away. Folds the TIME axis (vs geometry's ELECTRODE axis)."""

    def __init__(self, spec: HeadSpec, context: HeadContext):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(context.d, 64, 3, padding=1), nn.GELU(),
            nn.Conv1d(64, 64, 3, padding=1), nn.GELU(), nn.AdaptiveAvgPool1d(1))
        self.mlp = Head.build_mlp(64, spec.hidden, spec.dropout, context.embed_dim)

    @override
    def forward(self, tokens: Float[Tensor, "n ch s d"]) -> Float[Tensor, "n d_embed"]:
        x = tokens.flatten(1, 2)                          # [B, N·embed_num, d] — keep the summary tokens as
        z = self.conv(x.transpose(1, 2)).flatten(1)       # ordered positions; conv over the token sequence -> [B, 64]
        return self.mlp(z)
