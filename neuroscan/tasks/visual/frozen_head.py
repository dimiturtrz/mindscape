"""Frozen-backbone head-architecture search (bd nm5) — the FAST perception lever.

The fine-tuned CBraMod win (single 2.38%) adapted all 4.9M backbone params — expensive, and per the intuition
that pretrained-feature *adaptation* is lower-yield than *head engineering*, likely the wrong lever. The frozen
LINEAR probe hit chance (0.63%), but that was a LAZY head (mean-pool → linear), NOT evidence the frozen
features are capped. This asks the real question: can a well-engineered head on the SAME frozen features reach
the fine-tuned number — cheaply?

Speed comes from precomputing CBraMod's frozen token grid ONCE (backbone eval, no grad) and caching it in RAM;
every head then trains on cached features at seconds/epoch (no backbone forward per step), so many head
architectures are swept in one run. The backbone emits `[B, C, S, d]`; at 200 Hz / 1.0 s the epoch is a single
200-point patch (S=1), so the tokens are the 63 channel embeddings — the head's job is to pool + map them to
CLIP space.

    python -m neuroscan.tasks.visual.frozen_head --train 1 2 3 4 --test 5

Each arm is matched (same cached features, same InfoNCE + leak-free val); compare single-trial top-1 to the
frozen-probe floor (mean_lin ~0.6%) and the fine-tune reference (2.38%). A head that clears NICE (~1.6%) frozen
= head engineering, not backbone adaptation, is the lever (and a far cheaper deploy: one frozen backbone).
"""
from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from core.data.eeg import things_eeg2 as things
from core.features.eeg.montage import eeg_positions
from neuroscan.models.foundation import _load_backbone
from neuroscan.models.nice import clip_infonce, retrieval_topk
from neuroscan.tasks.visual import clip_targets

logger = logging.getLogger(__name__)

_PATCH = 200          # CBraMod points/patch — implies 200 Hz epochs
_RESAMPLE = 200.0
_EMBED = 512          # CLIP dim
_FEAT_BATCH = 256     # backbone forward batch for the one-time precompute
_GRID = 16            # topo_cnn: scalp interpolation grid (H=W)
_RBF_SIGMA = 0.2      # topo_cnn: RBF interp width on the unit-disk montage
_NHEAD = 4            # pos_attn: transformer heads (d_model=200 divisible)


def _mlp(in_dim: int, hidden: int, dropout: float) -> nn.Module:
    """Shared head tail: bare linear (hidden=0) or one GELU-MLP block → CLIP dim."""
    if hidden == 0:
        return nn.Linear(in_dim, _EMBED)
    return nn.Sequential(nn.Linear(in_dim, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, _EMBED))


def _topo_weights(pos: np.ndarray) -> torch.Tensor:
    """Fixed RBF interpolation operator `[H·W, C]` mapping the C electrode features onto a `_GRID×_GRID` scalp
    image (Bashivan 2016). Computed once from the montage; applied per batch as one einsum."""
    axis = np.linspace(-1.0, 1.0, _GRID)
    gx, gy = np.meshgrid(axis, axis)
    grid = np.stack([gx.ravel(), gy.ravel()], axis=1)                    # [H·W, 2]
    d2 = ((grid[:, None, :] - pos[None, :, :]) ** 2).sum(-1)             # [H·W, C]
    w = np.exp(-d2 / (2 * _RBF_SIGMA ** 2))
    w = w / (w.sum(1, keepdims=True) + 1e-8)
    return torch.tensor(w, dtype=torch.float32)


@dataclass
class HeadSpec:
    """One head arm: how to pool the C·S token grid + the MLP depth. `pool`: mean | attn | flat.
    `hidden=0` = a bare linear map (the probe floor)."""
    name: str
    pool: str
    hidden: int = 512
    dropout: float = 0.5


@dataclass
class FitCfg:
    """Head-training knobs shared by every arm (head-only fit — a higher LR than a backbone fine-tune is fine)."""
    epochs: int = 60
    lr: float = 1e-3
    seed: int = 0


_ARMS = [
    HeadSpec("mean_lin", "mean", hidden=0),        # frozen linear-probe floor (~0.6%, chance)
    HeadSpec("flat_mlp", "flat", hidden=1024),     # best geometry-blind (unordered bag of tokens, ~1.2%)
    HeadSpec("pos_attn", "pos_attn", hidden=512),  # geometry: electrode-position embedding + self-attention
    HeadSpec("topo_cnn", "topo", hidden=512),      # geometry: scalp-grid RBF interpolation + 2D CNN (Bashivan)
]


class Head(nn.Module):
    """Frozen-feature head: fold the `[B, n_tok, d]` token grid to a vector, then MLP → CLIP dim (L2-normed).

    Geometry-blind pools — `mean` / `attn` (bottleneck) and `flat` (unordered bag) — ignore where each
    electrode sits. Geometry pools use the scalp positions `pos [C,2]`: `pos_attn` adds a learned positional
    embedding then self-attends; `topo` interpolates the electrode features onto a 2D scalp image and convolves
    (both require the C electrode tokens, i.e. S=1)."""

    def __init__(self, spec: HeadSpec, n_tok: int, d: int, pos: np.ndarray):
        super().__init__()
        self.pool = spec.pool
        self.attn = nn.Linear(d, 1) if spec.pool == "attn" else None
        if spec.pool == "pos_attn":
            self.register_buffer("pos", torch.tensor(pos, dtype=torch.float32))          # [C, 2]
            self.pos_proj = nn.Linear(2, d)
            self.enc = nn.TransformerEncoderLayer(d, _NHEAD, dim_feedforward=spec.hidden or d,
                                                  dropout=spec.dropout, batch_first=True, activation="gelu")
            self.mlp = _mlp(d, spec.hidden, spec.dropout)
        elif spec.pool == "topo":
            self.register_buffer("wtopo", _topo_weights(pos))                            # [H·W, C]
            self.conv = nn.Sequential(
                nn.Conv2d(d, 64, 3, padding=1), nn.GELU(),
                nn.Conv2d(64, 64, 3, stride=2, padding=1), nn.GELU(), nn.AdaptiveAvgPool2d(1))
            self.mlp = _mlp(64, spec.hidden, spec.dropout)
        else:
            self.mlp = _mlp(n_tok * d if spec.pool == "flat" else d, spec.hidden, spec.dropout)

    def forward(self, f: torch.Tensor) -> torch.Tensor:
        if self.pool == "mean":
            z = f.mean(dim=1)
        elif self.pool == "attn":
            z = (self.attn(f).softmax(dim=1) * f).sum(dim=1)
        elif self.pool == "flat":
            z = f.flatten(1)
        elif self.pool == "pos_attn":
            z = self.enc(f + self.pos_proj(self.pos)).mean(dim=1)
        else:                                                                            # topo
            b, _, d = f.shape
            grid = torch.einsum("hc,bcd->bhd", self.wtopo, f)                            # [B, H·W, d]
            z = self.conv(grid.transpose(1, 2).reshape(b, d, _GRID, _GRID)).flatten(1)
        return F.normalize(self.mlp(z), dim=-1)


@torch.no_grad()
def _features(backbone, eeg: np.ndarray, device: str) -> torch.Tensor:
    """Frozen CBraMod token grid for every epoch, `[N, C·S, d]` float16 on CPU (the one-time cost)."""
    out = []
    for i in range(0, len(eeg), _FEAT_BATCH):
        x = torch.tensor(eeg[i:i + _FEAT_BATCH]).to(device)
        b, c, t = x.shape
        s = t // _PATCH
        x = x[:, :, :s * _PATCH]
        x = (x - x.mean(-1, keepdim=True)) / (x.std(-1, keepdim=True) + 1e-6)   # match foundation.forward
        feats = backbone(x.reshape(b, c, s, _PATCH))                            # [B, C, S, d]
        out.append(feats.reshape(b, c * s, feats.shape[-1]).half().cpu())
    return torch.cat(out)


def _clip_targets(image_files: np.ndarray, split: str) -> np.ndarray:
    by_file = clip_targets.embeddings_by_file(split)
    return np.stack([by_file[name] for name in image_files]).astype(np.float32)


def _val_concepts(concept: np.ndarray, targets: np.ndarray, seed: int, fraction: float):
    """Hold out a fraction of TRAIN concepts as a leak-free early-stop bank (mirrors train_nice._val_split)."""
    rng = np.random.default_rng(seed)
    concepts = np.unique(concept)
    val = sorted(rng.choice(concepts, max(1, int(len(concepts) * fraction)), replace=False).tolist())
    remap = {c: i for i, c in enumerate(val)}
    bank = np.stack([targets[concept == c].mean(0) for c in val]).astype(np.float32)
    bank /= (np.linalg.norm(bank, axis=1, keepdims=True) + 1e-8)
    mask = np.isin(concept, val)
    return ~mask, mask, np.array([remap[c] for c in concept[mask]]), bank


@dataclass
class Cache:
    """Precomputed frozen features + targets/labels for one train/test pair — shared across every head arm."""
    tr_feat: torch.Tensor
    tr_tgt: torch.Tensor
    tr_concept: np.ndarray
    test_feat: torch.Tensor
    test_concept: np.ndarray
    test_bank: torch.Tensor
    n_tok: int
    d: int
    pos: np.ndarray          # [C, 2] electrode positions on the unit-disk scalp (for the geometry heads)


def _build_cache(train_subjects: list[int], test_subject: int, device: str) -> Cache:
    backbone = _load_backbone().to(device).eval()
    for p in backbone.parameters():
        p.requires_grad = False
    tr_eeg, tr_concept, tr_files, _ = _load(train_subjects, "training")
    te_eeg, te_concept, _, _ = _load([test_subject], "test")
    logger.info(f"precompute: train {tr_eeg.shape} · test {te_eeg.shape}")
    tr_feat = _features(backbone, tr_eeg, device)
    test_feat = _features(backbone, te_eeg, device)
    tr_tgt = torch.tensor(_clip_targets(tr_files, "training"))
    test_bank = torch.tensor(clip_targets.concept_prototypes("test"))
    pos = eeg_positions(things.channels())          # [C, 2], same channel order as the features
    logger.info(f"features: train {tuple(tr_feat.shape)} · test {tuple(test_feat.shape)} (float16, RAM)")
    return Cache(tr_feat, tr_tgt, tr_concept, test_feat, te_concept, test_bank,
                 n_tok=tr_feat.shape[1], d=tr_feat.shape[2], pos=pos)


def _load(subjects: list[int], split: str):
    eeg, concept, files, meta = things.get_epochs(
        subjects, things.ThingsEpochCfg(split=split, resample=_RESAMPLE), n_jobs=4)
    return eeg, concept, files, meta


@torch.no_grad()
def _retrieval(head, feat: torch.Tensor, concept: np.ndarray, bank: torch.Tensor, device: str) -> dict:
    head.eval()
    emb = torch.cat([head(feat[i:i + 4096].float().to(device)).cpu() for i in range(0, len(feat), 4096)])
    labels = torch.tensor(concept)
    single = retrieval_topk(emb, bank, labels)
    n = int(concept.max()) + 1
    averaged = torch.stack([F.normalize(emb[labels == c].mean(0), dim=-1) for c in range(n)])
    return {"single_trial": single, "concept_avg": retrieval_topk(averaged, bank, torch.arange(n))}


def _train_arm(spec: HeadSpec, cache: Cache, device: str, cfg: FitCfg) -> dict:
    torch.manual_seed(cfg.seed)
    fit_mask, val_mask, val_lab, val_bank = _val_concepts(cache.tr_concept, cache.tr_tgt.numpy(), cfg.seed, 0.1)
    fit_idx = np.where(fit_mask)[0]
    val_feat, val_bank_t = cache.tr_feat[val_mask], torch.tensor(val_bank)
    head = Head(spec, cache.n_tok, cache.d, cache.pos).to(device)
    logit_scale = nn.Parameter(torch.tensor(np.log(1 / 0.07), dtype=torch.float32, device=device))
    opt = torch.optim.AdamW([*head.parameters(), logit_scale], lr=cfg.lr, weight_decay=1e-4)
    rng = np.random.default_rng(cfg.seed)
    best_val, best_state, best_ep = -1.0, None, -1
    for ep in range(cfg.epochs):
        head.train()
        perm = rng.permutation(fit_idx)
        for i in range(0, len(perm) - 512, 512):
            idx = perm[i:i + 512]
            f = cache.tr_feat[idx].float().to(device)
            tgt = F.normalize(cache.tr_tgt[idx].to(device), dim=-1)
            opt.zero_grad()
            loss = clip_infonce(head(f), tgt, logit_scale.exp().clamp(max=100))
            loss.backward()
            opt.step()
        val_top1 = _retrieval(head, val_feat, val_lab, val_bank_t, device)["single_trial"][1]
        if val_top1 > best_val:
            best_val, best_ep = val_top1, ep
            best_state = {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}
    head.load_state_dict(best_state)
    test = _retrieval(head, cache.test_feat, cache.test_concept, cache.test_bank, device)
    return {"arm": spec.name, "best_val_epoch": best_ep, "val_top1": best_val, **test}


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    for lib in ("mne", "moabb", "braindecode"):
        logging.getLogger(lib).setLevel(logging.WARNING)
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--train", type=int, nargs="+", default=[1, 2, 3, 4])
    ap.add_argument("--test", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    cache = _build_cache(args.train, args.test, device)
    fit = FitCfg(epochs=args.epochs, lr=args.lr, seed=args.seed)
    logger.info(f"\nhead sweep · train={args.train} test={args.test} · {args.epochs}ep lr{args.lr} · "
                f"chance {1 / (int(cache.test_concept.max()) + 1):.3f}")
    results = []
    for spec in _ARMS:
        r = _train_arm(spec, cache, device, fit)
        s, a = r["single_trial"], r["concept_avg"]
        logger.info(f"  {spec.name:9s} (val {r['val_top1']*100:.2f}% ep{r['best_val_epoch']:2d})  "
                    f"single {s[1]*100:.2f}%/{s[5]*100:.2f}%  concept {a[1]*100:.2f}%/{a[5]*100:.2f}%")
        results.append(r)
    if args.out:
        Path(args.out).write_text(json.dumps({"train": args.train, "test": args.test, "arms": results}, indent=2))


if __name__ == "__main__":
    main()
