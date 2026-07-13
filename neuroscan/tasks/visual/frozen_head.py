"""Frozen-backbone head-architecture search (bd nm5) — the FAST perception lever.

The fine-tuned CBraMod win (single 2.38%) adapted all 4.9M backbone params — expensive, and per the intuition
that pretrained-feature *adaptation* is lower-yield than *head engineering*, likely the wrong lever. The frozen
LINEAR probe hit chance (0.63%), but that was a LAZY head (mean-pool → linear), NOT evidence the frozen
features are capped. This asks the real question: can a well-engineered head on the SAME frozen features reach
the fine-tuned number — cheaply?

Speed comes from precomputing the backbone's frozen token grid ONCE (eval, no grad) and caching it to DISK
(keyed by backbone/split/subject-set under <data>/cache); every head then trains on cached features at
seconds/epoch (no backbone forward per step), and a REUSED sweep loads the cache in seconds — no mne, no GPU
backbone pass. `--backbone` selects the frozen model via `Foundation.load_backbone` (the swap seam, bd m69x):
CBraMod emits `[B, C, S, d]` and at 200 Hz / 1.0 s the epoch is a single 200-point patch (S=1), so the tokens
are the 63 channel embeddings — the head pools + maps them to CLIP space. A finer-patching backbone gives S>1
(the sub-patch temporal timing CBraMod's single patch buries).

    python -m neuroscan.tasks.visual.frozen_head --train 1 2 3 4 --test 5 --backbone cbramod

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

from core.config import Config
from core.data.eeg import things_eeg2 as things
from core.features.eeg.montage import EegMontage
from neuroscan.models.foundation import Foundation, LoadedBackbone
from neuroscan.models.nice import Nice
from neuroscan.tasks.visual import clip_targets

logger = logging.getLogger(__name__)

_EMBED = 512          # CLIP dim
_FEAT_BATCH = 256     # backbone forward batch for the one-time precompute
_GRID = 16            # topo_cnn: scalp interpolation grid (H=W)
_RBF_SIGMA = 0.2      # topo_cnn: RBF interp width on the unit-disk montage
_NHEAD = 4            # pos_attn: transformer heads (d_model=200 divisible)
_K_GCN = 8            # gcn: electrode-adjacency kNN degree (each electrode links its 8 nearest)


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
    HeadSpec("gcn", "gcn", hidden=512),            # geometry: electrode-adjacency graph message passing (2-layer GCN)
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
            self.mlp = FrozenHead._mlp(d, spec.hidden, spec.dropout)
        elif spec.pool == "topo":
            self.register_buffer("wtopo", FrozenHead._topo_weights(pos))                 # [H·W, C]
            self.conv = nn.Sequential(
                nn.Conv2d(d, 64, 3, padding=1), nn.GELU(),
                nn.Conv2d(64, 64, 3, stride=2, padding=1), nn.GELU(), nn.AdaptiveAvgPool2d(1))
            self.mlp = FrozenHead._mlp(64, spec.hidden, spec.dropout)
        elif spec.pool == "gcn":
            self.register_buffer("adj", FrozenHead._adjacency(pos))                      # [C, C] normalized Â
            self.gcn1 = nn.Linear(d, d)
            self.gcn2 = nn.Linear(d, d)
            self.mlp = FrozenHead._mlp(d, spec.hidden, spec.dropout)
        else:
            self.mlp = FrozenHead._mlp(n_tok * d if spec.pool == "flat" else d, spec.hidden, spec.dropout)

    def forward(self, f: torch.Tensor) -> torch.Tensor:
        if self.pool == "mean":
            z = f.mean(dim=1)
        elif self.pool == "attn":
            z = (self.attn(f).softmax(dim=1) * f).sum(dim=1)
        elif self.pool == "flat":
            z = f.flatten(1)
        elif self.pool == "pos_attn":
            z = self.enc(f + self.pos_proj(self.pos)).mean(dim=1)
        elif self.pool == "gcn":
            h = F.gelu(self.gcn1(torch.einsum("ck,bkd->bcd", self.adj, f)))              # Â X W₁, propagate+transform
            h = F.gelu(self.gcn2(torch.einsum("ck,bkd->bcd", self.adj, h)))              # second graph-conv layer
            z = h.mean(dim=1)                                                            # readout over electrodes
        else:                                                                            # topo
            b, _, d = f.shape
            grid = torch.einsum("hc,bcd->bhd", self.wtopo, f)                            # [B, H·W, d]
            z = self.conv(grid.transpose(1, 2).reshape(b, d, _GRID, _GRID)).flatten(1)
        return F.normalize(self.mlp(z), dim=-1)


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


class FrozenHead:
    """Frozen-backbone head-architecture search — the free helpers folded in as staticmethods (public names
    kept). Precomputes CBraMod's frozen token grid once (`_build_cache`), then `_train_arm` fits each head
    architecture on the cached features."""

    @staticmethod
    def _mlp(in_dim: int, hidden: int, dropout: float) -> nn.Module:
        """Shared head tail: bare linear (hidden=0) or one GELU-MLP block → CLIP dim."""
        if hidden == 0:
            return nn.Linear(in_dim, _EMBED)
        return nn.Sequential(nn.Linear(in_dim, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, _EMBED))

    @staticmethod
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

    @staticmethod
    def _adjacency(pos: np.ndarray, k: int = _K_GCN) -> torch.Tensor:
        """Symmetric-normalized adjacency Â = D^-1/2 (A+I) D^-1/2 [C, C] from a kNN graph over the electrode
        positions — the fixed message-passing operator for the GCN head (electrode geometry as a graph, so
        signal mixes between physically-adjacent electrodes)."""
        d2 = ((pos[:, None, :] - pos[None, :, :]) ** 2).sum(-1)               # [C, C] pairwise sq-distance
        knn = np.argsort(d2, axis=1)[:, 1:k + 1]                              # k nearest electrodes (exclude self)
        a = np.zeros_like(d2)
        np.put_along_axis(a, knn, 1.0, axis=1)
        a = np.maximum(a, a.T) + np.eye(len(pos))                            # symmetric + self-loops
        dinv = np.diag(1.0 / np.sqrt(a.sum(1)))
        return torch.tensor(dinv @ a @ dinv, dtype=torch.float32)

    @staticmethod
    @torch.no_grad()
    def _features(module: nn.Module, patch_points: int, eeg: np.ndarray, device: str) -> torch.Tensor:
        """Frozen token grid for every epoch, `[N, C·S, d]` float16 on CPU (the one-time cost). `patch_points`
        comes from the backbone (CBraMod 200 -> S=1 on a 1s epoch); a finer-patching backbone gives S>1."""
        out = []
        for i in range(0, len(eeg), _FEAT_BATCH):
            x = torch.tensor(eeg[i:i + _FEAT_BATCH]).to(device)
            b, c, t = x.shape
            s = t // patch_points
            x = x[:, :, :s * patch_points]
            x = (x - x.mean(-1, keepdim=True)) / (x.std(-1, keepdim=True) + 1e-6)   # match foundation.forward
            feats = module(x.reshape(b, c, s, patch_points))                       # [B, C, S, d]
            out.append(feats.reshape(b, c * s, feats.shape[-1]).half().cpu())
        return torch.cat(out)

    @staticmethod
    def _clip_targets(image_files: np.ndarray, split: str) -> np.ndarray:
        by_file = clip_targets.ClipTargets.embeddings_by_file(split)
        return np.stack([by_file[name] for name in image_files]).astype(np.float32)

    @staticmethod
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

    @staticmethod
    def _cache_path(backbone: str, subjects: list[int], split: str) -> Path:
        """One home per (backbone, split, subject-set) frozen-feature blob, out-of-repo under <data>/cache."""
        subj = "-".join(str(s) for s in sorted(subjects))
        return Config.data_root("cache") / "frozen_features" / f"{backbone}__{split}__{subj}.pt"

    @staticmethod
    def _loaded(backbone: str, device: str) -> LoadedBackbone:
        """Load + freeze a backbone onto `device` (only called on a cache miss — a full sweep off the cache
        never touches the backbone)."""
        lb = Foundation.load_backbone(backbone)
        module = lb.module.to(device).eval()
        for p in module.parameters():
            p.requires_grad = False
        return LoadedBackbone(module, lb.patch_points, lb.d_model, lb.sample_rate, lb.name)

    @staticmethod
    def _split_features(path: Path, subjects: list[int], split: str, loaded: LoadedBackbone | None,
                        device: str) -> tuple[torch.Tensor, np.ndarray, np.ndarray]:
        """Frozen features + (concept, files) for one split — from the disk cache if present (skips the eeg
        load AND the backbone forward), else compute through `loaded` and persist. Cache holds the
        backbone-independent concept/files too, so a reused sweep needs neither mne nor the GPU."""
        if path.exists():
            blob = torch.load(path, weights_only=False)   # our own cache (feat tensor + numpy concept/files)
            logger.info(f"cache hit {path.name}: {tuple(blob['feat'].shape)}")
            return blob["feat"], blob["concept"], blob["files"]
        eeg, concept, files, _ = FrozenHead._load(subjects, split, loaded.sample_rate)
        feat = FrozenHead._features(loaded.module, loaded.patch_points, eeg, device)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"feat": feat, "concept": concept, "files": files}, path)
        logger.info(f"cached {path.name}: {tuple(feat.shape)} (float16)")
        return feat, concept, files

    @staticmethod
    def _build_cache(train_subjects: list[int], test_subject: int, device: str,
                     backbone: str = "cbramod") -> Cache:
        tr_path = FrozenHead._cache_path(backbone, train_subjects, "training")
        te_path = FrozenHead._cache_path(backbone, [test_subject], "test")
        loaded = None if (tr_path.exists() and te_path.exists()) else FrozenHead._loaded(backbone, device)
        tr_feat, tr_concept, tr_files = FrozenHead._split_features(tr_path, train_subjects, "training", loaded, device)
        test_feat, te_concept, _ = FrozenHead._split_features(te_path, [test_subject], "test", loaded, device)
        tr_tgt = torch.tensor(FrozenHead._clip_targets(tr_files, "training"))
        test_bank = torch.tensor(clip_targets.ClipTargets.concept_prototypes("test"))
        pos = EegMontage.eeg_positions(things.ThingsEeg2.channels())   # [C, 2], same channel order as the features
        return Cache(tr_feat, tr_tgt, tr_concept, test_feat, te_concept, test_bank,
                     n_tok=tr_feat.shape[1], d=tr_feat.shape[2], pos=pos)

    @staticmethod
    def _load(subjects: list[int], split: str, sample_rate: float):
        eeg, concept, files, meta = things.ThingsEeg2.get_epochs(
            subjects, things.ThingsEpochCfg(split=split, resample=sample_rate), n_jobs=4)
        return eeg, concept, files, meta

    @staticmethod
    @torch.no_grad()
    def _retrieval(head, feat: torch.Tensor, concept: np.ndarray, bank: torch.Tensor, device: str) -> dict:
        head.eval()
        emb = torch.cat([head(feat[i:i + 4096].float().to(device)).cpu() for i in range(0, len(feat), 4096)])
        labels = torch.tensor(concept)
        single = Nice.retrieval_topk(emb, bank, labels)
        n = int(concept.max()) + 1
        averaged = torch.stack([F.normalize(emb[labels == c].mean(0), dim=-1) for c in range(n)])
        return {"single_trial": single, "concept_avg": Nice.retrieval_topk(averaged, bank, torch.arange(n))}

    @staticmethod
    def _train_arm(spec: HeadSpec, cache: Cache, device: str, cfg: FitCfg) -> dict:
        torch.manual_seed(cfg.seed)
        fit_mask, val_mask, val_lab, val_bank = FrozenHead._val_concepts(
            cache.tr_concept, cache.tr_tgt.numpy(), cfg.seed, 0.1)
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
                loss = Nice.clip_infonce(head(f), tgt, logit_scale.exp().clamp(max=100))
                loss.backward()
                opt.step()
            val_top1 = FrozenHead._retrieval(head, val_feat, val_lab, val_bank_t, device)["single_trial"][1]
            if val_top1 > best_val:
                best_val, best_ep = val_top1, ep
                best_state = {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}
        head.load_state_dict(best_state)
        test = FrozenHead._retrieval(head, cache.test_feat, cache.test_concept, cache.test_bank, device)
        return {"arm": spec.name, "best_val_epoch": best_ep, "val_top1": best_val, **test}


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    for lib in ("mne", "moabb", "braindecode"):
        logging.getLogger(lib).setLevel(logging.WARNING)
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--train", type=int, nargs="+", default=[1, 2, 3, 4])
    ap.add_argument("--test", type=int, default=5)
    ap.add_argument("--backbone", default="cbramod", help="frozen backbone (registry name in Foundation.load_backbone)")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    cache = FrozenHead._build_cache(args.train, args.test, device, args.backbone)
    fit = FitCfg(epochs=args.epochs, lr=args.lr, seed=args.seed)
    logger.info(f"\nhead sweep · backbone={args.backbone} train={args.train} test={args.test} · "
                f"{args.epochs}ep lr{args.lr} · chance {1 / (int(cache.test_concept.max()) + 1):.3f}")
    results = []
    for spec in _ARMS:
        r = FrozenHead._train_arm(spec, cache, device, fit)
        s, a = r["single_trial"], r["concept_avg"]
        logger.info(f"  {spec.name:9s} (val {r['val_top1']*100:.2f}% ep{r['best_val_epoch']:2d})  "
                    f"single {s[1]*100:.2f}%/{s[5]*100:.2f}%  concept {a[1]*100:.2f}%/{a[5]*100:.2f}%")
        results.append(r)
    if args.out:
        Path(args.out).write_text(json.dumps(
            {"backbone": args.backbone, "train": args.train, "test": args.test, "arms": results}, indent=2))


if __name__ == "__main__":
    main()
