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
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from core.config import Config
from core.data.eeg import things_eeg2 as things
from core.features.eeg.montage import EegMontage
from neuroscan.models.composite import Heads, HeadSpec
from neuroscan.models.foundation import Foundation, LoadedBackbone
from neuroscan.models.nice import Nice
from neuroscan.tasks.visual import clip_targets
from neuroscan.tracking import Tracking

logger = logging.getLogger(__name__)

_FEAT_BATCH = 256     # backbone forward batch for the one-time precompute
_TOPO_GRIDS = (12, 16, 24)         # topo mini-sweep (bd m69x.2): scalp-image resolution
_TOPO_SIGMAS = (0.1, 0.2, 0.35)    # topo mini-sweep: RBF interpolation width on the unit-disk montage


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


@dataclass
class _EvalSet:
    """One retrieval eval: the cached features to embed, their concept labels, and the candidate CLIP bank."""
    feat: torch.Tensor
    concept: np.ndarray
    bank: torch.Tensor


@dataclass
class Cache:
    """Precomputed frozen features + targets/labels for one train/test pair — shared across every head arm."""
    tr_feat: torch.Tensor
    tr_tgt: torch.Tensor
    tr_concept: np.ndarray
    test_feat: torch.Tensor
    test_concept: np.ndarray
    test_bank: torch.Tensor
    d: int
    pos: np.ndarray          # [C, 2] electrode positions on the unit-disk scalp (for the geometry heads)


class FrozenHead:
    """Frozen-backbone head-architecture search — the free helpers folded in as staticmethods (public names
    kept). Precomputes CBraMod's frozen token grid once (`_build_cache`), then `_train_arm` fits each head
    architecture on the cached features."""

    @staticmethod
    def _topo_arms() -> list[HeadSpec]:
        """The topo grid×sigma mini-sweep arms (bd m69x.2) — confirm topo_cnn's ~1.68 isn't left on the table
        by the interpolation resolution. Rides the disk cache (head-only fits, no backbone forward)."""
        return [HeadSpec(f"topo_g{g}_s{str(s).replace('.', '')}", "topo", grid=g, rbf_sigma=s)
                for g in _TOPO_GRIDS for s in _TOPO_SIGMAS]

    @staticmethod
    def _grid(feat: torch.Tensor, n_channels: int) -> torch.Tensor:
        """Cached tokens `[B, C·S, d]` -> `[B, C, S, d]` (the composite Head input). S = tokens / C."""
        b, n_tok, d = feat.shape
        return feat.reshape(b, n_channels, n_tok // n_channels, d)

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
                     d=tr_feat.shape[2], pos=pos)

    @staticmethod
    def _load(subjects: list[int], split: str, sample_rate: float):
        eeg, concept, files, meta = things.ThingsEeg2.get_epochs(
            subjects, things.ThingsEpochCfg(split=split, resample=sample_rate), n_jobs=4)
        return eeg, concept, files, meta

    @staticmethod
    @torch.no_grad()
    def _retrieval(head, eval_set: _EvalSet, device: str, n_channels: int) -> dict:
        head.eval()
        feat, concept, bank = eval_set.feat, eval_set.concept, eval_set.bank
        emb = torch.cat([F.normalize(head(FrozenHead._grid(feat[i:i + 4096].float().to(device), n_channels)),
                                     dim=-1).cpu() for i in range(0, len(feat), 4096)])   # composite head, then L2
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
        n_channels = len(cache.pos)
        head = Heads.build(spec, cache.d, cache.pos, n_tok=cache.tr_feat.shape[1]).to(device)
        logit_scale = nn.Parameter(torch.tensor(np.log(1 / 0.07), dtype=torch.float32, device=device))
        opt = torch.optim.AdamW([*head.parameters(), logit_scale], lr=cfg.lr, weight_decay=1e-4)
        rng = np.random.default_rng(cfg.seed)
        best_val, best_state, best_ep = -1.0, None, -1
        arm_start = time.perf_counter()
        for ep in range(cfg.epochs):
            head.train()
            perm = rng.permutation(fit_idx)
            total_loss, n_batches = 0.0, 0
            for i in range(0, len(perm) - 512, 512):
                idx = perm[i:i + 512]
                g = FrozenHead._grid(cache.tr_feat[idx].float().to(device), n_channels)
                tgt = F.normalize(cache.tr_tgt[idx].to(device), dim=-1)
                opt.zero_grad()
                loss = Nice.clip_infonce(F.normalize(head(g), dim=-1), tgt, logit_scale.exp().clamp(max=100))
                loss.backward()
                opt.step()
                total_loss += loss.item()
                n_batches += 1
            val_set = _EvalSet(val_feat, val_lab, val_bank_t)
            val_top1 = FrozenHead._retrieval(head, val_set, device, n_channels)["single_trial"][1]
            if val_top1 > best_val:
                best_val, best_ep = val_top1, ep
                best_state = {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}
            elapsed = time.perf_counter() - arm_start
            eta = elapsed / (ep + 1) * (cfg.epochs - ep - 1)
            Tracking.metrics({"loss": total_loss / max(1, n_batches), "val_top1": val_top1,
                              "sec_per_epoch": elapsed / (ep + 1)}, step=ep)   # per-epoch curve + speed in mlflow
            if ep % 15 == 0 or ep == cfg.epochs - 1:
                logger.info(f"    {spec.name} ep {ep:2d}/{cfg.epochs}  loss {total_loss / max(1, n_batches):.3f}  "
                            f"val-top1 {val_top1*100:.2f}%  {elapsed:.0f}s  ~{eta:.0f}s left")
        head.load_state_dict(best_state)
        test = FrozenHead._retrieval(head, _EvalSet(cache.test_feat, cache.test_concept, cache.test_bank),
                                     device, n_channels)
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
    ap.add_argument("--topo-sweep", action="store_true", help="sweep topo grid×sigma instead of the head zoo")
    ap.add_argument("--only", default=None, help="run only arms whose name contains this substring (e.g. g24_s02)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    cache = FrozenHead._build_cache(args.train, args.test, device, args.backbone)
    fit = FitCfg(epochs=args.epochs, lr=args.lr, seed=args.seed)
    arms = FrozenHead._topo_arms() if args.topo_sweep else _ARMS
    if args.only:
        arms = [a for a in arms if args.only in a.name]
    logger.info(f"\nhead sweep · backbone={args.backbone} train={args.train} test={args.test} · "
                f"{args.epochs}ep lr{args.lr} · chance {1 / (int(cache.test_concept.max()) + 1):.3f}")
    results = []
    for spec in arms:
        params = {"backbone": args.backbone, "arm": spec.name, "pool": spec.pool, "grid": spec.grid,
                  "rbf_sigma": spec.rbf_sigma, "epochs": args.epochs, "lr": args.lr, "seed": args.seed}
        tags = {"task": "perception-frozen-head", "train": args.train, "test": args.test}
        with Tracking.run("mindscape-perception", f"frozen_{args.backbone}_{spec.name}_test{args.test}_s{args.seed}",
                          params=params, tags=tags):
            r = FrozenHead._train_arm(spec, cache, device, fit)
            s, a = r["single_trial"], r["concept_avg"]
            Tracking.metrics({"test_single_top1": s[1], "test_single_top5": s[5],
                              "test_concept_top1": a[1], "test_concept_top5": a[5], "best_val_top1": r["val_top1"]})
        logger.info(f"  {spec.name:9s} (val {r['val_top1']*100:.2f}% ep{r['best_val_epoch']:2d})  "
                    f"single {s[1]*100:.2f}%/{s[5]*100:.2f}%  concept {a[1]*100:.2f}%/{a[5]*100:.2f}%")
        results.append(r)
    if args.out:
        Path(args.out).write_text(json.dumps(
            {"backbone": args.backbone, "train": args.train, "test": args.test, "arms": results}, indent=2))


if __name__ == "__main__":
    main()
