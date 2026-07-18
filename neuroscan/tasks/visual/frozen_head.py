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
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from jaxtyping import Float, Int, Shaped
from torch import Tensor, nn

from core.config import Config
from core.data.eeg import things_eeg2 as things
from core.features.eeg.montage import EegMontage
from neuroscan.models.composite import HeadContext, Heads, HeadSpec
from neuroscan.models.encoders import NORMALIZE_CHOICES, EncoderRegistry
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


@dataclass
class _Extract:
    """How to compute features on a cache miss: the loaded backbone (None if the cache is warm), the device,
    and the normalization override that picks the input chain (bd 4aoz)."""
    loaded: LoadedBackbone | None
    device: str
    normalize: str


_GEOMETRY_POOLS = {"pos_attn", "topo", "gcn"}   # fold the ELECTRODE axis — need grid C == #electrodes
_TEMPORAL_POOLS = {"temporal"}                  # fold the TIME axis — need a time-first grid (S>1 backbone)

_ARMS = [
    HeadSpec("mean_lin", "mean", hidden=0),        # frozen linear-probe floor (~0.6%, chance)
    HeadSpec("flat_mlp", "flat", hidden=1024),     # best geometry-blind (unordered bag of tokens, ~1.2%)
    HeadSpec("pos_attn", "pos_attn", hidden=512),  # geometry: electrode-position embedding + self-attention
    HeadSpec("topo_cnn", "topo", hidden=512, grid=24, rbf_sigma=0.2),  # scalp-grid RBF + 2D CNN (Bashivan);
    # grid24/sigma0.2 = 1.78±0.03 (3-seed) beats the grid16 default 1.68 by +0.10 (bd m69x.2) — the frozen ceiling

    HeadSpec("gcn", "gcn", hidden=512),            # geometry: electrode-adjacency graph message passing (2-layer GCN)
    HeadSpec("temporal_cnn", "temporal", hidden=512),  # temporal: 1D conv along time patches (S>1 backbones only)
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

    @classmethod
    def _topo_arms(cls) -> list[HeadSpec]:
        """The topo grid×sigma mini-sweep arms (bd m69x.2) — confirm topo_cnn's ~1.68 isn't left on the table
        by the interpolation resolution. Rides the disk cache (head-only fits, no backbone forward)."""
        return [HeadSpec(f"topo_g{g}_s{str(s).replace('.', '')}", "topo", grid=g, rbf_sigma=s)
                for g in _TOPO_GRIDS for s in _TOPO_SIGMAS]

    @classmethod
    @torch.no_grad()
    def _features(cls, module: nn.Module, eeg: Float[np.ndarray, "n ch t"], device: str, backbone: str,
                  normalize: str) -> Float[Tensor, "n ch s d"]:
        """Frozen token grid for every epoch, `[N, C, S, d]` float16 on CPU (the one-time cost). Raw epochs are
        first run through the backbone's normalization chain (bd 4aoz — `normalize='auto'` picks CBraMod's
        amplitude scale / EEGPT's z-score; a forced value drives the A/B), then patched by the Backbone: CBraMod
        gives S=1, a finer-patching backbone (EEGPT) gives S>1."""
        chain = EncoderRegistry.normalization(backbone, normalize)   # cbramod/eegpt -> stateless z-score (or scale)
        eeg = chain.fit(eeg).apply(eeg)
        out = []
        for i in range(0, len(eeg), _FEAT_BATCH):
            x = torch.tensor(eeg[i:i + _FEAT_BATCH]).to(device)
            out.append(module(x).half().cpu())                                     # [B, C, S, d]
        return torch.cat(out)

    @classmethod
    def _clip_targets(cls, image_files: Shaped[np.ndarray, "n"], split: str) -> Float[np.ndarray, "n d"]:
        by_file = clip_targets.ClipTargets.embeddings_by_file(split)
        return np.stack([by_file[name] for name in image_files]).astype(np.float32)

    @classmethod
    def _val_concepts(cls, concept: Int[np.ndarray, "n"], targets: Float[np.ndarray, "n d"], seed: int,
                      fraction: float):
        """Hold out a fraction of TRAIN concepts as a leak-free early-stop bank (mirrors train_nice._val_split)."""
        rng = np.random.default_rng(seed)
        concepts = np.unique(concept)
        val = sorted(rng.choice(concepts, max(1, int(len(concepts) * fraction)), replace=False).tolist())
        remap = {c: i for i, c in enumerate(val)}
        bank = np.stack([targets[concept == c].mean(0) for c in val]).astype(np.float32)
        bank /= (np.linalg.norm(bank, axis=1, keepdims=True) + 1e-8)
        mask = np.isin(concept, val)
        return ~mask, mask, np.array([remap[c] for c in concept[mask]]), bank

    @classmethod
    def _cache_path(cls, backbone: str, subjects: list[int], split: str, normalize: str) -> Path:
        """One home per (backbone, normalization, split, subject-set) frozen-feature blob, out-of-repo under
        <data>/cache. `normalize` is in the key so the scale-vs-zscore A/B (bd 7mi4) does not collide; the __n2
        suffix retires pre-4aoz blobs (normalization moved to the core.normalization chain)."""
        subj = "-".join(str(s) for s in sorted(subjects))
        return Config.data_root("cache") / "frozen_features" / f"{backbone}__{normalize}__{split}__{subj}__n2.pt"

    @classmethod
    def _loaded(cls, backbone: str, device: str) -> LoadedBackbone:
        """Load + freeze a backbone onto `device` (only called on a cache miss — a full sweep off the cache
        never touches the backbone). Passes the montage channel names for the backbone's adapter (EEGPT)."""
        lb = Foundation.load_backbone(backbone, channel_names=things.ThingsEeg2.channels())
        module = lb.module.to(device).eval()
        for p in module.parameters():
            p.requires_grad = False
        return LoadedBackbone(module, lb.patch_points, lb.d_model, lb.sample_rate, lb.name)

    @classmethod
    def _split_features(cls, path: Path, subjects: list[int], split: str, extract: _Extract
                        ) -> tuple[torch.Tensor, np.ndarray, np.ndarray]:
        """Frozen features + (concept, files) for one split — from the disk cache if present (skips the eeg
        load AND the backbone forward), else compute through `extract` and persist. Cache holds the
        backbone-independent concept/files too, so a reused sweep needs neither mne nor the GPU."""
        if path.exists():
            blob = torch.load(path, weights_only=False)   # our own cache (feat tensor + numpy concept/files)
            logger.info(f"cache hit {path.name}: {tuple(blob['feat'].shape)}")
            return blob["feat"], blob["concept"], blob["files"]
        loaded = extract.loaded
        if loaded is None:
            raise RuntimeError("backbone must be loaded when cache is missing")
        eeg, concept, files, _ = cls._load(subjects, split, loaded.sample_rate)
        feat = cls._features(loaded.module, eeg, extract.device, loaded.name, extract.normalize)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"feat": feat, "concept": concept, "files": files}, path)
        logger.info(f"cached {path.name}: {tuple(feat.shape)} (float16)")
        return feat, concept, files

    @classmethod
    def _build_cache(cls, train_subjects: list[int], test_subject: int, device: str,
                     backbone: str = "cbramod", normalize: str = "auto") -> Cache:
        tr_path = cls._cache_path(backbone, train_subjects, "training", normalize)
        te_path = cls._cache_path(backbone, [test_subject], "test", normalize)
        loaded = None if (tr_path.exists() and te_path.exists()) else cls._loaded(backbone, device)
        extract = _Extract(loaded, device, normalize)
        tr_feat, tr_concept, tr_files = cls._split_features(tr_path, train_subjects, "training", extract)
        test_feat, te_concept, _ = cls._split_features(te_path, [test_subject], "test", extract)
        tr_tgt = torch.tensor(cls._clip_targets(tr_files, "training"))
        test_bank = torch.tensor(clip_targets.ClipTargets.concept_prototypes("test"))
        pos = EegMontage.eeg_positions(things.ThingsEeg2.channels())   # [C, 2], same channel order as the features
        return Cache(tr_feat, tr_tgt, tr_concept, test_feat, te_concept, test_bank,
                     d=tr_feat.shape[-1], pos=pos)

    @classmethod
    def _load(cls, subjects: list[int], split: str, sample_rate: float):
        eeg, concept, files, meta = things.ThingsEeg2.get_epochs(
            subjects, things.ThingsEpochCfg(split=split, resample=sample_rate), n_jobs=4)
        return eeg, concept, files, meta

    @classmethod
    @torch.no_grad()
    def _retrieval(cls, head: nn.Module, eval_set: _EvalSet, device: str) -> dict[str, Any]:
        head.eval()
        feat, concept, bank = eval_set.feat, eval_set.concept, eval_set.bank
        emb = torch.cat([F.normalize(head(feat[i:i + 4096].float().to(device)), dim=-1).cpu()
                         for i in range(0, len(feat), 4096)])   # [B,C,S,d] straight into the composite head, then L2
        labels = torch.tensor(concept)
        single = Nice.retrieval_topk(emb, bank, labels)
        n = int(concept.max()) + 1
        averaged = torch.stack([F.normalize(emb[labels == c].mean(0), dim=-1) for c in range(n)])
        return {"single_trial": single, "concept_avg": Nice.retrieval_topk(averaged, bank, torch.arange(n)),
                "continuous": Nice.retrieval_continuous(emb, bank, labels)}   # angular-error extras (bd 2y7k)

    @classmethod
    def _train_arm(cls, spec: HeadSpec, cache: Cache, device: str, cfg: FitCfg) -> dict[str, Any]:
        torch.manual_seed(cfg.seed)
        fit_mask, val_mask, val_lab, val_bank = cls._val_concepts(
            cache.tr_concept, cache.tr_tgt.numpy(), cfg.seed, 0.1)
        fit_idx = np.where(fit_mask)[0]
        val_feat, val_bank_t = cache.tr_feat[val_mask], torch.tensor(val_bank)
        n_tok = cache.tr_feat.shape[1] * cache.tr_feat.shape[2]   # C·S tokens (for the flat head's MLP in-dim)
        head = Heads.build(spec, HeadContext(cache.d, cache.pos), n_tok=n_tok).to(device)
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
                g = cache.tr_feat[idx].float().to(device)         # [B, C, S, d] straight into the head
                tgt = F.normalize(cache.tr_tgt[idx].to(device), dim=-1)
                opt.zero_grad()
                loss = Nice.clip_infonce(F.normalize(head(g), dim=-1), tgt, logit_scale.exp().clamp(max=100))
                loss.backward()
                opt.step()
                total_loss += loss.item()
                n_batches += 1
            val_set = _EvalSet(val_feat, val_lab, val_bank_t)
            val_top1 = cls._retrieval(head, val_set, device)["single_trial"][1]
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
        if best_state is not None:
            head.load_state_dict(best_state)
        test = cls._retrieval(head, _EvalSet(cache.test_feat, cache.test_concept, cache.test_bank), device)
        return {"arm": spec.name, "best_val_epoch": best_ep, "val_top1": best_val, **test}

    @classmethod
    def main(cls):
        logging.basicConfig(level=logging.INFO, format="%(message)s")
        for lib in ("mne", "moabb", "braindecode"):
            logging.getLogger(lib).setLevel(logging.WARNING)
        ap = argparse.ArgumentParser(description=__doc__)
        ap.add_argument("--train", type=int, nargs="+", default=[1, 2, 3, 4])
        ap.add_argument("--test", type=int, default=5)
        ap.add_argument("--backbone", default="cbramod", help="frozen backbone (registry name)")
        ap.add_argument("--normalize", default="auto", choices=NORMALIZE_CHOICES,
                        help="input-normalization chain (bd 4aoz): auto = the backbone's canonical chain")
        ap.add_argument("--epochs", type=int, default=60)
        ap.add_argument("--lr", type=float, default=1e-3)
        ap.add_argument("--seed", type=int, default=0)
        ap.add_argument("--topo-sweep", action="store_true", help="sweep topo grid×sigma instead of the head zoo")
        ap.add_argument("--only", default=None, help="run only arms whose name contains this substring (e.g. g24_s02)")
        ap.add_argument("--out", default=None)
        args = ap.parse_args()

        device = "cuda" if torch.cuda.is_available() else "cpu"
        cache = cls._build_cache(args.train, args.test, device, args.backbone, args.normalize)
        fit = FitCfg(epochs=args.epochs, lr=args.lr, seed=args.seed)
        arms = cls._topo_arms() if args.topo_sweep else _ARMS
        if args.only:
            arms = [a for a in arms if args.only in a.name]
        spatial_grid = cache.tr_feat.shape[1] == len(cache.pos)   # grid axis IS electrodes (CBraMod) vs time (EEGPT)
        drop = _TEMPORAL_POOLS if spatial_grid else _GEOMETRY_POOLS
        skipped = [a.name for a in arms if a.pool in drop]
        arms = [a for a in arms if a.pool not in drop]
        if skipped:
            axis = "electrodes" if spatial_grid else f"time-patches (C={cache.tr_feat.shape[1]})"
            logger.info(f"grid axis = {axis}; skipping inapplicable heads: {skipped}")
        logger.info(f"\nhead sweep · backbone={args.backbone} train={args.train} test={args.test} · "
                    f"{args.epochs}ep lr{args.lr} · chance {1 / (int(cache.test_concept.max()) + 1):.3f}")
        results = []
        for spec in arms:
            params = {"backbone": args.backbone, "arm": spec.name, "pool": spec.pool, "grid": spec.grid,
                      "rbf_sigma": spec.rbf_sigma, "epochs": args.epochs, "lr": args.lr, "seed": args.seed}
            tags = {"task": "perception-frozen-head", "train": args.train, "test": args.test}
            run_name = f"frozen_{args.backbone}_{spec.name}_test{args.test}_s{args.seed}"
            with Tracking.run("mindscape-perception", run_name,
                              params=params, tags=tags):
                r = cls._train_arm(spec, cache, device, fit)
                single, concept, cont = r["single_trial"], r["concept_avg"], r["continuous"]
                Tracking.metrics({"test_single_top1": single[1], "test_single_top5": single[5],
                                  "test_concept_top1": concept[1], "test_concept_top5": concept[5],
                                  "best_val_top1": r["val_top1"],
                                  **{f"test_{k}": v for k, v in cont.items()}})   # angular error (bd 2y7k)
            logger.info(f"  {spec.name:9s} (val {r['val_top1']*100:.2f}% ep{r['best_val_epoch']:2d})  "
                        f"single {single[1]*100:.2f}%/{single[5]*100:.2f}%  "
                        f"concept {concept[1]*100:.2f}%/{concept[5]*100:.2f}%  "
                        f"cos {cont['cos_to_true_mean']:.3f} margin {cont['margin_mean']:.3f}")
            results.append(r)
        if args.out:
            Path(args.out).write_text(json.dumps(
                {"backbone": args.backbone, "train": args.train, "test": args.test, "arms": results}, indent=2))


if __name__ == "__main__":
    FrozenHead.main()
