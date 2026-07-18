"""Train + evaluate the NICE EEG->image baseline on THINGS-EEG2.

Pipeline: adapter epochs (our own preprocessing off the raw) -> NICE encoder -> InfoNCE against the viewed
image's CLIP embedding -> zero-shot retrieval on the 200 held-out test concepts. The unbiased number is the
CROSS-SUBJECT single-trial top-k (train subjects != test subject); within-subject and repeat-averaged are
reported alongside as the (inflated) references the field usually quotes.

    # within-subject:
    python -m neuroscan.tasks.visual.train_nice --train 1 --test 1
    # cross-subject (the headline; needs >=2 subjects downloaded):
    python -m neuroscan.tasks.visual.train_nice --train 1 2 3 --test 4

Test concepts are disjoint from training images by dataset design, so retrieval is zero-shot in both regimes;
"cross-subject" additionally holds out the *person*. Chance = 1/200 = 0.5%.

`train()` is the reusable entry (config in, result dict out); `main()` is just argv -> TrainConfig -> train.
A shared multi-method trainer isn't warranted yet — this is the first contrastive/retrieval method (the
braindecode trainer in models/decoders.py is classification); revisit when a second one appears (bd note).
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
from jaxtyping import Float, Int, Shaped
from pydantic import BaseModel
from torch.utils.data import DataLoader, Dataset

from core.data.eeg import things_eeg2 as things
from core.features.eeg.covariance import Covariance
from core.features.eeg.montage import EegMontage
from neuroscan.evaluation.invariants import Invariants
from neuroscan.evaluation.metrics import Metrics
from neuroscan.evaluation.retrieval import Retrieval
from neuroscan.models.encoders import NORMALIZE_CHOICES, EncoderRegistry, EncoderSpec
from neuroscan.models.nice import Nice, SubjectDiscriminator
from neuroscan.tasks.cli import Cli
from neuroscan.tasks.visual import clip_targets
from neuroscan.tasks.visual.sampling import BatchSpec, Sampling
from neuroscan.tracking import Tracking

logger = logging.getLogger(__name__)

_EVAL_BATCH = 512   # batch >=2048 trips a cuDNN illegal-access on this conv shape (Blackwell / cu130)


class _EpochDataset(Dataset):
    """Numpy-backed — converts per sample, so the (large) training array is never copied into one torch
    tensor up front. `indices` optionally views a subset of the arrays without copying them (the fit split
    of a much larger epoch pile). Together these keep a full 9-subject LOSO (~38 GB of epochs) in RAM instead
    of OOM-ing on the doubled copies (torch tensor + boolean-mask slice)."""

    def __init__(self, eeg: Float[np.ndarray, "n ch t"], targets: Float[np.ndarray, "n d"],
                 indices: Int[np.ndarray, "m"] | None = None, subject: Int[np.ndarray, "n"] | None = None):
        self.eeg, self.targets, self.subject = eeg, targets, subject
        self.indices = indices

    def __len__(self) -> int:
        return len(self.indices) if self.indices is not None else len(self.eeg)

    def __getitem__(self, idx: int):
        row = int(self.indices[idx]) if self.indices is not None else idx
        subj = 0 if self.subject is None else int(self.subject[row])
        return torch.from_numpy(self.eeg[row]), torch.from_numpy(self.targets[row]), subj


class TrainConfig(BaseModel):
    """Training hyperparameters (data-independent; the encoder's data-dependent shape is derived at fit)."""
    model: str = "nice"          # encoder name in the models registry (bd bji): "nice" (from-scratch conv
                                 # baseline) | a pretrained foundation backbone — swapped behind one contract
    epochs: int = 40
    batch: int = 512
    lr: float = 3e-4
    weight_decay: float = 1e-4
    backbone_lr_scale: float = 1.0   # foundation fine-tune (bd 07m): backbone LR = lr × this; <1 = discriminative
                                     # LR (lower backbone, avoids forgetting pretrained features). 1.0 = the single-
                                     # LR winning recipe (unchanged default). Only bites when the backbone is trainable.
    warmup_epochs: int = 0           # >0 = linear LR warmup 0->full over N epochs (preserves the per-group ratio),
                                     # then hold. Pairs with backbone_lr_scale for the standard foundation-FT recipe.
    resample: float = 250.0
    normalize: str = "auto"      # input-normalization chain (bd 4aoz), resolved per-encoder by
                                 # EncoderRegistry.normalization(model, normalize). "auto" = the canonical chain
                                 # for the encoder (NICE->MVNN, CBraMod->amplitude scale, EEGPT->z-score); a
                                 # forced value ("zscore"|"mvnn"|"scale") pins one link for the matched A/B.
    seed: int = 0
    patience: int = 8            # early-stop patience on val-top1 (epochs)
    val_fraction: float = 0.1    # share of TRAINING concepts held out for leak-free model selection
    train_frac: float = 1.0      # per-epoch random subset of fit trials (bd pqh: <1 speeds over-sized data)
    sampling: str = "uniform"    # "uniform" | "stratified" (round-robin) | "balanced" (strict, bd 2j2)
    concepts_per_batch: int = 64  # sampling="balanced": concepts × samples = effective batch (64×8=512)
    samples_per_concept: int = 8  # strict equal per-concept representation each batch (bd 2j2)
    geo_lambda: float = 0.0       # >0 = graph-Laplacian spatial-smoothness prior on the NICE spatial conv (bd 1x0):
                                  # penalize weight differences between neighbouring electrodes = montage adjacency
                                  # as a small-data prior. Only bites for encoders exposing geo_penalty (NICE).
    geo_sigma: float = 0.2        # RBF neighbourhood width (unit-disk radii) for the channel graph (matches nm5 topo)
    hard_beta: float = 0.0        # >0 = online hard-negative weighting in the InfoNCE loss (bd fww)
    soft_tau: float = 0.0         # >0 = concept-aware soft InfoNCE targets from CLIP-target sim (bd lbd) —
                                  # same-concept pairs become partial positives, not false negatives
    clip_target: str = "vitb32"   # CLIP target zoo (bd ooi): "vitb32" (512-d, NICE default) | "vitl14" (768-d,
                                  # sharper recon latent). The encoder auto-sizes to the target dim (EncoderSpec).
    mse_weight: float = 0.0       # >0 = add MSE-to-CLIP to InfoNCE (bd ooi): hit the actual embedding, not only
                                  # its direction, so the predicted vector is a usable decoder-conditioning latent
    save_encoder: bool = False    # persist the trained encoder (+ spec) to runs/enc_*.pt for reconstruction (bd 71n)
    val_every: int = 1           # eval the (big) held-out val set every N epochs — strides its per-epoch cost
    amp: bool = True             # bf16 autocast on cuda; False = fp32 (the naive arm of the parity test, bd 9s5)
    recenter: bool = False       # per-subject signal re-centering M^-1/2 X before the encoder (bd dpi) —
                                 # the Stage-1/2 cross-subject transfer template (unsupervised, deployment-safe)
    recenter_shrinkage: float = 0.0   # shrink M toward scaled-I before whitening (bd 36g dig): full whitening
                                      # amplifies noise on ill-conditioned M (cond~1500); >0 aligns signal only
    adversarial: bool = False    # domain-adversarial subject-invariance (bd 36g): GRL + subject discriminator
    adv_lambda: float = 1.0      # gradient-reversal strength (how hard the encoder is pushed to be invariant)
    adv_weight: float = 1.0      # weight of the subject-CE adversary term in the total loss
    adv_lambda_ramp: bool = False  # DANN schedule: ramp lambda 0->adv_lambda over training so the adversary
                                   # doesn't destabilize the still-random early encoder (bd 36g dose-response)


@dataclass
class RetrievalSet:
    """One retrieval eval set: the EEG epochs, their per-trial concept labels, and the candidate embedding
    bank to retrieve against."""
    eeg: np.ndarray
    concept: np.ndarray
    candidates: np.ndarray


@dataclass
class _StepCtx:
    """The mutable training components a single epoch/step needs, bundled so `_run_epoch` takes few args: the
    encoder + its optimizer + learned logit-scale, plus the optional subject-adversary (bd 36g) and geo-prior
    Laplacian (bd 1x0). None for either aux term = that term is off."""
    encoder: object
    optimizer: object
    logit_scale: object
    discriminator: object | None
    geo_lap: object | None


@dataclass
class _FitArrays:
    """The fit-split arrays + their positions, bundled so the per-epoch batch builder takes few args."""
    eeg: np.ndarray
    targets: np.ndarray
    concept: np.ndarray
    fit_indices: np.ndarray
    subject: np.ndarray          # per-row subject index (0-based, for the adversary); zeros when unused


@dataclass
class TrainData:
    """The training inputs for `train_encoder`: epochs + per-trial CLIP targets + concept ids (for the
    leak-free val split) + optional per-row subject id (for the domain-adversary). One bundle so the core
    trainer stays a 3-arg (data, cfg, device) signature across its callers."""
    eeg: np.ndarray
    targets: np.ndarray
    concept: np.ndarray
    subject: np.ndarray | None = None
    positions: np.ndarray | None = None   # [C,2] unit-disk channel positions for the geo-prior (bd 1x0); None = off


class TrainNice:
    """NICE EEG->image trainer — the free helpers folded in as staticmethods (public names kept). `train` is
    the reusable entry (config in, result dict out); `train_encoder` is the dataset-agnostic contrastive core
    reused by the cross-dataset runner."""

    @staticmethod
    def _clip_targets(image_files: Shaped[np.ndarray, "n"], split: str,
                      target: str = "vitb32") -> Float[np.ndarray, "n d"]:
        """CLIP embedding per epoch, looked up by the image the subject viewed (target = CLIP zoo name, bd ooi)."""
        by_file = clip_targets.ClipTargets.embeddings_by_file(split, target)
        return np.stack([by_file[name] for name in image_files]).astype(np.float32)

    @staticmethod
    def test_features(test_subject: int, cfg: TrainConfig):
        """Load + normalize a held-out subject's TEST epochs, encoder-ready (public: bd 71n reconstruct reuse).
        Returns (eeg [n,ch,t] normalized, concept [n]). cbramod's z-score chain is stateless, so no train-fit
        is needed here — for a stateful chain (MVNN) the leak-free fit lives in `train`, not this path."""
        test_eeg, test_concept, _, test_subj, _ = TrainNice._load_split([test_subject], "test", cfg)
        chain = EncoderRegistry.normalization(cfg.model)
        return chain.apply(test_eeg, test_subj), test_concept

    @staticmethod
    def _load_split(subjects: list[int], split: str, cfg: TrainConfig):
        """RAW epochs + metadata for a split — normalization is NOT applied here (the chain is fit on train and
        applied to both splits by `train`, so the eval set is never fit on). Returns (epochs, concept, targets,
        subject, condition=same-image id for MVNN)."""
        epochs, concept, image_files, meta = things.ThingsEeg2.get_epochs(
            subjects, things.ThingsEpochCfg(split=split, resample=cfg.resample))
        subject = meta["subject"].to_numpy()
        condition = np.unique(image_files, return_inverse=True)[1]          # same exemplar image = one condition (MVNN)
        if cfg.recenter:   # opt-in per-subject signal recenter (transductive; off by default, bd dpi killed)
            epochs = Covariance.recenter_signals(epochs, subject, shrinkage=cfg.recenter_shrinkage)  # M^-1/2 X
        targets = TrainNice._clip_targets(image_files, split, cfg.clip_target)
        return epochs, concept, targets, subject.astype(np.int64), condition

    @staticmethod
    @torch.no_grad()
    def evaluate(encoder, data: RetrievalSet, device, batch: int = _EVAL_BATCH) -> dict:
        """Single-trial + concept-averaged retrieval top-1/5 against a per-concept prototype bank."""
        encoder.eval()
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=(device == "cuda")):
            embedded = torch.cat([encoder(torch.tensor(data.eeg[i:i + batch]).to(device)).float().cpu()
                                  for i in range(0, len(data.eeg), batch)])  # [N,512] normalized (back to fp32)
        candidate_bank = torch.tensor(data.candidates)
        labels = torch.tensor(data.concept)
        hits = Nice.retrieval_hits(embedded, candidate_bank, labels)             # per-trial, for the bootstrap CI
        single = {k: float(h.mean()) for k, h in hits.items()}
        single_ci = {k: Metrics.boot_ci(np.mean, h) for k, h in hits.items()}   # honest CI from ONE run (bd 5s3l)
        n_concepts = int(data.concept.max()) + 1
        averaged = torch.stack([torch.nn.functional.normalize(embedded[labels == c].mean(0), dim=-1)
                                for c in range(n_concepts)])
        concept_avg = Nice.retrieval_topk(averaged, candidate_bank, torch.arange(n_concepts))
        continuous = Nice.retrieval_continuous(embedded, candidate_bank, labels)   # angular-error extras (bd 2y7k)
        # rank-aware retrieval quality (MRR / median-rank / PR-AUC) off the [N,C] cosine score matrix — the same
        # numbers cross_dataset_eval reports, now on the primary cross-subject single-trial eval (bd 7tl).
        scores = (embedded @ candidate_bank.t()).numpy()
        rank_metrics = Retrieval.retrieval_metrics(scores, labels.numpy(), ks=(1, 5))
        emb_mse = float(F.mse_loss(embedded, candidate_bank[labels]))   # recon proxy: predicted vs prototype (bd ooi)
        return {"single_trial": single, "single_trial_ci": single_ci, "concept_avg": concept_avg,
                "continuous": continuous, "retrieval_metrics": rank_metrics, "emb_mse": emb_mse,
                "single_trial_hits": {k: h.tolist() for k, h in hits.items()}}   # persisted for paired delta (s1t2)

    @staticmethod
    def _val_split(concept: Int[np.ndarray, "n"], targets: Float[np.ndarray, "n d"], seed: int, fraction: float):
        """Hold out a fraction of TRAINING concepts as a leak-free early-stop signal.

        Returns (train_mask, val_eeg_mask, val_labels, val_prototypes). Selection reads only this val set, so the
        test subject/concepts never influence which checkpoint is kept.
        """
        rng = np.random.default_rng(seed)
        all_concepts = np.unique(concept)
        n_val = max(1, int(len(all_concepts) * fraction))
        val_concepts = sorted(rng.choice(all_concepts, n_val, replace=False).tolist())
        remap = {concept_id: i for i, concept_id in enumerate(val_concepts)}
        prototypes = np.stack([targets[concept == c].mean(0) for c in val_concepts]).astype(np.float32)
        prototypes /= (np.linalg.norm(prototypes, axis=1, keepdims=True) + 1e-8)
        val_mask = np.isin(concept, val_concepts)
        val_labels = np.array([remap[c] for c in concept[val_mask]])
        return ~val_mask, val_mask, val_labels, prototypes

    @staticmethod
    def _concept_neighbor_groups(targets: Float[np.ndarray, "n d"], concept_ids: Int[np.ndarray, "n"],
                                 k: int) -> dict[int, list[int]]:
        """Per-concept CLIP-nearest neighbours (keyed by actual concept id), from the trials' own CLIP targets —
        the model-free hardness prior for clip_hard batching (bd 4ru). Prototype = mean target per concept."""
        concepts = sorted({int(c) for c in concept_ids})
        protos = np.stack([targets[concept_ids == c].mean(0) for c in concepts])
        protos = protos / (np.linalg.norm(protos, axis=1, keepdims=True) + 1e-8)
        raw = Sampling.clip_neighbor_groups(protos @ protos.T, k=k)
        return {concepts[i]: [concepts[j] for j in raw[i]] for i in range(len(concepts))}

    @staticmethod
    def _epoch_steps(data: _FitArrays, neighbor_groups, cfg: TrainConfig, rng, epoch_n: int):
        """This epoch's (eeg, target, subject) batch iterator per `cfg.sampling`: strict balanced / CLIP-hard
        (bd 2j2/4ru), round-robin stratified (bd ewd), or uniform DataLoader. train_frac subsamples (bd kqa/pqh).
        Every source yields the subject index too (dummy zeros unless adversarial) so the loop has one shape."""
        fit = data.fit_indices
        epoch_idx = fit if cfg.train_frac >= 1.0 else rng.choice(fit, epoch_n, replace=False)
        if cfg.sampling in ("balanced", "clip_hard"):
            full_bpe = max(1, len(fit) // (cfg.concepts_per_batch * cfg.samples_per_concept))
            n_bpe = full_bpe if cfg.train_frac >= 1.0 else max(1, int(full_bpe * cfg.train_frac))
            spec = BatchSpec(cfg.concepts_per_batch, cfg.samples_per_concept, n_bpe)
            pos_batches = (Sampling.clip_hard_batches(data.concept[fit], neighbor_groups, spec, rng)
                           if cfg.sampling == "clip_hard" else Sampling.balanced_batches(data.concept[fit], spec, rng))
            # generator, NOT a list: build+free each batch's tensors on demand (a list materializes the whole
            # epoch of copied tensors at once -> OOM on the 240k-trial cross set).
            return ((torch.from_numpy(data.eeg[fit[pos]]), torch.from_numpy(data.targets[fit[pos]]),
                     torch.from_numpy(data.subject[fit[pos]])) for pos in pos_batches)
        if cfg.sampling == "stratified":
            return ((torch.from_numpy(data.eeg[epoch_idx[pos]]), torch.from_numpy(data.targets[epoch_idx[pos]]),
                     torch.from_numpy(data.subject[epoch_idx[pos]]))
                    for pos in Sampling.stratified_batches(data.concept[epoch_idx], cfg.batch, rng) if len(pos) > 1)
        return DataLoader(_EpochDataset(data.eeg, data.targets, epoch_idx, data.subject),
                          batch_size=cfg.batch, shuffle=True, drop_last=True)

    @staticmethod
    def _enable_fast_matmul(device: str) -> None:
        """TF32 for the residual fp32 matmuls (`high` precision). Measured −22% step time (bd 62ak: the win is in
        backward, 25.8→17.3 ms) — parity-safe since training already runs bf16 autocast, and TF32's 10-bit mantissa
        is MORE precise than the bf16 already in use. cudnn.benchmark is deliberately NOT set: measured neutral/worse
        for the small NICE convs, and variable end-of-epoch batch shapes would re-trigger its autotune."""
        if device != "cuda":
            return
        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    @staticmethod
    def _build_optim(spec: EncoderSpec, n_subjects: int, cfg: TrainConfig, device: str):
        """Encoder (by `cfg.model`, from the registry) + logit-scale + (optional) subject discriminator under one
        AdamW, plus a linear LR-warmup scheduler (opt-in, bd 07m: scales every group equally so the discriminative
        backbone/head ratio survives the ramp; warmup_epochs=0 -> a no-op constant-1.0 schedule)."""
        TrainNice._enable_fast_matmul(device)               # every training path builds an optimizer here (bd 62ak)
        encoder = EncoderRegistry.build_encoder(cfg.model, spec).to(device)
        logit_scale = torch.nn.Parameter(torch.tensor(np.log(1 / 0.07), dtype=torch.float32, device=device))
        # per-encoder optimizer groups if the encoder defines them (foundation: discriminative backbone/head LR),
        # else one group at cfg.lr (NICE — unchanged).
        make_groups = getattr(encoder, "param_groups", None)
        groups = (make_groups(cfg.lr, cfg.backbone_lr_scale) if make_groups
                  else [{"params": [*encoder.parameters()], "lr": cfg.lr}])
        groups.append({"params": [logit_scale], "lr": cfg.lr})
        discriminator = None
        if cfg.adversarial:
            discriminator = SubjectDiscriminator(spec.embed_dim, n_subjects).to(device)
            groups.append({"params": [*discriminator.parameters()], "lr": cfg.lr})
            logger.info(f"domain-adversarial: {n_subjects} subjects, lambda {cfg.adv_lambda}, weight {cfg.adv_weight}")
        optimizer = torch.optim.AdamW(groups, lr=cfg.lr, weight_decay=cfg.weight_decay)
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lambda ep: min(1.0, (ep + 1) / cfg.warmup_epochs) if cfg.warmup_epochs > 0 else 1.0)
        return encoder, logit_scale, discriminator, optimizer, scheduler

    @staticmethod
    def _geo_laplacian(encoder, data: TrainData, cfg: TrainConfig, device: str):
        """Channel graph-Laplacian tensor for the spatial-smoothness prior (bd 1x0), or None when the prior is off
        (geo_lambda=0), no montage positions travelled with the data, or the encoder has no spatial conv to
        smooth (foundation backbones). Built once per fit; the penalty rides the training loop."""
        if not (cfg.geo_lambda > 0 and data.positions is not None and hasattr(encoder, "geo_penalty")):
            return None
        logger.info(f"geo-prior: graph-Laplacian smoothness lambda={cfg.geo_lambda} sigma={cfg.geo_sigma}")
        return torch.tensor(EegMontage.channel_laplacian(data.positions, cfg.geo_sigma), device=device)

    @staticmethod
    def _run_epoch(tr: _StepCtx, steps, lam: float, cfg: TrainConfig, device: str) -> tuple[float, int]:
        """One training epoch over `steps`; returns (summed batch loss, n_batches). Keeps the per-step loss
        assembly in one place — InfoNCE (bd, CLIP loss) + optional subject-adversary term (GRL, bd 36g) +
        optional graph-Laplacian spatial-smoothness prior (bd 1x0) — with the mutable components carried by `tr`."""
        tr.encoder.train()
        total_loss, n_batches = 0.0, 0
        for eeg_batch, target_batch, subj_batch in steps:
            eeg_batch = eeg_batch.to(device)
            target_batch = torch.nn.functional.normalize(target_batch.to(device), dim=-1)
            tr.optimizer.zero_grad()
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=(device == "cuda" and cfg.amp)):
                z = tr.encoder(eeg_batch)
                loss = Nice.clip_infonce(z, target_batch, tr.logit_scale.exp().clamp(max=100),
                                    hard_beta=cfg.hard_beta, soft_tau=cfg.soft_tau)
                if cfg.mse_weight > 0:                         # hit the CLIP embedding, not only its direction (bd ooi)
                    loss = loss + cfg.mse_weight * F.mse_loss(z, target_batch)   # both L2-normed: MSE on the sphere
                if tr.discriminator is not None:               # push encoder to be subject-invariant (GRL, bd 36g)
                    loss = loss + cfg.adv_weight * F.cross_entropy(tr.discriminator(z, lam), subj_batch.to(device))
                if tr.geo_lap is not None:                     # montage-adjacency smoothness prior (bd 1x0)
                    loss = loss + cfg.geo_lambda * tr.encoder.geo_penalty(tr.geo_lap)
            loss.backward()                                    # bf16 autocast needs no GradScaler (unlike fp16)
            tr.optimizer.step()
            total_loss += loss.item()
            n_batches += 1
        return total_loss, n_batches

    @staticmethod
    def _dann_lambda(progress: float, max_lambda: float, gamma: float = 10.0) -> float:
        """DANN gradient-reversal schedule (Ganin & Lempitsky 2015): ramp 0 -> `max_lambda` as
        `2/(1+exp(-gamma·p)) - 1` over training progress `p ∈ [0,1]`, so the adversary engages gradually once the
        encoder has structure instead of fighting the random early features (constant high lambda flattened the
        dose-response — w1.0 fell back to baseline)."""
        return max_lambda * (2.0 / (1.0 + np.exp(-gamma * progress)) - 1.0)

    @staticmethod
    def train_encoder(data: TrainData, cfg: TrainConfig, device: str):
        """Contrastive-fit a NICE encoder with leak-free early stopping on held-out TRAIN concepts, returning the
        best-val checkpoint. Dataset-agnostic core shared by the within/cross-subject EEG2 runs (train()) and the
        cross-dataset EEG1 run (cross_dataset_eval). `data.subject` (per-row 0-based id) drives the optional
        domain-adversarial head (bd 36g); None -> a single dummy domain, adversary off."""
        train_eeg, train_targets, train_concept = data.eeg, data.targets, data.concept
        fit_mask, val_mask, val_labels, val_bank = TrainNice._val_split(
            train_concept, train_targets, cfg.seed, cfg.val_fraction)
        fit_indices = np.where(fit_mask)[0]            # view, not a copy — fit is the bulk of the epoch pile
        val_eeg = train_eeg[val_mask]                  # small (one split fraction), copy is fine
        subject = np.zeros(len(train_eeg), dtype=np.int64) if data.subject is None else data.subject.astype(np.int64)
        rng = np.random.default_rng(cfg.seed)
        epoch_n = (len(fit_indices) if cfg.train_frac >= 1.0
                   else min(len(fit_indices), max(cfg.batch, int(len(fit_indices) * cfg.train_frac))))
        logger.info(f"early-stop val: {len(val_bank)} held-out train concepts, {int(val_mask.sum())} epochs; "
              f"fit {len(fit_indices)} trials, {epoch_n}/epoch (train_frac {cfg.train_frac})")

        spec = EncoderSpec(n_channels=train_eeg.shape[1], n_times=train_eeg.shape[2], embed_dim=train_targets.shape[1])
        encoder, logit_scale, discriminator, optimizer, scheduler = TrainNice._build_optim(
            spec, int(subject.max()) + 1, cfg, device)

        neighbor_groups = None
        if cfg.sampling == "clip_hard":                        # CLIP-prior hard-negative neighbours (bd 4ru)
            neighbor_groups = TrainNice._concept_neighbor_groups(
                train_targets[fit_indices], train_concept[fit_indices], cfg.concepts_per_batch)

        tr = _StepCtx(encoder, optimizer, logit_scale, discriminator,
                      TrainNice._geo_laplacian(encoder, data, cfg, device))   # spatial-smoothness prior (bd 1x0)

        fit_data = _FitArrays(train_eeg, train_targets, train_concept, fit_indices, subject)
        best_val, best_state, best_epoch, since_improved, run_start = -1.0, None, -1, 0, time.perf_counter()
        for epoch in range(cfg.epochs):
            lam = (TrainNice._dann_lambda(epoch / max(1, cfg.epochs - 1), cfg.adv_lambda)
                   if cfg.adv_lambda_ramp else cfg.adv_lambda)
            steps = TrainNice._epoch_steps(fit_data, neighbor_groups, cfg, rng, epoch_n)
            epoch_start = time.perf_counter()
            total_loss, n_batches = TrainNice._run_epoch(tr, steps, lam, cfg, device)
            scheduler.step()                               # per-epoch LR warmup ramp (no-op when warmup_epochs=0)

            train_s = time.perf_counter() - epoch_start
            if epoch % cfg.val_every == 0 or epoch == cfg.epochs - 1:      # stride the big val eval (bd)
                val_eval = TrainNice.evaluate(encoder, RetrievalSet(val_eeg, val_labels, val_bank), device)
                val_top1 = val_eval["single_trial"][1]
                if val_top1 > best_val:
                    best_val, best_epoch, since_improved = val_top1, epoch, 0
                    best_state = {k: v.detach().cpu().clone() for k, v in encoder.state_dict().items()}
                else:
                    since_improved += 1
                Tracking.metrics({"loss": total_loss / max(1, n_batches), "val_top1": val_top1,
                                  "val_cos_to_true": val_eval["continuous"]["cos_to_true_mean"],
                                  "sec_per_epoch": train_s}, step=epoch)   # per-epoch curve + speed in mlflow
                if epoch % 5 == 0 or epoch == cfg.epochs - 1:
                    logger.info(f"ep {epoch:3d}  loss {total_loss / max(1, n_batches):.3f}  "
                          f"val-top1 {val_top1*100:.2f}%  cos {val_eval['continuous']['cos_to_true_mean']:.3f}"
                          f"  {train_s:.1f}s ({n_batches / train_s:.0f} batch/s)  "
                          f"~{(time.perf_counter() - run_start) / (epoch + 1) * (cfg.epochs - epoch - 1):.0f}s left")
                if since_improved >= cfg.patience:
                    logger.info(f"early stop at ep {epoch} (best val = ep {best_epoch})")
                    break

        encoder.load_state_dict(best_state)                            # keep the best-VAL checkpoint
        return encoder, {"best_val_epoch": best_epoch, "val_top1": best_val, "epochs_run": epoch + 1}

    @staticmethod
    def train(train_subjects: list[int], test_subject: int, cfg: TrainConfig) -> dict:
        """Fit the encoder (contrastive) on EEG2 with leak-free early stopping; return the test result at the
        best-val checkpoint. Reusable entry point — `main()` only builds the config and prints/saves this dict."""
        torch.manual_seed(cfg.seed)
        np.random.seed(cfg.seed)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        regime = "within" if train_subjects == [test_subject] else "cross"
        logger.info(f"regime={regime}  train={train_subjects}  test={test_subject}  device={device}")

        train_eeg, train_concept, train_targets, train_subj, train_cond = TrainNice._load_split(
            train_subjects, "training", cfg)
        test_eeg, test_concept, _, test_subj, _ = TrainNice._load_split([test_subject], "test", cfg)
        # Per-subject normalization is fit on CALIBRATION epochs, leak-free: every subject's OWN training-image
        # trials — for the held-out test subject, its training trials, DISJOINT from the scored test-image
        # trials it is evaluated on. The encoder still trains only on `train_eeg`; this bundle only fits the
        # whitener. Stateless chains (z-score/scale) ignore the fit data, so one bundle drives every chain.
        fit_eeg, fit_subj, fit_cond = train_eeg, train_subj, train_cond
        if test_subject not in train_subjects:
            calib_eeg, _, _, calib_subj, calib_cond = TrainNice._load_split([test_subject], "training", cfg)
            fit_eeg = np.concatenate([train_eeg, calib_eeg])
            fit_subj = np.concatenate([train_subj, calib_subj])
            fit_cond = np.concatenate([train_cond, calib_cond])
        chain = EncoderRegistry.normalization(cfg.model, cfg.normalize, fit_subj, fit_cond)
        chain.fit(fit_eeg)
        train_eeg = chain.apply(train_eeg, train_subj)   # each subject whitened by its own calibration whitener
        test_eeg = chain.apply(test_eeg, test_subj)      # test subject's whitener never saw these scored trials
        subj_map = {s: i for i, s in enumerate(sorted(set(train_subjects)))}
        subj_idx = np.array([subj_map[int(s)] for s in train_subj], dtype=np.int64) if cfg.adversarial else None
        test_bank = clip_targets.ClipTargets.concept_prototypes("test", cfg.clip_target)
        logger.info(f"train {train_eeg.shape} -> CLIP {train_targets.shape} | "
              f"test {test_eeg.shape} ({int(test_concept.max())+1} concepts)")

        params = {"model": cfg.model, "regime": regime, "train": train_subjects, "test": test_subject,
                  "epochs": cfg.epochs, "lr": cfg.lr, "resample": cfg.resample, "seed": cfg.seed}
        tags = {"task": "perception-nice", "train": train_subjects, "test": test_subject}
        with Tracking.run("mindscape-perception", f"{cfg.model}_{regime}_test{test_subject}_s{cfg.seed}",
                          params=params, tags=tags):
            positions = EegMontage.eeg_positions(things.ThingsEeg2.channels()) if cfg.geo_lambda > 0 else None
            encoder, stats = TrainNice.train_encoder(
                TrainData(train_eeg, train_targets, train_concept, subj_idx, positions), cfg, device)
            test = TrainNice.evaluate(encoder, RetrievalSet(test_eeg, test_concept, test_bank), device)
            single, concept = test["single_trial"], test["concept_avg"]
            Tracking.metrics({"test_single_top1": single[1], "test_single_top5": single[5],
                              "test_concept_top1": concept[1], "test_concept_top5": concept[5],
                              "best_val_top1": stats["val_top1"],
                              **{f"test_{k}": v for k, v in test["continuous"].items()},   # angular error (bd 2y7k)
                              **{f"test_{k}": v for k, v in test["retrieval_metrics"].items()}})   # rank-aware (bd 7tl)
            rm = test["retrieval_metrics"]
            logger.info(f"test5 single-top1 {single[1]*100:.2f}%  MRR {rm['mrr']:.3f}  "
                        f"median-rank {rm['median_rank']:.0f}  PR-AUC {rm['pr_auc']:.3f}")
        if cfg.save_encoder:                                   # decodable checkpoint for reconstruction (bd 71n)
            ckpt = Path(f"runs/enc_{cfg.model}_test{test_subject}_{cfg.clip_target}.pt")
            torch.save({"state_dict": encoder.state_dict(), "model": cfg.model,
                        "n_channels": train_eeg.shape[1], "n_times": train_eeg.shape[2],
                        "embed_dim": train_targets.shape[1], "clip_target": cfg.clip_target,
                        "resample": cfg.resample}, ckpt)
            logger.info(f"saved encoder -> {ckpt}")
        result = {"regime": regime, "train": train_subjects, "test": test_subject, **stats,
                  "chance_top1": 1 / (int(test_concept.max()) + 1), **test}
        Invariants.check(result)                               # fail loud on a silently-inconsistent number
        return result


def main():
    Cli.setup_logging()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--train", type=int, nargs="+", default=[1], help="training subject id(s)")
    ap.add_argument("--test", type=int, default=1, help="held-out test subject id")
    ap.add_argument("--config", default=None,
                    help="JSON file of TrainConfig fields (the recipe home; e.g. the balanced+clip_hard "
                         "perception config). Explicit flags below override it.")
    ap.add_argument("--model", default=None, help="encoder name (registry): 'nice' | a foundation backbone")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch", type=int, default=None, help="<=1024 (cuDNN cap on this shape)")
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--resample", type=float, default=None)
    ap.add_argument("--normalize", default=None, choices=NORMALIZE_CHOICES,
                    help="input-normalization chain (bd 4aoz): auto = the per-encoder canonical chain")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--geo-lambda", dest="geo_lambda", type=float, default=None,
                    help="graph-Laplacian spatial-smoothness prior weight (bd 1x0); 0 = off")
    ap.add_argument("--geo-sigma", dest="geo_sigma", type=float, default=None, help="geo-prior RBF width (unit-disk)")
    ap.add_argument("--patience", type=int, default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    base = json.loads(Path(args.config).read_text()) if args.config else {}
    overrides = {k: v for k, v in vars(args).items()
                 if k in TrainConfig.model_fields and v is not None}
    cfg = TrainConfig(**{**base, **overrides})
    result = TrainNice.train(args.train, args.test, cfg)
    logged = {k: v for k, v in result.items() if k != "single_trial_hits"}   # per-trial vector -> --out only
    logger.info(json.dumps(logged, indent=2))
    if args.out:
        with Path(args.out).open("w") as f:
            json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()
