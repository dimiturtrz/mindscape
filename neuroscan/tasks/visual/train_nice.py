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
from pydantic import BaseModel
from torch.utils.data import DataLoader, Dataset

from core.data.eeg import things_eeg2 as things
from core.features.eeg.covariance import Covariance
from neuroscan.evaluation.metrics import Metrics
from neuroscan.models.encoders import EncoderRegistry, EncoderSpec
from neuroscan.models.nice import Nice, SubjectDiscriminator
from neuroscan.tasks.cli import Cli
from neuroscan.tasks.visual import clip_targets
from neuroscan.tasks.visual.sampling import BatchSpec, Sampling

logger = logging.getLogger(__name__)

_EVAL_BATCH = 512   # batch >=2048 trips a cuDNN illegal-access on this conv shape (Blackwell / cu130)


class _EpochDataset(Dataset):
    """Numpy-backed — converts per sample, so the (large) training array is never copied into one torch
    tensor up front. `indices` optionally views a subset of the arrays without copying them (the fit split
    of a much larger epoch pile). Together these keep a full 9-subject LOSO (~38 GB of epochs) in RAM instead
    of OOM-ing on the doubled copies (torch tensor + boolean-mask slice)."""

    def __init__(self, eeg: np.ndarray, targets: np.ndarray, indices: np.ndarray | None = None,
                 subject: np.ndarray | None = None):
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
    resample: float = 250.0
    seed: int = 0
    patience: int = 8            # early-stop patience on val-top1 (epochs)
    val_fraction: float = 0.1    # share of TRAINING concepts held out for leak-free model selection
    train_frac: float = 1.0      # per-epoch random subset of fit trials (bd pqh: <1 speeds over-sized data)
    sampling: str = "uniform"    # "uniform" | "stratified" (round-robin) | "balanced" (strict, bd 2j2)
    concepts_per_batch: int = 64  # sampling="balanced": concepts × samples = effective batch (64×8=512)
    samples_per_concept: int = 8  # strict equal per-concept representation each batch (bd 2j2)
    hard_beta: float = 0.0        # >0 = online hard-negative weighting in the InfoNCE loss (bd fww)
    soft_tau: float = 0.0         # >0 = concept-aware soft InfoNCE targets from CLIP-target sim (bd lbd) —
                                  # same-concept pairs become partial positives, not false negatives
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


class TrainNice:
    """NICE EEG->image trainer — the free helpers folded in as staticmethods (public names kept). `train` is
    the reusable entry (config in, result dict out); `train_encoder` is the dataset-agnostic contrastive core
    reused by the cross-dataset runner."""

    @staticmethod
    def _clip_targets(image_files: np.ndarray, split: str) -> np.ndarray:
        """CLIP embedding per epoch, looked up by the image the subject viewed."""
        by_file = clip_targets.ClipTargets.embeddings_by_file(split)
        return np.stack([by_file[name] for name in image_files]).astype(np.float32)

    @staticmethod
    def _load_split(subjects: list[int], split: str, resample: float, *, recenter: bool = False,
                    shrinkage: float = 0.0):
        epochs, concept, image_files, meta = things.ThingsEeg2.get_epochs(
            subjects, things.ThingsEpochCfg(split=split, resample=resample))
        if recenter:
            epochs = Covariance.recenter_signals(epochs, meta["subject"].to_numpy(), shrinkage=shrinkage)  # M^-1/2 X
        return epochs, concept, TrainNice._clip_targets(image_files, split), meta["subject"].to_numpy().astype(np.int64)

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
        return {"single_trial": single, "single_trial_ci": single_ci, "concept_avg": concept_avg,
                "single_trial_hits": {k: h.tolist() for k, h in hits.items()}}   # persisted for paired delta (s1t2)

    @staticmethod
    def _val_split(concept: np.ndarray, targets: np.ndarray, seed: int, fraction: float):
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
    def _concept_neighbor_groups(targets: np.ndarray, concept_ids: np.ndarray, k: int) -> dict[int, list[int]]:
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
    def _build_optim(spec: EncoderSpec, n_subjects: int, cfg: TrainConfig, device: str):
        """Encoder (by `cfg.model`, from the registry) + logit-scale + (optional) subject discriminator, all under
        one AdamW."""
        encoder = EncoderRegistry.build_encoder(cfg.model, spec).to(device)
        logit_scale = torch.nn.Parameter(torch.tensor(np.log(1 / 0.07), dtype=torch.float32, device=device))
        # per-encoder optimizer groups if the encoder defines them (foundation: discriminative backbone/head LR),
        # else one group at cfg.lr (NICE — unchanged).
        make_groups = getattr(encoder, "param_groups", None)
        groups = make_groups(cfg.lr) if make_groups else [{"params": [*encoder.parameters()], "lr": cfg.lr}]
        groups.append({"params": [logit_scale], "lr": cfg.lr})
        discriminator = None
        if cfg.adversarial:
            discriminator = SubjectDiscriminator(spec.embed_dim, n_subjects).to(device)
            groups.append({"params": [*discriminator.parameters()], "lr": cfg.lr})
            logger.info(f"domain-adversarial: {n_subjects} subjects, lambda {cfg.adv_lambda}, weight {cfg.adv_weight}")
        return encoder, logit_scale, discriminator, torch.optim.AdamW(groups, lr=cfg.lr, weight_decay=cfg.weight_decay)

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
        encoder, logit_scale, discriminator, optimizer = TrainNice._build_optim(
            spec, int(subject.max()) + 1, cfg, device)

        neighbor_groups = None
        if cfg.sampling == "clip_hard":                        # CLIP-prior hard-negative neighbours (bd 4ru)
            neighbor_groups = TrainNice._concept_neighbor_groups(
                train_targets[fit_indices], train_concept[fit_indices], cfg.concepts_per_batch)

        fit_data = _FitArrays(train_eeg, train_targets, train_concept, fit_indices, subject)
        best_val, best_state, best_epoch, since_improved = -1.0, None, -1, 0
        for epoch in range(cfg.epochs):
            lam = (TrainNice._dann_lambda(epoch / max(1, cfg.epochs - 1), cfg.adv_lambda)
                   if cfg.adv_lambda_ramp else cfg.adv_lambda)
            steps = TrainNice._epoch_steps(fit_data, neighbor_groups, cfg, rng, epoch_n)
            encoder.train()
            total_loss, n_batches = 0.0, 0
            epoch_start = time.perf_counter()
            for eeg_batch, target_batch, subj_batch in steps:
                eeg_batch = eeg_batch.to(device)
                target_batch = torch.nn.functional.normalize(target_batch.to(device), dim=-1)
                optimizer.zero_grad()
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=(device == "cuda" and cfg.amp)):
                    z = encoder(eeg_batch)
                    loss = Nice.clip_infonce(z, target_batch, logit_scale.exp().clamp(max=100),
                                        hard_beta=cfg.hard_beta, soft_tau=cfg.soft_tau)
                    if discriminator is not None:              # push encoder to be subject-invariant (GRL, bd 36g)
                        subj_logits = discriminator(z, lam)
                        loss = loss + cfg.adv_weight * F.cross_entropy(subj_logits, subj_batch.to(device))
                loss.backward()                                # bf16 autocast needs no GradScaler (unlike fp16)
                optimizer.step()
                total_loss += loss.item()
                n_batches += 1

            train_s = time.perf_counter() - epoch_start
            if epoch % cfg.val_every == 0 or epoch == cfg.epochs - 1:      # stride the big val eval (bd)
                val_top1 = TrainNice.evaluate(
                    encoder, RetrievalSet(val_eeg, val_labels, val_bank), device)["single_trial"][1]
                if val_top1 > best_val:
                    best_val, best_epoch, since_improved = val_top1, epoch, 0
                    best_state = {k: v.detach().cpu().clone() for k, v in encoder.state_dict().items()}
                else:
                    since_improved += 1
                if epoch % 5 == 0 or epoch == cfg.epochs - 1:
                    logger.info(f"ep {epoch:3d}  loss {total_loss / max(1, n_batches):.3f}  "
                          f"val-top1 {val_top1*100:.2f}%"
                          f"  {train_s:.1f}s ({n_batches / train_s:.0f} batch/s)")
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

        train_eeg, train_concept, train_targets, train_subj = TrainNice._load_split(
            train_subjects, "training", cfg.resample, recenter=cfg.recenter, shrinkage=cfg.recenter_shrinkage)
        test_eeg, test_concept, _, _ = TrainNice._load_split(
            [test_subject], "test", cfg.resample, recenter=cfg.recenter, shrinkage=cfg.recenter_shrinkage)
        subj_map = {s: i for i, s in enumerate(sorted(set(train_subjects)))}
        subj_idx = np.array([subj_map[int(s)] for s in train_subj], dtype=np.int64) if cfg.adversarial else None
        test_bank = clip_targets.ClipTargets.concept_prototypes("test")
        logger.info(f"train {train_eeg.shape} -> CLIP {train_targets.shape} | "
              f"test {test_eeg.shape} ({int(test_concept.max())+1} concepts)")

        encoder, stats = TrainNice.train_encoder(
            TrainData(train_eeg, train_targets, train_concept, subj_idx), cfg, device)
        test = TrainNice.evaluate(encoder, RetrievalSet(test_eeg, test_concept, test_bank), device)
        return {"regime": regime, "train": train_subjects, "test": test_subject, **stats,
                "chance_top1": 1 / (int(test_concept.max()) + 1), **test}


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
    ap.add_argument("--seed", type=int, default=None)
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
