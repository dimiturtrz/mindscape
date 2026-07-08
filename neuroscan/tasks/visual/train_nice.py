"""Train + evaluate the NICE EEG->image baseline on THINGS-EEG2.

Pipeline: adapter epochs (our own preprocessing off the raw) -> NICE encoder -> InfoNCE against the viewed
image's CLIP embedding -> zero-shot retrieval on the 200 held-out test concepts. The honest number is the
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

import numpy as np
import torch
from pydantic import BaseModel
from torch.utils.data import DataLoader, Dataset

from core.data.eeg import things_eeg2 as things
from neuroscan.models.nice import NiceConfig, NiceEncoder, clip_infonce, retrieval_topk
from neuroscan.tasks.visual import clip_targets

logger = logging.getLogger(__name__)

_EVAL_BATCH = 512   # batch >=2048 trips a cuDNN illegal-access on this conv shape (Blackwell / cu130)


class _EpochDataset(Dataset):
    """Numpy-backed — converts per sample, so the (large) training array is never copied into one torch
    tensor up front. `indices` optionally views a subset of the arrays without copying them (the fit split
    of a much larger epoch pile). Together these keep a full 9-subject LOSO (~38 GB of epochs) in RAM instead
    of OOM-ing on the doubled copies (torch tensor + boolean-mask slice)."""

    def __init__(self, eeg: np.ndarray, targets: np.ndarray, indices: np.ndarray | None = None):
        self.eeg, self.targets = eeg, targets
        self.indices = indices

    def __len__(self) -> int:
        return len(self.indices) if self.indices is not None else len(self.eeg)

    def __getitem__(self, idx: int):
        row = int(self.indices[idx]) if self.indices is not None else idx
        return torch.from_numpy(self.eeg[row]), torch.from_numpy(self.targets[row])


class TrainConfig(BaseModel):
    """Training hyperparameters (data-independent; the encoder's data-dependent shape is derived at fit)."""
    epochs: int = 40
    batch: int = 512
    lr: float = 3e-4
    weight_decay: float = 1e-4
    resample: float = 250.0
    seed: int = 0
    patience: int = 8            # early-stop patience on val-top1 (epochs)
    val_fraction: float = 0.1    # share of TRAINING concepts held out for leak-free model selection


def _clip_targets(image_files: np.ndarray, split: str) -> np.ndarray:
    """CLIP embedding per epoch, looked up by the image the subject viewed."""
    by_file = clip_targets.embeddings_by_file(split)
    return np.stack([by_file[name] for name in image_files]).astype(np.float32)


def _load_split(subjects: list[int], split: str, resample: float):
    epochs, concept, image_files, _ = things.get_epochs(subjects, split=split, resample=resample)
    return epochs, concept, _clip_targets(image_files, split)


@torch.no_grad()
def evaluate(encoder, eeg, eeg_concept, candidates, device, batch: int = _EVAL_BATCH) -> dict:
    """Single-trial + concept-averaged retrieval top-1/5 against a per-concept prototype bank."""
    encoder.eval()
    embedded = torch.cat([encoder(torch.tensor(eeg[i:i + batch]).to(device)).cpu()
                          for i in range(0, len(eeg), batch)])          # [N,512] normalized
    candidate_bank = torch.tensor(candidates)
    labels = torch.tensor(eeg_concept)
    single = retrieval_topk(embedded, candidate_bank, labels)
    n_concepts = int(eeg_concept.max()) + 1
    averaged = torch.stack([torch.nn.functional.normalize(embedded[labels == c].mean(0), dim=-1)
                            for c in range(n_concepts)])
    concept_avg = retrieval_topk(averaged, candidate_bank, torch.arange(n_concepts))
    return {"single_trial": single, "concept_avg": concept_avg}


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


def train(train_subjects: list[int], test_subject: int, cfg: TrainConfig) -> dict:
    """Fit the encoder (contrastive) with leak-free early stopping; return the test result at the best-val
    checkpoint. Reusable entry point — `main()` only builds the config and prints/saves this dict."""
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    regime = "within" if train_subjects == [test_subject] else "cross"
    logger.info(f"regime={regime}  train={train_subjects}  test={test_subject}  device={device}")

    train_eeg, train_concept, train_targets = _load_split(train_subjects, "training", cfg.resample)
    test_eeg, test_concept, _ = _load_split([test_subject], "test", cfg.resample)
    test_bank = clip_targets.concept_prototypes("test")
    logger.info(f"train {train_eeg.shape} -> CLIP {train_targets.shape} | "
          f"test {test_eeg.shape} ({int(test_concept.max())+1} concepts)")

    fit_mask, val_mask, val_labels, val_bank = _val_split(
        train_concept, train_targets, cfg.seed, cfg.val_fraction)
    fit_indices = np.where(fit_mask)[0]            # view, not a copy — fit is the bulk of the epoch pile
    val_eeg = train_eeg[val_mask]                  # small (one split fraction), copy is fine
    logger.info(f"early-stop val: {len(val_bank)} held-out train concepts, {int(val_mask.sum())} epochs")

    encoder = NiceEncoder(NiceConfig(n_channels=train_eeg.shape[1], n_times=train_eeg.shape[2],
                                     embed_dim=train_targets.shape[1])).to(device)
    logit_scale = torch.nn.Parameter(torch.tensor(np.log(1 / 0.07), dtype=torch.float32, device=device))
    optimizer = torch.optim.AdamW([*encoder.parameters(), logit_scale],
                                  lr=cfg.lr, weight_decay=cfg.weight_decay)
    loader = DataLoader(_EpochDataset(train_eeg, train_targets, fit_indices),
                        batch_size=cfg.batch, shuffle=True, drop_last=True)

    best_val, best_state, best_epoch, since_improved = -1.0, None, -1, 0
    for epoch in range(cfg.epochs):
        encoder.train()
        total_loss = 0.0
        for eeg_batch, target_batch in loader:
            eeg_batch = eeg_batch.to(device)
            target_batch = torch.nn.functional.normalize(target_batch.to(device), dim=-1)
            optimizer.zero_grad()
            loss = clip_infonce(encoder(eeg_batch), target_batch, logit_scale.exp().clamp(max=100))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        val_top1 = evaluate(encoder, val_eeg, val_labels, val_bank, device)["single_trial"][1]
        if val_top1 > best_val:
            best_val, best_epoch, since_improved = val_top1, epoch, 0
            best_state = {k: v.detach().cpu().clone() for k, v in encoder.state_dict().items()}
        else:
            since_improved += 1
        if epoch % 5 == 0 or epoch == cfg.epochs - 1:
            test = evaluate(encoder, test_eeg, test_concept, test_bank, device)
            logger.info(f"ep {epoch:3d}  loss {total_loss/len(loader):.3f}  val-top1 {val_top1*100:.2f}%  "
                  f"test single {test['single_trial'][1]*100:.2f}%/{test['single_trial'][5]*100:.2f}%  "
                  f"avg {test['concept_avg'][1]*100:.2f}%/{test['concept_avg'][5]*100:.2f}%")
        if since_improved >= cfg.patience:
            logger.info(f"early stop at ep {epoch} (best val = ep {best_epoch})")
            break

    encoder.load_state_dict(best_state)                            # report TEST at the best-VAL checkpoint
    test = evaluate(encoder, test_eeg, test_concept, test_bank, device)
    return {"regime": regime, "train": train_subjects, "test": test_subject,
            "best_val_epoch": best_epoch, "val_top1": best_val, "epochs_run": epoch + 1,
            "chance_top1": 1 / (int(test_concept.max()) + 1), **test}


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    for _n in ("mne", "moabb", "braindecode"):
        logging.getLogger(_n).setLevel(logging.WARNING)
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--train", type=int, nargs="+", default=[1], help="training subject id(s)")
    ap.add_argument("--test", type=int, default=1, help="held-out test subject id")
    ap.add_argument("--epochs", type=int, default=TrainConfig().epochs)
    ap.add_argument("--batch", type=int, default=TrainConfig().batch, help="<=1024 (cuDNN cap on this shape)")
    ap.add_argument("--lr", type=float, default=TrainConfig().lr)
    ap.add_argument("--resample", type=float, default=TrainConfig().resample)
    ap.add_argument("--seed", type=int, default=TrainConfig().seed)
    ap.add_argument("--patience", type=int, default=TrainConfig().patience)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cfg = TrainConfig(epochs=args.epochs, batch=args.batch, lr=args.lr,
                      resample=args.resample, seed=args.seed, patience=args.patience)
    result = train(args.train, args.test, cfg)
    logger.info(json.dumps(result, indent=2))
    if args.out:
        with open(args.out, "w") as f:
            json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()
