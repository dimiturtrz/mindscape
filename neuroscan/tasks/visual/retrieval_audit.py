"""Robustness audit for EEG->image retrieval — quantify how much the commonly-quoted numbers inflate.

The EEG->image field routinely reports the *within-subject, concept-averaged* top-k, which leaks two ways:
the model has seen the test person, and averaging repeats borrows test-set structure. The robust cell is
CROSS-subject, SINGLE-trial, concept-disjoint. This runner trains both a within- and a cross-subject encoder
for each held-out subject (reusing train_nice.train), assembles the 2x2 (subject x averaging) top-1/5 grid,
and reports the inflation of every leaky cell over the robust one. It also verifies the dataset's zero-shot
claim directly (train vs test concept sets are disjoint) instead of taking it on faith.

    python -m neuroscan.tasks.visual.retrieval_audit --subjects 1 2 3        # hold out each, train the rest
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
from pydantic import BaseModel

from core.data.eeg import things_eeg2 as things
from neuroscan.tasks.visual.train_nice import TrainConfig, train

logger = logging.getLogger(__name__)

_CELLS = ("within_single", "within_avg", "cross_single", "cross_avg")
_ROBUST = "cross_single"        # the defensible number every leaky cell is measured against


class AuditConfig(BaseModel):
    """Which subjects to hold out + the shared training hyperparameters. `subjects` are evaluated one at a
    time as the held-out test subject; `within` trains on that subject alone, `cross` on all the others."""
    subjects: tuple[int, ...]
    all_subjects: tuple[int, ...] = ()      # pool the cross arm trains on (default: every downloaded subject)
    epochs: int = 40
    batch: int = 512
    lr: float = 3e-4
    resample: float = 250.0
    seed: int = 0
    patience: int = 8


def _concept_names(split_key: str) -> set[str]:
    """The concept IDENTITIES of a split (folder names minus the split-local 'NNNNN_' index prefix). The
    integer concept index is re-numbered from 0 WITHIN each split, so identity has to be the name, not the
    index — comparing indices would falsely report every test concept as 'seen'."""
    meta = things.ThingsEeg2._meta()
    return {str(name)[6:] if str(name)[:5].isdigit() else str(name)
            for name in meta[f"{split_key}_img_concepts"]}


def verify_concept_disjoint() -> dict:
    """Check the THINGS-EEG2 train/test concept sets don't overlap — the dataset's zero-shot claim, verified
    on concept NAMES rather than assumed. Returns the two set sizes + the overlap (must be 0)."""
    train_names, test_names = _concept_names("train"), _concept_names("test")
    overlap = train_names & test_names
    if overlap:
        raise ValueError(f"train/test concepts overlap by {len(overlap)} — retrieval is NOT zero-shot")
    return {"n_train_concepts": len(train_names), "n_test_concepts": len(test_names),
            "concept_overlap": len(overlap)}


def _cells_from_result(result: dict, regime: str) -> dict:
    """Pull the (single-trial, concept-avg) top-1/5 out of one train() result into flat `{regime}_{avg}` keys."""
    return {f"{regime}_single": dict(result["single_trial"]), f"{regime}_avg": dict(result["concept_avg"])}


def summarize(rows: list[dict], ks: tuple[int, ...] = (1, 5)) -> dict:
    """Mean each cell over held-out subjects, then the inflation of every leaky cell over the robust one.

    `rows` = one dict per held-out subject, each carrying all four `_CELLS` -> {k: acc}. Pure: no data/torch,
    so the grid + delta logic is unit-testable on synthetic accuracies.
    """
    grid = {cell: {k: float(np.mean([r[cell][k] for r in rows])) for k in ks} for cell in _CELLS}
    inflation = {cell: {k: grid[cell][k] - grid[_ROBUST][k] for k in ks}
                 for cell in _CELLS if cell != _ROBUST}
    return {"n_subjects": len(rows), "grid": grid, "robust_cell": _ROBUST, "inflation_over_robust": inflation}


def _load_row(path: Path) -> dict:
    """Read a checkpointed subject row, restoring the int top-k keys JSON turned into strings."""
    raw = json.loads(path.read_text())
    return {cell: {int(k): v for k, v in cell_scores.items()} for cell, cell_scores in raw.items()}


def run_audit(cfg: AuditConfig, ckpt_dir: str = "runs/retrieval_audit_ckpt") -> dict:
    """Train the within- + cross-subject encoder for each held-out subject, assemble the robustness grid.

    Checkpoints each subject's row to `ckpt_dir` AS IT COMPLETES and resumes from it (bd 9js) — a stall on the
    Nth subject never loses the first N-1 (this exact loss nearly happened during the qoa run)."""
    pool = cfg.all_subjects or tuple(things.ThingsEeg2.subjects())
    train_cfg = TrainConfig(epochs=cfg.epochs, batch=cfg.batch, lr=cfg.lr,
                            resample=cfg.resample, seed=cfg.seed, patience=cfg.patience)
    ckpt = Path(ckpt_dir)
    ckpt.mkdir(parents=True, exist_ok=True)
    rows = []
    for i, test_subject in enumerate(cfg.subjects, 1):
        row_path = ckpt / f"subject_{test_subject}.json"
        if row_path.exists():
            logger.info(f"[{i}/{len(cfg.subjects)}] subject {test_subject}: resumed from checkpoint")
            rows.append(_load_row(row_path))
            continue
        others = [s for s in pool if s != test_subject]
        if not others:
            raise ValueError(f"cross-subject arm needs >=2 subjects in the pool; got {pool}")
        logger.info(f"[{i}/{len(cfg.subjects)}] subject {test_subject}: training within + cross ...")
        within = train([test_subject], test_subject, train_cfg)
        cross = train(others, test_subject, train_cfg)
        row = {**_cells_from_result(within, "within"), **_cells_from_result(cross, "cross")}
        row_path.write_text(json.dumps(row))                       # checkpoint before moving on
        rows.append(row)
        logger.info(f"[{i}/{len(cfg.subjects)}] subject {test_subject}: within-avg-top1 "
              f"{within['concept_avg'][1]*100:.1f}% -> robust cross-single-top1 {cross['single_trial'][1]*100:.1f}%")
    return {"disjoint": verify_concept_disjoint(), **summarize(rows), "per_subject": rows}


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    for lib_name in ("mne", "moabb", "braindecode"):
        logging.getLogger(lib_name).setLevel(logging.WARNING)
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--subjects", type=int, nargs="+", required=True, help="subjects to hold out one at a time")
    ap.add_argument("--epochs", type=int, default=AuditConfig(subjects=(1,)).epochs)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    result = run_audit(AuditConfig(subjects=tuple(args.subjects), epochs=args.epochs, seed=args.seed))
    logger.info(json.dumps({k: v for k, v in result.items() if k != "per_subject"}, indent=2))
    if args.out:
        Path(args.out).write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
