"""Leave-one-subject-out retrieval eval (bd 830) — harden the cross-subject perception number across every
held-out subject, matched between encoders.

For each test subject in the pool, train on all the others and zero-shot-retrieve on the held-out one (via
`train_nice.train`), for each model + seed. Reports per-fold single-trial / concept-avg top-1/5 and the
**mean ± SE over folds×seeds** — the decision quantity for "does encoder A beat B beyond fold noise?" (the
single-seed 2-subject bji result showed a win; this is its hardening). Same folds for every model, so the
comparison is matched. LOSO folds are independent training runs (minutes–hours each) — bound the pool/seeds
to the wall-clock you have; whatever is dropped is logged, not silently capped.

    python -m neuroscan.tasks.visual.loso_eval --models nice cbramod_ft --subjects 1 2 3 4 5 \
        --config neuroscan/tasks/visual/configs/perception_converged.json \
        --resample 200 --lr 3e-4 --epochs 150 --patience 25
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np

from neuroscan.tasks.cli import Cli
from neuroscan.tasks.visual.train_nice import TrainConfig, TrainNice

logger = logging.getLogger(__name__)

_METRICS = [("single_trial", "1"), ("single_trial", "5"), ("concept_avg", "1")]
_N_PAIR = 2   # print the A−B delta only when exactly two models are compared


class LosoEval:
    """Leave-one-subject-out retrieval eval — the free helpers folded in as staticmethods (public names kept).
    `_fold` runs one held-out-subject training run; `_summary` reduces folds to mean ± SE."""

    @staticmethod
    def _fold(model: str, seed: int, test_subject: int, pool: list[int], base: dict) -> dict:
        """One LOSO fold: train on `pool \\ {test_subject}`, retrieve on the held-out subject."""
        train_subjects = [s for s in pool if s != test_subject]
        cfg = TrainConfig(**{**base, "model": model, "seed": seed})
        return TrainNice.train(train_subjects, test_subject, cfg)

    @staticmethod
    def _summary(folds: list[dict]) -> dict:
        """Mean ± SE over folds for each reported metric (SE = std / √n_folds — the decision quantity)."""
        out = {}
        n = len(folds)
        for block, k in _METRICS:
            vals = np.array([f[block][k] for f in folds], dtype=float)
            out[f"{block}.{k}"] = (float(vals.mean()), float(vals.std() / np.sqrt(max(1, n))))
        return out


def main():
    Cli.setup_logging()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--models", nargs="+", required=True, help="encoder names to compare on the SAME folds")
    ap.add_argument("--subjects", type=int, nargs="+", required=True, help="LOSO subject pool")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--config", default=None, help="JSON of TrainConfig fields (recipe home)")
    ap.add_argument("--resample", type=float, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--patience", type=int, default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    base = json.loads(Path(args.config).read_text()) if args.config else {}
    for k in ("resample", "lr", "epochs", "patience"):
        if getattr(args, k) is not None:
            base[k] = getattr(args, k)

    logger.info(f"LOSO · pool {args.subjects} · models {args.models} · seeds {args.seeds} "
                f"· {len(args.subjects) * len(args.seeds)} folds/model")
    results = {}
    for model in args.models:
        folds = [LosoEval._fold(model, seed, test, args.subjects, base)
                 for seed in args.seeds for test in args.subjects]
        results[model] = {"folds": folds, "summary": LosoEval._summary(folds)}
        logger.info(f"\n=== {model} ===")
        for key, (mean, se) in results[model]["summary"].items():
            logger.info(f"  {key:16s} {mean * 100:5.2f}% ± {se * 100:.2f}")

    if len(args.models) == _N_PAIR:
        a, b = args.models
        delta = results[b]["summary"]["single_trial.1"][0] - results[a]["summary"]["single_trial.1"][0]
        logger.info(f"\nΔ ({b} − {a}) single_trial.1: {delta * 100:+.2f}pp")
    if args.out:
        Path(args.out).write_text(json.dumps(results, indent=2))
        logger.info(f"-> {args.out}")


if __name__ == "__main__":
    main()
