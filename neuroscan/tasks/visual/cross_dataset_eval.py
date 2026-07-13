"""Cross-dataset zero-shot EEG->image — train NICE on THINGS-EEG1, retrieve on THINGS-EEG2.

The hardest leakage-free generalization test: different people, different rig (63-ch BrainVision @10 Hz RSVP vs
EEG2's setup), SAME 1,854 THINGS concepts. EEG2's 200 test concepts also occur in EEG1, so they're held out
of EEG1 training (cross_dataset.holdout_mask) to keep it concept-zero-shot; retrieval then runs on EEG2's
200-concept CLIP bank. The per-trial training target is the concept's shared CLIP prototype (looked up by
name), so both datasets live in one embedding space. Report top-k + confidence calibration, to compare
against the within-dataset cross-subject number from retrieval_audit.

    python -m neuroscan.tasks.visual.cross_dataset_eval --eeg1-subjects 32 33 34 35 36 --eeg2-subjects 1 2
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import torch
from pydantic import BaseModel

from core.data.eeg import things_eeg1, things_eeg2
from neuroscan.evaluation import cross_dataset as bridge
from neuroscan.evaluation.retrieval import Retrieval
from neuroscan.tasks.cli import Cli
from neuroscan.tasks.visual import clip_targets
from neuroscan.tasks.visual.train_nice import RetrievalSet, TrainConfig, TrainData, TrainNice

logger = logging.getLogger(__name__)

_EVAL_BATCH = 512
_LOGIT_SCALE = float(np.log(1 / 0.07))   # the CLIP temperature the encoder trains with — reuse for calibration


class CrossDatasetConfig(BaseModel):
    """EEG1 subjects to train on, EEG2 subjects whose test split we retrieve on, + the shared training knobs."""
    eeg1_subjects: tuple[int, ...]
    eeg2_subjects: tuple[int, ...] = (1, 2)
    resample: float = 250.0
    epochs: int = 40
    batch: int = 512
    lr: float = 3e-4
    seed: int = 0
    patience: int = 8


class CrossDatasetEval:
    """Cross-dataset zero-shot EEG->image (train EEG1, retrieve EEG2) — the free helpers folded in as
    staticmethods (public names kept). `run` trains on EEG1 with EEG2's test concepts held out and retrieves
    on EEG2's test split; `_shared_prototypes` builds the name->CLIP bridge over both datasets."""

    @staticmethod
    def _shared_prototypes() -> tuple[dict, list[str]]:
        """{concept name -> shared CLIP prototype} over all 1,854 THINGS concepts (EEG2 train + test prototypes),
        and the 200 test-concept names in bank order (the retrieval candidate set)."""
        names_train = [path.name[6:] for path in clip_targets.ClipTargets.concept_dirs("training")]
        names_test = [path.name[6:] for path in clip_targets.ClipTargets.concept_dirs("test")]
        proto_train = clip_targets.ClipTargets.concept_prototypes("training")
        proto_test = clip_targets.ClipTargets.concept_prototypes("test")
        name_to_proto = {**dict(zip(names_train, proto_train, strict=True)),
                         **dict(zip(names_test, proto_test, strict=True))}
        return name_to_proto, names_test

    @staticmethod
    @torch.no_grad()
    def _embed(encoder, eeg: np.ndarray, device: str) -> np.ndarray:
        encoder.eval()
        return torch.cat([encoder(torch.tensor(eeg[i:i + _EVAL_BATCH]).to(device)).cpu()
                          for i in range(0, len(eeg), _EVAL_BATCH)]).numpy()

    @staticmethod
    def run(cfg: CrossDatasetConfig) -> dict:
        """Train on EEG1 (zero-shot holdout of EEG2's test concepts), retrieve on EEG2's test split."""
        torch.manual_seed(cfg.seed)
        np.random.seed(cfg.seed)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        train_cfg = TrainConfig(epochs=cfg.epochs, batch=cfg.batch, lr=cfg.lr, resample=cfg.resample,
                                seed=cfg.seed, patience=cfg.patience)
        name_to_proto, eval_names = CrossDatasetEval._shared_prototypes()
        holdout = set(eval_names)          # EEG2's 200 test concepts — never seen in EEG1 training

        e1_ch, e2_ch = things_eeg1.ThingsEeg1.channels(), things_eeg2.ThingsEeg2.channels()
        common_ch = bridge.CrossDataset.common_channel_order(e2_ch, e1_ch)      # 62 shared electrodes, in EEG2 order
        logger.info(f"montage align: {len(common_ch)}/{len(e1_ch)} shared electrodes "
              f"(EEG1-only {sorted(set(e1_ch) - set(e2_ch))}, EEG2-only {sorted(set(e2_ch) - set(e1_ch))})")

        e1_eeg, e1_concept, _, _ = things_eeg1.ThingsEeg1.get_epochs(
            list(cfg.eeg1_subjects), things_eeg1.ThingsEeg1EpochCfg(resample=cfg.resample))
        e1_eeg = bridge.CrossDataset.align_channels(e1_eeg, e1_ch, common_ch)   # reorder EEG1 to the shared montage
        keep = bridge.CrossDataset.holdout_mask(e1_concept, holdout) & np.array([n in name_to_proto for n in e1_concept])
        e1_eeg, e1_names = e1_eeg[keep], e1_concept[keep]
        targets = np.stack([name_to_proto[name] for name in e1_names]).astype(np.float32)
        name_id = {name: i for i, name in enumerate(sorted(set(e1_names)))}      # concept ids for the val split
        concept_ids = np.array([name_id[name] for name in e1_names])
        logger.info(f"EEG1 train: {len(e1_names)} epochs, {len(name_id)} concepts (EEG2-test held out) "
              f"-> CLIP {targets.shape}")

        encoder, stats = TrainNice.train_encoder(TrainData(e1_eeg, targets, concept_ids), train_cfg, device)

        e2_eeg, e2_concept, _, _ = things_eeg2.ThingsEeg2.get_epochs(
            list(cfg.eeg2_subjects), things_eeg2.ThingsEpochCfg(split="test", resample=cfg.resample))
        e2_eeg = bridge.CrossDataset.align_channels(e2_eeg, e2_ch, common_ch)   # same shared montage the encoder trained on
        test_bank = clip_targets.ClipTargets.concept_prototypes("test")
        topk = TrainNice.evaluate(encoder, RetrievalSet(e2_eeg, e2_concept, test_bank), device)

        scores = CrossDatasetEval._embed(encoder, e2_eeg, device) @ test_bank.T
        metrics = Retrieval.retrieval_metrics(scores, e2_concept)
        calib = Retrieval.retrieval_calibration(scores, e2_concept, scale=_LOGIT_SCALE)
        logger.info(f"cross-dataset EEG1->EEG2: single top1 {topk['single_trial'][1]*100:.2f}%  "
              f"MRR {metrics['mrr']:.3f}  median-rank {metrics['median_rank']:.0f}/{len(eval_names)}  "
              f"PR-AUC {metrics['pr_auc']:.3f}  ECE {calib['ece']:.3f}")
        return {"direction": "eeg1->eeg2", "eeg1_subjects": list(cfg.eeg1_subjects),
                "eeg2_subjects": list(cfg.eeg2_subjects), "n_candidates": len(eval_names),
                "chance_top1": 1 / len(eval_names), **stats,
                "single_trial": topk["single_trial"], "concept_avg": topk["concept_avg"],
                "retrieval_metrics": metrics,
                "calibration": {"ece": calib["ece"], "conf_gap": calib["conf_gap"], "top1_acc": calib["top1_acc"]}}


def main():
    Cli.setup_logging()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--eeg1-subjects", type=int, nargs="+", required=True)
    ap.add_argument("--eeg2-subjects", type=int, nargs="+", default=[1, 2])
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    result = CrossDatasetEval.run(CrossDatasetConfig(eeg1_subjects=tuple(args.eeg1_subjects),
                                    eeg2_subjects=tuple(args.eeg2_subjects), epochs=args.epochs, seed=args.seed))
    logger.info(json.dumps(result, indent=2))
    if args.out:
        Path(args.out).write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
