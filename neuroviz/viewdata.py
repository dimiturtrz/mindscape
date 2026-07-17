"""Shared building blocks for the neuroviz subject-exporters (`export`, `export_fnirs`,
`export_eeg_workload`).

They all: parse the same `--subject/--out` CLI, and turn a held-out LOSO decode into the same
per-class `{truth, pred, probs, correct}` report + a `{acc, chance, regime, decoder}` score. One home
for those contracts so the on-disk shape the web viewer reads has a single source of truth.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np


@dataclass
class Decode:
    """One subject's held-out LOSO decode + how to label it — the input to `ViewData.prediction_report`."""
    y: np.ndarray                    # true class ids
    pred: np.ndarray                 # predicted class ids
    probs: np.ndarray                # per-trial class probabilities
    chance: float                    # chance level for this task
    decoder: str                     # human-readable decoder name


class ViewData:
    """Exporter-shared CLI + prediction-report builders (the viewer's JSON contract)."""

    @staticmethod
    def subject_args(doc: str, default_out: str = "neuroviz/web/data") -> argparse.Namespace:
        """The `--subject/--out` CLI every subject-exporter shares."""
        ap = argparse.ArgumentParser(description=doc)
        ap.add_argument("--subject", type=int, default=1)
        ap.add_argument("--out", default=default_out)
        return ap.parse_args()

    @staticmethod
    def prediction_report(id2name: dict[int, str], d: Decode) -> tuple[dict, dict]:
        """Per-class `{truth, pred, probs, correct}` for one shown example trial + the honest
        cross-subject `{acc, chance, regime, decoder}` score. `id2name` maps class id -> display name."""
        per = {}
        for c in sorted(np.unique(d.y).tolist()):
            i = int((d.y == c).argmax())                      # the example trial shown for this class
            name = id2name[c]
            per[name] = {"truth": name, "pred": id2name[int(d.pred[i])],
                         "probs": [round(float(p), 3) for p in d.probs[i]],
                         "correct": bool(d.pred[i] == c)}
        score = {"acc": round(float((d.pred == d.y).mean()), 3), "chance": round(d.chance, 3),
                 "regime": "cross-subject (LOSO)", "decoder": d.decoder}
        return per, score
