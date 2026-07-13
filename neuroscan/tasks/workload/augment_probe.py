"""Domain-randomization decode probe (bd jdh) — does augmenting the train set with nuisance-randomized copies
narrow the cross-subject fNIRS gap ON A TASK WITH HEADROOM?

Wires `domain_randomize` into a real cross-subject decode (the consumer the operator was stripped for lacking).
Two questions, in order:

  1. HEADROOM — is there a cross-subject gap to close? within-subject vs cross-subject `FnirsLda` on the binary
     **0-back vs 2-back** contrast (a stronger, less ceiling-bound boundary than the capped 2-vs-3). No gap =
     no nuisance-driven loss to fix, and augmentation can't help by construction.
  2. THE ARM — augment each fold's TRAIN set with K domain-randomized copies (log-normal coupling gain +
     common-mode systemic burst + timing jitter), refit, re-score cross-subject. Beat the un-augmented
     cross-subject acc = the nuisance-diversified train transfers better. ≤ it = augmentation adds nuisance
     variety, not discriminative signal (the honest jdh outcome unless headroom exists).

    python -m neuroscan.tasks.workload.augment_probe
"""
from __future__ import annotations

import logging

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

from baselines.fnirs.features import FnirsLda
from core.data import store
from core.data.fnirs.augment import AugConfig, Augment
from core.data.fnirs.base import FnirsCfg
from neuroscan.evaluation import metrics
from neuroscan.tasks.cli import Cli

logger = logging.getLogger(__name__)

_FS = 10.0                       # Shin fNIRS native rate (FnirsCfg resample=None)
_SEEDS, _K = [0], 5              # 1 seed default (escalate-on-signal, bd); k-fold = validity not rigor
_CLASSES = (0, 2)                # 0-back vs 2-back — the stronger contrast with cross-subject headroom
_N_AUG = 3                       # domain-randomized copies appended per real train epoch
_MIN_HEADROOM = 0.02             # within−cross gap below this = no nuisance loss for augmentation to fix


class AugmentProbe:
    """Domain-randomization decode probe helpers (bd jdh) — the free helpers folded in as staticmethods."""

    @staticmethod
    def _build():
        """Paired (HbO, HbR) epochs restricted to the binary `_CLASSES` contrast, relabelled 0/1."""
        meta = store.Store.load("shin2017_nback", FnirsCfg())
        x, y, g = [], [], []
        for s in sorted(meta["subject"].unique().to_list()):
            xs, ys = store.Store.gather(meta.filter(meta["subject"] == s))
            m = np.isin(ys, _CLASSES)
            x.append(xs[m])
            y.append((ys[m] == _CLASSES[1]).astype(int))
            g.append(np.array([s] * int(m.sum())))
        return np.concatenate(x), np.concatenate(y), np.concatenate(g)

    @staticmethod
    def _augment(x_tr: np.ndarray, y_tr: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray]:
        """Append `_N_AUG` domain-randomized copies of the train epochs (HbO=0:36, HbR=36:72)."""
        xs, ys = [x_tr], [y_tr]
        for k in range(_N_AUG):
            o, r = Augment.domain_randomize(x_tr[:, :36], x_tr[:, 36:], _FS, AugConfig(), seed=seed * 97 + k)
            xs.append(np.concatenate([o, r], axis=1))
            ys.append(y_tr)
        return np.concatenate(xs), np.concatenate(ys)

    @staticmethod
    def _cross(x, y, g, *, augment: bool) -> tuple[float, float]:
        """Cross-subject grouped k-fold `FnirsLda`; optionally augment each fold's train set."""
        accs = []
        for seed in _SEEDS:
            for tr, te in StratifiedGroupKFold(_K, shuffle=True, random_state=seed).split(x, y, g):
                x_tr, y_tr = (AugmentProbe._augment(x[tr], y[tr], seed) if augment else (x[tr], y[tr]))
                proba = FnirsLda().fit(x_tr, y_tr).predict_proba(x[te])
                accs.append(metrics.Metrics.accuracy(y[te], proba.argmax(1)))
        return float(np.mean(accs)), float(np.std(accs))

    @staticmethod
    def _within(x, y, g) -> tuple[float, float]:
        """Within-subject k-fold (pooled), the ceiling that bounds the cross-subject headroom."""
        accs = []
        for seed in _SEEDS:
            for tr, te in StratifiedKFold(_K, shuffle=True, random_state=seed).split(x, y):
                proba = FnirsLda().fit(x[tr], y[tr]).predict_proba(x[te])
                accs.append(metrics.Metrics.accuracy(y[te], proba.argmax(1)))
        return float(np.mean(accs)), float(np.std(accs))


def main():
    Cli.setup_logging()
    x, y, g = AugmentProbe._build()
    logger.info(f"{len(y)} epochs · {len(set(g.tolist()))} subj · 0-back vs 2-back · {len(_SEEDS)}x{_K}-fold")
    a_win, s_win = AugmentProbe._within(x, y, g)
    a_cs, s_cs = AugmentProbe._cross(x, y, g, augment=False)
    headroom = a_win - a_cs
    logger.info(f"  within-subject   acc {a_win:.3f} ± {s_win:.3f}")
    logger.info(f"  cross-subject    acc {a_cs:.3f} ± {s_cs:.3f}   (headroom {headroom:+.3f})")
    if headroom < _MIN_HEADROOM:
        logger.info("  -> no cross-subject headroom: augmentation can't close a gap that isn't there")
    a_aug, s_aug = AugmentProbe._cross(x, y, g, augment=True)
    logger.info(f"  cross + aug (x{_N_AUG}) acc {a_aug:.3f} ± {s_aug:.3f}")
    verdict = "AUGMENTATION HELPS" if a_aug > a_cs + 0.01 else "fair null (nuisance variety, not signal)"
    logger.info(f"  Δ aug − plain cross: {a_aug - a_cs:+.3f}  ->  {verdict}")


if __name__ == "__main__":
    main()
