"""Where do the 3-way workload errors actually concentrate? — the diagnostic that says what's improvable.

If EEG already nails 0-vs-load and only fails on 2-vs-3, fusion can't help (fNIRS is redundant there). If EEG
ALSO errs on 0-vs-load, fusion has a real job (fNIRS separates rest from load) and boundary-aware fusion can
lift the REAL 3-way number. Prints the EEG-alone confusion matrix + per-boundary accuracy.

    python -m neuroscan.tasks.workload.workload_confusion
"""
from __future__ import annotations

import logging

import numpy as np
from pyriemann.estimation import Covariances
from sklearn.model_selection import StratifiedGroupKFold

from baselines.eeg import transfer
from core.data import store
from core.data.eeg.base import EpochCfg
from neuroscan.evaluation import metrics

logger = logging.getLogger(__name__)

_EEG_CFG = EpochCfg(fmin=4, fmax=30, tmin=0.0, tmax=40.0, resample=100.0)
_SEEDS, _K = [0, 1, 2], 5


def _cov(X):
    return Covariances("oas").transform(X.astype(np.float64))


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    for _n in ("mne", "moabb", "braindecode"):
        logging.getLogger(_n).setLevel(logging.WARNING)
    me = store.load("shin2017_nback_eeg", _EEG_CFG)
    subs = sorted(me["subject"].unique().to_list())
    Cs, ys, gs = [], [], []
    for s in subs:
        X, y = store.gather(me.filter(me["subject"] == s))
        Cs.append(_cov(X))
        ys.append(y)
        gs.append(np.array([s] * len(y)))
    C, y, g = np.concatenate(Cs), np.concatenate(ys), np.concatenate(gs)
    n_cls = int(y.max()) + 1

    conf = np.zeros((n_cls, n_cls), int)
    accs = []
    for seed in _SEEDS:
        for tr, te in StratifiedGroupKFold(_K, shuffle=True, random_state=seed).split(C, y, g):
            pred = transfer.zero_shot_predict(C[tr], y[tr], g[tr], C[te], scale=False, target_groups=g[te]).argmax(1)
            accs.append(metrics.accuracy(y[te], pred))
            for t, p in zip(y[te], pred, strict=True):
                conf[t, p] += 1

    logger.info(f"EEG-alone re-centered Riemann · 3-way acc {np.mean(accs):.3f} · classes 0-back/2-back/3-back")
    logger.info("confusion (row=true, col=pred), summed over seeds×folds:")
    logger.info("        p0    p2    p3")
    for i, name in enumerate(["0-back", "2-back", "3-back"]):
        logger.info(f"  {name}  " + "  ".join(f"{conf[i,j]:4d}" for j in range(n_cls)))

    # 2 vs 3 among true-load trials that weren't called 0
    load_rows = conf[1:, 1:]                                          # true 2/3 x pred 2/3
    acc_2v3 = np.trace(load_rows) / load_rows.sum()
    zerorow = conf[0]
    logger.info(f"\n  0-vs-load separability (is rest confused with load?): "
          f"0-back called load {100*(zerorow[1]+zerorow[2])/zerorow.sum():.0f}% · "
          f"load called 0-back {100*conf[1:,0].sum()/conf[1:].sum():.0f}%")
    logger.info(f"  2-vs-3 accuracy among true-load trials: {acc_2v3:.3f}")
    logger.info("  -> if 0-vs-load leaks, fusion (fNIRS strong there) can fix it; if clean, the loss is all 2-vs-3.")


if __name__ == "__main__":
    main()
