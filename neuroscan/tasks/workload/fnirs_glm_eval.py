"""GLM-β vs the mean/slope/peak collapse — does a model-based (HRF-template) amplitude beat the hand-collapsed
features on fNIRS n-back? Fixed everything else (shrinkage-LDA, subject-grouped CV, no cleaning), so the delta
is the feature. Within AND cross-subject; plus the per-boundary binary breakdown, because the interesting
question is whether GLM-β sharpens the *decodable* boundary (0-vs-load) or — the real test — finally cracks
2-vs-3 (which the collapse and the literature say is physiologically absent).

    python -m neuroscan.tasks.workload.fnirs_glm_eval
"""
from __future__ import annotations

import numpy as np

from baselines.fnirs.features import FnirsLda
from baselines.fnirs.glm import GlmBeta
from core.data import store
from core.data.fnirs.base import FnirsCfg
from neuroscan.evaluation import metrics

_SEEDS = [0, 1, 2]
_K = 5
_DATASET = "shin2017_nback"
_ARMS = [("collapse (mean+slope+peak)", lambda: FnirsLda()),
         ("glm-β (HRF, no deriv)", lambda: GlmBeta(derivatives=False)),
         ("glm-β (HRF + derivs)", lambda: GlmBeta(derivatives=True))]


def _cv(build, X, y, groups, grouped, classes=None):
    from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
    if classes is not None:
        m = np.isin(y, classes); X, y, groups = X[m], y[m], groups[m]
        y = (y == classes[1]).astype(int)
    accs = []
    for seed in _SEEDS:
        sp = (StratifiedGroupKFold(_K, shuffle=True, random_state=seed) if grouped
              else StratifiedKFold(_K, shuffle=True, random_state=seed))
        for tr, te in sp.split(X, y, groups if grouped else None):
            pred = build().fit(X[tr], y[tr]).predict_proba(X[te]).argmax(1)
            accs.append(metrics.accuracy(y[te], pred))
    return float(np.mean(accs)), float(np.std(accs))


def main():
    meta = store.load(_DATASET, FnirsCfg())                          # clean=None (no filter, as requested)
    X, y = store.gather(meta)
    groups = meta["subject"].to_numpy()
    print(f"GLM-β vs collapse · Shin n-back · {len(y)} blocks · {meta['subject'].n_unique()} subj · "
          f"{len(_SEEDS)}x{_K}-fold · no cleaning\n")
    print(f"  {'arm':<28}{'within':>8}{'cross':>8}   {'0v-load':>8}{'2-v-3':>7}   (3-class chance .333)")
    for name, build in _ARMS:
        wa, _ = _cv(build, X, y, groups, grouped=False)
        ca, _ = _cv(build, X, y, groups, grouped=True)
        ol, _ = _cv(build, X, y, groups, grouped=True, classes=[0, 2])   # 0-back vs 3-back (load on/off proxy)
        lv, _ = _cv(build, X, y, groups, grouped=True, classes=[1, 2])   # 2-back vs 3-back (level)
        print(f"  {name:<28}{wa:>8.3f}{ca:>8.3f}   {ol:>8.3f}{lv:>7.3f}")
    print("\n  0v-load / 2-v-3 are cross-subject binary (chance 0.5). GLM-β sharpening 0v-load but NOT 2-v-3\n"
          "  = it's physiology, not features (the honest diagnostic).")


if __name__ == "__main__":
    main()
