"""Honest confirmation of the feature-search finding — compare a few FIXED fNIRS feature recipes under
plain subject-grouped CV. No search here, so no selection optimism: each recipe is a fixed set of descriptor
families → shrinkage-LDA, scored by repeated-seeded StratifiedGroupKFold (subject-grouped, no leakage). This
is the clean test of "does slope alone match the mean+slope+peak baseline?" that the Optuna/torch studies
suggested but couldn't claim (they maximise over a search; this doesn't).

    python -m neuroscan.tasks.workload.fnirs_recipes
"""
from __future__ import annotations

import numpy as np

from core.data import store
from core.data.fnirs.base import FnirsCfg
from core.features import extract_bank, family_names
from neuroscan.evaluation import metrics

# Fixed recipes (family lists) — the comparison the search motivated. `amplitude` = the current fnirs_lda.
_RECIPES = {
    "full (15 families)":   family_names(),
    "amplitude (baseline)": ["mean", "slope", "peak"],
    "slope only":           ["slope"],
    "dynamics":             ["slope", "early_slope", "late_slope"],
    "mean only":            ["mean"],
    "peak only":            ["peak"],
    "all but slope":        [f for f in family_names() if f != "slope"],
}
_SEEDS = [0, 1, 2]
_K = 5


def _cv(F, fam, y, groups, families):
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.model_selection import StratifiedGroupKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    cols = np.isin(fam, families)
    Fr = F[:, cols]
    accs, kaps = [], []
    for seed in _SEEDS:
        for tr, te in StratifiedGroupKFold(_K, shuffle=True, random_state=seed).split(Fr, y, groups):
            clf = make_pipeline(StandardScaler(),
                                LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")).fit(Fr[tr], y[tr])
            pred = clf.predict(Fr[te])
            accs.append(metrics.accuracy(y[te], pred)); kaps.append(metrics.kappa(y[te], pred))
    return float(np.mean(accs)), float(np.std(accs)), float(np.mean(kaps))


def main():
    meta = store.load("shin2017_nback", FnirsCfg())
    X, y = store.gather(meta)
    groups = meta["subject"].to_numpy()
    F, fam = extract_bank(X)
    chance = 1.0 / (int(y.max()) + 1)
    print(f"fNIRS feature recipes · Shin n-back · {len(y)} blocks · {meta['subject'].n_unique()} subjects · "
          f"{len(_SEEDS)}x{_K}-fold GroupKFold · chance {chance:.3f}\n")
    print(f"  {'recipe':<22}{'acc':>7} {'±sd':>6} {'kappa':>7}  #fam")
    rows = []
    for name, fams in _RECIPES.items():
        acc, sd, kap = _cv(F, fam, y, groups, fams)
        rows.append((name, acc, sd, kap, len(fams)))
        print(f"  {name:<22}{acc:>7.3f} {sd:>6.3f} {kap:>7.3f}  {len(fams)}")
    base = dict((r[0], r[1]) for r in rows)["amplitude (baseline)"]
    slope = dict((r[0], r[1]) for r in rows)["slope only"]
    print(f"\n  slope-only vs amplitude-baseline: {slope - base:+.3f}  "
          f"({'slope alone matches/beats the triple' if slope >= base - 0.01 else 'triple beats slope alone'})")


if __name__ == "__main__":
    main()
