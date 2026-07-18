"""Leakage-free confirmation of the feature-search finding — compare a few FIXED fNIRS feature recipes under plain
subject-grouped CV. No search here, so no selection optimism: each recipe is a fixed set of descriptor
families → shrinkage-LDA, scored by repeated-seeded StratifiedGroupKFold (subject-grouped, no leakage). This
is the clean test of "does slope alone match the mean+slope+peak baseline?" that the Optuna/torch studies
suggested but couldn't claim (they maximise over a search; this doesn't). Each recipe's fold-mean acc/kappa
is recorded to the results snapshot, so the README table is marker-backed, not hand-typed.

    python -m neuroscan.tasks.workload.feature_importance.recipes
"""
from __future__ import annotations

import json
import logging
from typing import Any

import numpy as np
from jaxtyping import Float, Int
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from core.config import REPO
from core.data import store
from core.data.fnirs.base import FnirsCfg
from core.features import DescriptorBank
from neuroscan.evaluation import metrics, results
from neuroscan.tasks.cli import Cli
from neuroscan.tasks.workload.feature_importance._cv import Cv

logger = logging.getLogger(__name__)

# key -> (label, family list). `amplitude` = the current fnirs_lda triple; `full` = the whole bank. Keys are
# marker-/run-name-safe (they become results.json run names -> README markers).
_RECIPES = {
    "full":         ("full (15 families)",                   DescriptorBank.family_names()),
    "amplitude":    ("amplitude — mean+slope+peak (baseline)", ["mean", "slope", "peak"]),
    "slope_only":   ("slope only",                            ["slope"]),
    "dynamics":     ("dynamics (slope + early/late-slope)",   ["slope", "early_slope", "late_slope"]),
    "mean_only":    ("mean only",                             ["mean"]),
    "peak_only":    ("peak only",                             ["peak"]),
    "all_but_slope": ("all but slope",
                      [f for f in DescriptorBank.family_names() if f != "slope"]),
}
_SEEDS = [0, 1, 2]
_K = 5
_DATASET = "shin2017_nback"


class Recipes:
    """The fixed-recipe fNIRS confirmation helpers (free functions folded in as staticmethods, public names
    kept)."""

    @classmethod
    def _cv(cls, F: Float[np.ndarray, "n f"], fam: Any, y: Int[np.ndarray, "n"],
            groups: Int[np.ndarray, "n"], families: list[str]):
        Fr = F[:, np.isin(fam, families)]
        accs: list[float] = []
        kaps: list[float] = []
        for tr, te in Cv.grouped_folds(Fr, y, groups, _SEEDS, _K):
            clf = make_pipeline(StandardScaler(),
                                LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")).fit(Fr[tr], y[tr])
            pred = clf.predict(Fr[te])
            accs.append(metrics.Metrics.accuracy(y[te], pred))  # type: ignore[arg-type]
            kaps.append(metrics.Metrics.kappa(y[te], pred))  # type: ignore[arg-type]
        return float(np.mean(accs)), float(np.std(accs)), float(np.mean(kaps))

    @classmethod
    def _record(cls, key: str, acc: float, kappa: float, n_classes: int):
        """Write a harness-schema aggregate for this recipe and merge it into results.json (marker-backing)."""
        run = f"fnirs_recipe_{key}_{_DATASET}"
        run_dir = REPO / "runs" / run
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "aggregate.json").write_text(json.dumps({
            "method": f"fnirs_recipe_{key}", "regime": "cross_subject", "n_classes": n_classes,
            "fold_mean": {"acc": acc, "kappa": kappa, "ece": None},
        }, indent=2))
        return results.Results.record(run_dir)

    @classmethod
    def main(cls):
        Cli.setup_logging()
        meta = store.Store.load(_DATASET, FnirsCfg())  # type: ignore[arg-type]
        X, y = store.Store.gather(meta)
        groups = meta["subject"].to_numpy()
        F, fam = DescriptorBank.extract_bank(X)
        n_classes = int(y.max()) + 1
        chance = 1.0 / n_classes
        logger.info(f"fNIRS feature recipes · Shin n-back · {len(y)} blocks · {meta['subject'].n_unique()} subjects · "
              f"{len(_SEEDS)}x{_K}-fold GroupKFold · chance {chance:.3f}\n")
        logger.info(f"  {'recipe':<22}{'acc':>7} {'±sd':>6} {'kappa':>7}  #fam")
        accs = {}
        for key, (label, fams) in _RECIPES.items():
            acc, sd, kap = cls._cv(F, fam, y, groups, fams)
            accs[key] = acc
            cls._record(key, acc, kap, n_classes)
            logger.info(f"  {label:<22}{acc:>7.3f} {sd:>6.3f} {kap:>7.3f}  {len(fams)}")
        verdict = ("slope alone matches/beats the triple" if accs["slope_only"] >= accs["amplitude"] - 0.01
                   else "triple beats slope alone")
        logger.info(f"\n  slope-only vs amplitude-baseline: {accs['slope_only'] - accs['amplitude']:+.3f}  "
              f"({verdict})")
        logger.info("  recorded to results.json — run `sync_numbers` to push into the README")


if __name__ == "__main__":
    Recipes.main()
