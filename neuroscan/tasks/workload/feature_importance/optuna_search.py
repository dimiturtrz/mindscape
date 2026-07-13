"""fNIRS weighted-feature importance study — Optuna as a wrapper feature-selector.

Each trial picks a weight in [0,1] per descriptor family (post-standardisation; see WeightedFamilyScaler),
scored by mean accuracy over repeated seeded StratifiedGroupKFold (grouped by subject → no leakage). The
deliverable is the feature IMPORTANCE, read two ways and checked for stability across study reruns:
  - fANOVA importance      (optuna.importance.get_param_importances)
  - top-trial weight means (what the best trials consistently up-weight)
  - cross-seed stability   (rerun the whole study under each tpe_seed; do the top families hold?)
The per-study peak accuracy is optimistic (max over a search) and is reported as such, NOT as a
generalisation estimate — that would need a sealed outer fold (nested CV). See mindscape-b9g.

Every study is persisted to an Optuna **JournalStorage** DB under `runs/optuna_fnirs/` (one study per TPE
seed), so the trials are reproducible and queryable after the fact — and the written `importance.json` is the
artifact the README importance table is regenerated from (reproducible from the stored trials, not hand-typed).

    python -m neuroscan.tasks.workload.feature_importance.optuna_search              # uses optuna.yaml (local)
    python -m neuroscan.tasks.workload.feature_importance.optuna_search --trials 400 # sparse override
"""
from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import optuna
from omegaconf import OmegaConf
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from core.config import REPO
from core.data import store
from core.data.fnirs.base import FnirsCfg
from core.features import DescriptorBank, WeightedFamilyScaler
from neuroscan.tasks.cli import Cli
from neuroscan.tasks.workload.feature_importance._cv import Cv

logger = logging.getLogger(__name__)

_CFG = Path(__file__).with_name("optuna.yaml")            # study config lives beside the code (config-as-data)
_STABLE_JACCARD = 0.6   # top-family sets agree across seeds at/above this Jaccard -> trust the ranking


@dataclass
class Bank:
    """The fNIRS feature bank the search scores over: the `[n, cols]` features, each column's family label
    `fam`, the class labels `y`, and the per-block subject `groups`. These four travel together everywhere."""
    F: np.ndarray
    fam: np.ndarray
    y: np.ndarray
    groups: np.ndarray


class OptunaSearch:
    """The Optuna wrapper-feature-selector helpers (free functions folded in as staticmethods, public names
    kept)."""

    @staticmethod
    def _cv_score(bank: Bank, weights, fold_seeds, k) -> float:
        """Mean accuracy of standardise→per-family-weight→shrinkage-LDA over StratifiedGroupKFold repeated for
        each seed in `fold_seeds`. Grouped by subject (whole subjects per fold); the scaler fits on train only."""
        accs = []
        for tr, te in Cv.grouped_folds(bank.F, bank.y, bank.groups, fold_seeds, k):
            sc = WeightedFamilyScaler(bank.fam, weights).fit(bank.F[tr])
            # fixed shrinkage (not "auto"/Ledoit-Wolf): ~10x cheaper on 1080 features and constant across trials,
            # so it doesn't confound the relative feature-weight comparison the search is after.
            lda = LinearDiscriminantAnalysis(solver="lsqr", shrinkage=0.4).fit(sc.transform(bank.F[tr]), bank.y[tr])
            accs.append(float((lda.predict(sc.transform(bank.F[te])) == bank.y[te]).mean()))
        return float(np.mean(accs))

    @staticmethod
    def _storage(out: Path):
        """Optuna JournalStorage (file-backed) under the run dir — persists all trials for later querying. Uses
        the open()-based file lock, not the default symlink lock, so it works on Windows (symlink creation needs
        a privilege the client doesn't hold there)."""
        out.mkdir(parents=True, exist_ok=True)
        path = str(out / "journal.log")
        backend = optuna.storages.journal.JournalFileBackend(path,
                                                             lock_obj=optuna.storages.journal.JournalFileOpenLock(path))
        return optuna.storages.JournalStorage(backend)

    @staticmethod
    def _run_one_study(bank: Bank, families, cfg, tpe_seed, storage):
        """One Optuna study (one TPE seed): returns (importances, top_weight_means, best_value, best_params)."""
        def objective(trial):
            weights = {f: trial.suggest_float(f, cfg.weight_low, cfg.weight_high) for f in families}
            return OptunaSearch._cv_score(bank, weights, list(cfg.fold_seeds), cfg.k)

        study = optuna.create_study(direction="maximize", storage=storage, load_if_exists=True,
                                    study_name=f"fnirs_importance_seed{tpe_seed}",
                                    sampler=optuna.samplers.TPESampler(seed=tpe_seed))
        # resume: only run the missing trials. The importance/stability deliverable reads the whole trial cloud,
        # so resuming (vs one shot) doesn't change the conclusion; delete runs/optuna_fnirs to force a clean study.
        if len(study.trials) < cfg.n_trials:
            study.optimize(objective, n_trials=cfg.n_trials - len(study.trials), show_progress_bar=False)

        try:
            importances = optuna.importance.get_param_importances(study)      # family -> fANOVA importance
        except (RuntimeError, ValueError):                                    # fANOVA needs enough trials to fit
            importances = {f: 0.0 for f in families}                         # (top-trial weights still hold below)
        n_top = max(1, int(cfg.top_frac * cfg.n_trials))
        top = sorted(study.trials, key=lambda t: t.value, reverse=True)[:n_top]
        top_w = {f: float(np.mean([t.params[f] for t in top])) for f in families}   # mean weight in best trials
        return importances, top_w, float(study.best_value), dict(study.best_params)

    @staticmethod
    def _stability(per_seed_importances, families, topn=5):
        """How much do the top-`topn` important families agree across the study reruns? Jaccard overlap of the
        top sets — high = a stable, trustworthy importance ranking; low = the 'importance' is search noise."""
        top_sets = [set(sorted(imp, key=imp.get, reverse=True)[:topn]) for imp in per_seed_importances]
        pairs = [(a, b) for i, a in enumerate(top_sets) for b in top_sets[i + 1:]]
        jac = [len(a & b) / len(a | b) for a, b in pairs] if pairs else [1.0]
        consensus = sorted(families, key=lambda f: np.mean([imp.get(f, 0.0) for imp in per_seed_importances]),
                           reverse=True)
        return {"topn": topn, "mean_jaccard": float(np.mean(jac)), "consensus_order": consensus}


def main():
    Cli.setup_logging()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=None, help="study config (default: optuna.yaml beside this module)")
    ap.add_argument("--trials", type=int, default=None, help="override n_trials (the one common knob)")
    args = ap.parse_args()

    optuna.logging.set_verbosity(optuna.logging.WARNING)                  # no per-trial spam

    cfg = OmegaConf.load(args.config or _CFG)
    if args.trials:
        cfg.n_trials = args.trials

    meta = store.Store.load(cfg.dataset, FnirsCfg())
    X, y = store.Store.gather(meta)
    groups = meta["subject"].to_numpy()
    F, fam = DescriptorBank.extract_bank(X)
    bank = Bank(F, fam, y, groups)
    families = DescriptorBank.family_names()
    out = REPO / cfg.out
    storage = OptunaSearch._storage(out)
    logger.info(f"fNIRS bank: {F.shape[0]} blocks · {meta['subject'].n_unique()} subjects · {len(families)} families "
          f"· {F.shape[1]} cols · {cfg.n_trials} trials × {len(cfg.tpe_seeds)} seeds "
          f"(chance {1/len(set(y.tolist())):.3f}) · journal {out}/journal.log")

    per_seed_imp, per_seed_topw, peaks, bests = [], [], [], []
    for s in cfg.tpe_seeds:
        imp, topw, best, bparams = OptunaSearch._run_one_study(bank, families, cfg, int(s), storage)
        per_seed_imp.append(imp)
        per_seed_topw.append(topw)
        peaks.append(best)
        bests.append(bparams)
        top3 = sorted(imp, key=imp.get, reverse=True)[:3]
        logger.info(f"  seed {s}: peak-acc {best:.3f} (optimistic) · top-3 by importance {top3}")

    stab = OptunaSearch._stability(per_seed_imp, families)
    cons_imp = {f: float(np.mean([imp.get(f, 0.0) for imp in per_seed_imp])) for f in families}
    cons_topw = {f: float(np.mean([tw[f] for tw in per_seed_topw])) for f in families}
    topw_sd = {f: float(np.std([tw[f] for tw in per_seed_topw])) for f in families}   # cross-seed spread

    logger.info(f"\n=== importance (consensus over {len(cfg.tpe_seeds)} seeds) ===")
    for f in sorted(families, key=lambda f: cons_imp[f], reverse=True):
        logger.info(f"  {f:<14} importance {cons_imp[f]:.3f}  ·  mean top-trial weight {cons_topw[f]:.2f} "
              f"(±{topw_sd[f]:.2f})")
    logger.info(f"\nstability: top-{stab['topn']} families agree across seeds at Jaccard {stab['mean_jaccard']:.2f} "
          f"({'STABLE — trust the ranking' if stab['mean_jaccard'] >= _STABLE_JACCARD else 'UNSTABLE — importance is search noise'})")
    logger.info(f"peak-acc range {min(peaks):.3f}-{max(peaks):.3f} (optimistic; not a generalisation estimate)")

    (out / "importance.json").write_text(json.dumps({
        "dataset": str(cfg.dataset), "n_trials": int(cfg.n_trials), "tpe_seeds": list(cfg.tpe_seeds),
        "consensus_importance": cons_imp, "consensus_top_weight": cons_topw, "top_weight_sd": topw_sd,
        "stability": stab, "peak_acc_per_seed": peaks, "best_params_per_seed": bests,
    }, indent=2))
    logger.info(f"-> {out}/importance.json (reproducible artifact — the README importance table is generated from this)")


if __name__ == "__main__":
    main()
