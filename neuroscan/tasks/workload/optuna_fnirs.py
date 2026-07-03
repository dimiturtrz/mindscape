"""fNIRS weighted-feature importance study — Optuna as a wrapper feature-selector.

Each trial picks a weight in [0,1] per descriptor family (post-standardisation; see WeightedFamilyScaler),
scored by mean accuracy over repeated seeded StratifiedGroupKFold (grouped by subject → no leakage). The
deliverable is the feature IMPORTANCE, read two ways and checked for stability across study reruns:
  - fANOVA importance      (optuna.importance.get_param_importances)
  - top-trial weight means (what the best trials consistently up-weight)
  - cross-seed stability   (rerun the whole study under each tpe_seed; do the top families hold?)
The per-study peak accuracy is optimistic (max over a search) and is reported as such, NOT as a
generalisation estimate — that would need a sealed outer fold (nested CV). See mindscape-b9g.

    python -m neuroscan.tasks.workload.optuna_fnirs                 # uses optuna_fnirs.yaml
    python -m neuroscan.tasks.workload.optuna_fnirs --trials 400    # sparse override
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from core.config import REPO
from core.data import store
from core.data.fnirs.base import FnirsCfg
from core.features import extract_bank, family_names, WeightedFamilyScaler


def _cv_score(F, fam, y, groups, weights, fold_seeds, k) -> float:
    """Mean accuracy of standardise→per-family-weight→shrinkage-LDA over StratifiedGroupKFold repeated for
    each seed in `fold_seeds`. Grouped by subject (whole subjects per fold); the scaler fits on train only."""
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.model_selection import StratifiedGroupKFold

    accs = []
    for seed in fold_seeds:
        sgkf = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=seed)
        for tr, te in sgkf.split(F, y, groups):
            sc = WeightedFamilyScaler(fam, weights).fit(F[tr])
            # fixed shrinkage (not "auto"/Ledoit-Wolf): ~10x cheaper on 1080 features and constant across
            # trials, so it doesn't confound the relative feature-weight comparison the search is after.
            lda = LinearDiscriminantAnalysis(solver="lsqr", shrinkage=0.4).fit(sc.transform(F[tr]), y[tr])
            accs.append(float((lda.predict(sc.transform(F[te])) == y[te]).mean()))
    return float(np.mean(accs))


def _run_one_study(F, fam, y, groups, families, cfg, tpe_seed):
    """One Optuna study (one TPE seed): returns (importances, top_weight_means, best_value, best_params)."""
    import optuna

    def objective(trial):
        weights = {f: trial.suggest_float(f, cfg.weight_low, cfg.weight_high) for f in families}
        return _cv_score(F, fam, y, groups, weights, list(cfg.fold_seeds), cfg.k)

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=tpe_seed))
    study.optimize(objective, n_trials=cfg.n_trials, show_progress_bar=False)

    try:
        importances = optuna.importance.get_param_importances(study)      # family -> fANOVA importance
    except Exception:                                                     # fANOVA needs enough trials to fit
        importances = {f: 0.0 for f in families}                         # (top-trial weights still hold below)
    n_top = max(1, int(cfg.top_frac * cfg.n_trials))
    top = sorted(study.trials, key=lambda t: t.value, reverse=True)[:n_top]
    top_w = {f: float(np.mean([t.params[f] for t in top])) for f in families}   # mean weight in best trials
    return importances, top_w, float(study.best_value), dict(study.best_params)


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
    from omegaconf import OmegaConf

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="optuna_fnirs.yaml", help="study config (config-as-data)")
    ap.add_argument("--trials", type=int, default=None, help="override n_trials (the one common knob)")
    args = ap.parse_args()

    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)                  # no per-trial spam

    cfg = OmegaConf.load(REPO / args.config)
    if args.trials:
        cfg.n_trials = args.trials

    meta = store.load(cfg.dataset, FnirsCfg())
    X, y = store.gather(meta)
    groups = meta["subject"].to_numpy()
    F, fam = extract_bank(X)
    families = family_names()
    print(f"fNIRS bank: {F.shape[0]} blocks · {meta['subject'].n_unique()} subjects · {len(families)} families "
          f"· {F.shape[1]} cols · {cfg.n_trials} trials × {len(cfg.tpe_seeds)} seeds "
          f"(chance {1/len(set(y.tolist())):.3f})")

    per_seed_imp, per_seed_topw, peaks, bests = [], [], [], []
    for s in cfg.tpe_seeds:
        imp, topw, best, bparams = _run_one_study(F, fam, y, groups, families, cfg, int(s))
        per_seed_imp.append(imp); per_seed_topw.append(topw); peaks.append(best); bests.append(bparams)
        top3 = sorted(imp, key=imp.get, reverse=True)[:3]
        print(f"  seed {s}: peak-acc {best:.3f} (optimistic) · top-3 by importance {top3}")

    stab = _stability(per_seed_imp, families)
    cons_imp = {f: float(np.mean([imp.get(f, 0.0) for imp in per_seed_imp])) for f in families}
    cons_topw = {f: float(np.mean([tw[f] for tw in per_seed_topw])) for f in families}

    print(f"\n=== importance (consensus over {len(cfg.tpe_seeds)} seeds) ===")
    for f in sorted(families, key=lambda f: cons_imp[f], reverse=True):
        print(f"  {f:<14} importance {cons_imp[f]:.3f}  ·  mean top-trial weight {cons_topw[f]:.2f}")
    print(f"\nstability: top-{stab['topn']} families agree across seeds at Jaccard {stab['mean_jaccard']:.2f} "
          f"({'STABLE — trust the ranking' if stab['mean_jaccard'] >= 0.6 else 'UNSTABLE — importance is search noise'})")
    print(f"peak-acc range {min(peaks):.3f}-{max(peaks):.3f} (optimistic; not a generalisation estimate)")

    out = Path(cfg.out); out.mkdir(parents=True, exist_ok=True)
    (out / "importance.json").write_text(json.dumps({
        "dataset": str(cfg.dataset), "n_trials": int(cfg.n_trials), "tpe_seeds": list(cfg.tpe_seeds),
        "consensus_importance": cons_imp, "consensus_top_weight": cons_topw,
        "stability": stab, "peak_acc_per_seed": peaks, "best_params_per_seed": bests,
    }, indent=2))
    print(f"-> {out}/importance.json")


if __name__ == "__main__":
    main()
