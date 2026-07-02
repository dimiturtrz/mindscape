"""The single eval spine — every decoder (CSP+LDA, EEGNet, …) is a (fit_fn, score_fn) pair fed
through it, under a chosen evaluation regime. This is the project's contribution layer.

    fit_fn(X[n,ch,t], y[n]) -> clf
    score_fn(clf, X[n,ch,t]) -> probs[n, C]      # probabilities, so calibration (ECE) is measurable

A *fold* is (name, train_df, test_df). A *regime* turns the data cloud into a fold list:
    within          one fold per subject, tested in isolation        -> the ceiling
    cross_subject   leave-one-subject-out                            -> the OOD gap (headline)

`aggregate` is pure (folds in, metrics out — unit-testable). `run` calls it then logs to MLflow.
Reports BOTH fold-mean (per-subject equal weight) and pooled (per-epoch) — the honest pair.
"""
from __future__ import annotations

import numpy as np

from core.data import splits, store
from neuroscan import tracking
from neuroscan.evaluation import diagnostics, metrics


def folds_for(meta, regime: str, test_sessions=()):
    """Build the (name, train, test) fold list for a regime over the epoch cloud `meta`."""
    out = []
    if regime == "within":
        for s in sorted(meta["subject"].unique().to_list()):
            tr, _val, te = splits.within_subject(meta, s, test_sessions=test_sessions)
            out.append((s, tr, te))
    elif regime == "cross_subject":
        out.extend(splits.leave_one_subject_out(meta))
    elif regime == "cross_subject_kfold":
        out.extend(splits.grouped_kfold(meta, k=5))
    else:
        raise ValueError(f"unknown regime {regime!r} (want within / cross_subject / cross_subject_kfold)")
    return out


def _fit_score_fold(fold, fit_fn, score_fn):
    """One fold: gather -> fit -> score -> metrics. Returns (name, row, probs, yte, clf)."""
    name, train, test = fold
    Xtr, ytr = store.gather(train)
    Xte, yte = store.gather(test)
    clf = fit_fn(Xtr, ytr)
    probs = np.asarray(score_fn(clf, Xte), dtype=float)
    pred = probs.argmax(1)
    row = {"fold": str(name), "n": int(len(yte)),
           "acc": metrics.accuracy(yte, pred), "kappa": metrics.kappa(yte, pred),
           "ece": metrics.ece_from_probs(probs, yte)}
    return str(name), row, probs, yte, clf


def aggregate(method: str, fit_fn, score_fn, folds, n_classes: int, regime: str = "",
              models_out: list | None = None, n_jobs: int = 1) -> dict:
    """Pure: run the method over folds, compute the metrics. No MLflow, no side effects.
    If `models_out` is given, each fold's fitted clf is appended as (fold_name, clf) for the caller to
    persist. `n_jobs`: parallelize the (independent) folds with the **threading** backend. NOTE the tradeoff:
    threading shares memory but can't beat the GIL, so GIL-bound classical code (FBCSP/CSP/ACM via mne/sklearn)
    effectively serialises here (measured: a 26-fold FBCSP cell ran ~= 26× one fold). The obvious fix —
    process parallelism (loky) — was tried and FAILS in this environment: Windows spawns (not forks) each
    worker, which re-imports the heavy native stack (mne/scipy/mkl) and then computes but never returns /
    respawns → 0 folds completed in 200-400 s. So threading stays; real speedups come from *less per-fold
    compute* (downsample recipes, crop windows — mindscape-241) and *coarse* cell-level process parallelism
    (independent `reproduce_all --only` chunks), NOT fine-grained fold loky. See mindscape-07n.
    Default 1 (sequential); use -1 for CPU baselines, keep 1 for GPU nets (one device)."""
    if n_jobs == 1:
        done = [_fit_score_fold(f, fit_fn, score_fn) for f in folds]
    else:
        from joblib import Parallel, delayed
        done = Parallel(n_jobs=n_jobs, backend="threading")(
            delayed(_fit_score_fold)(f, fit_fn, score_fn) for f in folds)

    per, P, Y, G = [], [], [], []
    for name, row, probs, yte, clf in done:                         # collected in fold order
        if models_out is not None:
            models_out.append((name, clf))
        per.append(row)
        print(f"  {row['fold']:>6}  acc {row['acc']:.3f}  kappa {row['kappa']:.3f}  ece {row['ece']:.3f}  (n={row['n']})")
        P.append(probs); Y.append(yte); G.append(np.full(len(yte), name))

    probs, y, g = np.concatenate(P), np.concatenate(Y), np.concatenate(G)
    pred = probs.argmax(1)
    fold_mean = {k: float(np.mean([r[k] for r in per])) for k in ("acc", "kappa", "ece")}
    pooled = {"acc": metrics.accuracy(y, pred), "kappa": metrics.kappa(y, pred),
              "ece": metrics.ece_from_probs(probs, y),
              "confusion": metrics.confusion(y, pred, n_classes).tolist()}
    sp = diagnostics.spread(per, "acc")
    print(f"  {'MEAN':>6}  acc {fold_mean['acc']:.3f}  kappa {fold_mean['kappa']:.3f}  "
          f"ece {fold_mean['ece']:.3f}   (spread {sp['min']:.3f}-{sp['max']:.3f}, std {sp['std']:.3f})")
    return {"method": method, "regime": regime, "n_classes": n_classes, "n_folds": len(per),
            "per_fold": per, "fold_mean": fold_mean, "pooled": pooled, "acc_spread": sp}


def run(method: str, fit_fn, score_fn, folds, n_classes: int, regime: str = "",
        params: dict | None = None, run_dir=None, save_models: bool = True, n_jobs: int = 1) -> dict:
    """aggregate + log to MLflow (guarded). `run_dir` (a runs/<name>/ dir) enables resume + artifacts.
    `save_models` persists each fold's trained model. `n_jobs` parallelizes folds (see aggregate)."""
    models: list = [] if save_models else None
    res = aggregate(method, fit_fn, score_fn, folds, n_classes, regime, models_out=models, n_jobs=n_jobs)
    fm, pooled = res["fold_mean"], res["pooled"]
    tags = {"method": method, "regime": regime, "dataset": (params or {}).get("dataset", "")}
    with tracking.run("mindscape", f"{method}_{regime}", params=params or {"method": method, "regime": regime},
                      tags=tags, run_dir=run_dir):
        tracking.metrics({"acc_mean": fm["acc"], "kappa_mean": fm["kappa"], "ece_mean": fm["ece"],
                          "acc_pooled": pooled["acc"], "kappa_pooled": pooled["kappa"],
                          "ece_pooled": pooled["ece"], "acc_std": res["acc_spread"]["std"],
                          "acc_min": res["acc_spread"]["min"], "acc_max": res["acc_spread"]["max"]})
        tracking.per_group("acc_subject", {r["fold"]: r["acc"] for r in res["per_fold"]})
        tracking.per_group("ece_subject", {r["fold"]: r["ece"] for r in res["per_fold"]})
        tracking.artifact_json("aggregate.json", res)
        for fold_name, clf in (models or []):
            tracking.save_model(clf, f"model_{method}_{fold_name}", run_dir=run_dir)
    return res
