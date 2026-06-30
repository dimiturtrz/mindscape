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
        for s, tr, _val, te in splits.leave_one_subject_out(meta):
            out.append((s, tr, te))
    else:
        raise ValueError(f"unknown regime {regime!r} (want 'within' or 'cross_subject')")
    return out


def aggregate(method: str, fit_fn, score_fn, folds, n_classes: int, regime: str = "") -> dict:
    """Pure: run the method over folds, compute the metrics. No MLflow, no side effects."""
    per, P, Y, G = [], [], [], []
    for name, train, test in folds:
        Xtr, ytr = store.gather(train)
        Xte, yte = store.gather(test)
        probs = np.asarray(score_fn(fit_fn(Xtr, ytr), Xte), dtype=float)
        pred = probs.argmax(1)
        row = {"fold": str(name), "n": int(len(yte)),
               "acc": metrics.accuracy(yte, pred), "kappa": metrics.kappa(yte, pred),
               "ece": metrics.ece_from_probs(probs, yte)}
        per.append(row)
        print(f"  {row['fold']:>6}  acc {row['acc']:.3f}  kappa {row['kappa']:.3f}  ece {row['ece']:.3f}  (n={row['n']})")
        P.append(probs); Y.append(yte); G.append(np.full(len(yte), str(name)))

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
        params: dict | None = None, run_dir=None) -> dict:
    """aggregate + log to MLflow (guarded). `run_dir` (a runs/<name>/ dir) enables resume + artifacts."""
    res = aggregate(method, fit_fn, score_fn, folds, n_classes, regime)
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
    return res
