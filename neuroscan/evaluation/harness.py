"""The single eval spine — every decoder (CSP+LDA, EEGNet, …) is a (fit_fn, score_fn) pair fed
through it, under a chosen evaluation regime. This is the project's contribution layer.

    fit_fn(X[n,ch,t], y[n]) -> clf
    score_fn(clf, X[n,ch,t]) -> probs[n, C]      # probabilities, so calibration (ECE) is measurable

A *fold* is (name, train_df, test_df). A *regime* turns the data cloud into a fold list:
    within          one fold per subject, tested in isolation        -> the ceiling
    cross_subject   leave-one-subject-out                            -> the OOD gap (headline)

`aggregate` is pure (folds in, metrics out — unit-testable). `run` calls it then logs to MLflow.
Reports BOTH fold-mean (per-subject equal weight) and pooled (per-epoch) — the measured pair.
"""
from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from joblib import Parallel, delayed, parallel_config

from core.data import splits, store
from neuroscan import tracking
from neuroscan.evaluation import diagnostics, metrics

logger = logging.getLogger(__name__)


@dataclass
class Method:
    """A decoder as the harness consumes it: a `name` + its `(fit, score)` pair + the class count and
    evaluation `regime`. These five always travel together — every fold is fit/scored the same way."""
    name: str
    fit: Callable[[np.ndarray, np.ndarray], Any]
    score: Callable[[Any, np.ndarray], np.ndarray]
    n_classes: int
    regime: str = ""


@dataclass
class TrackConfig:
    """MLflow logging + model-persistence options for a `run` (no effect on the computed metrics). `params` =
    the run params/tags; `run_dir` (a runs/<name>/ dir) enables resume + artifacts; `save_models` persists
    each fold's trained model."""
    params: dict[str, Any] | None = None
    run_dir: Path | str | None = None
    save_models: bool = True


class Harness:
    @staticmethod
    def folds_for(meta: Any, regime: str, test_sessions: tuple[Any, ...] = ()):
        """Build the (name, train, test) fold list for a regime over the epoch cloud `meta`."""
        out: list[tuple[Any, Any, Any]] = []
        if regime == "within":
            for s in sorted(meta["subject"].unique().to_list()):
                tr, _val, te = splits.Splits.within_subject(meta, s, test_sessions=test_sessions)
                out.append((s, tr, te))
        elif regime == "cross_subject":
            out.extend(splits.Splits.leave_one_subject_out(meta))
        elif regime == "cross_subject_kfold":
            out.extend(splits.Splits.grouped_kfold(meta, k=5))
        else:
            raise ValueError(f"unknown regime {regime!r} (want within / cross_subject / cross_subject_kfold)")
        return out

    @staticmethod
    def _fit_score_fold(
        fold: tuple[Any, Any, Any],
        fit_fn: Callable[[np.ndarray, np.ndarray], Any],
        score_fn: Callable[[Any, np.ndarray], np.ndarray],
    ):
        """One fold: gather -> fit -> score -> metrics. Returns (name, row, probs, yte, clf)."""
        name, train, test = fold
        Xtr, ytr = store.Store.gather(train)
        Xte, yte = store.Store.gather(test)
        clf = fit_fn(Xtr, ytr)
        probs = np.asarray(score_fn(clf, Xte), dtype=float)
        pred = probs.argmax(1)
        row = {"fold": str(name), "n": len(yte),
               "acc": metrics.Metrics.accuracy(yte, pred), "kappa": metrics.Metrics.kappa(yte, pred),
               "ece": metrics.Metrics.ece_from_probs(probs, yte)}
        return str(name), row, probs, yte, clf

    @staticmethod
    def aggregate(
        method: Method,
        folds: list[tuple[Any, Any, Any]],
        *,
        models_out: list[tuple[str, Any]] | None = None,
        n_jobs: int = 1,
        backend: str = "threading",
    ) -> dict[str, Any]:
        """Pure: run the method over folds, compute the metrics. No MLflow, no side effects.
        If `models_out` is given, each fold's fitted clf is appended as (fold_name, clf) for the caller to
        persist. `n_jobs`: parallelize the (independent) folds; `backend` picks how.

        Two backends (mindscape-07n — the "loky is a dead end here" verdict was a MISDIAGNOSIS; guarded loky
        returns cleanly, it just doesn't WIN at fold granularity on our methods — MEASURED, not assumed):
          - "threading" (default, keep it): shares memory, zero spawn cost. Our classical inner ops are BLAS/C
            (Ledoit-Wolf covariance, eig via mne/pyriemann/scipy) that release the GIL AND scale with BLAS
            threads, so threading already parallelises the heavy part. Measured best on FBCSP (bnci, 4-fold:
            86.8 s vs loky 121.4 s).
          - "loky" (process): guarded with `inner_max_num_threads=1` (pins BLAS/OMP to 1 thread/worker, which
            prevents the N-proc×N-thread oversubscription DEADLOCK that earlier looked like "workers never
            return"). Folds carry only lightweight polars frames (each gathers its own arrays) so pickling is
            cheap, BUT: the guard strips the per-fold BLAS threads (21.7→30.4 s/fold) AND Windows `spawn`
            re-imports the whole torch/mne stack per worker — so it LOSES here. Available for a future case
            with genuinely GIL-bound Python-level per-fold work; not a win for the current classical zoo. The
            real speedup remains coarse cell-level OS-process parallelism (`reproduce_all --only` chunks).
        Default n_jobs 1 (sequential); use -1 for CPU baselines, keep 1 for GPU nets (one device)."""
        if n_jobs == 1:
            done = [Harness._fit_score_fold(f, method.fit, method.score) for f in folds]
        elif backend == "loky":
            with parallel_config(backend="loky", inner_max_num_threads=1):
                done = Parallel(n_jobs=n_jobs)(
                    delayed(Harness._fit_score_fold)(f, method.fit, method.score) for f in folds)
        else:
            done = Parallel(n_jobs=n_jobs, backend=backend)(
                delayed(Harness._fit_score_fold)(f, method.fit, method.score) for f in folds)

        per: list[dict[str, Any]] = []
        P: list[np.ndarray] = []
        Y: list[np.ndarray] = []
        G: list[np.ndarray] = []
        for name, row, probs, yte, clf in done:                         # collected in fold order
            if models_out is not None:
                models_out.append((name, clf))
            per.append(row)
            logger.info(f"  {row['fold']:>6}  acc {row['acc']:.3f}  kappa {row['kappa']:.3f}  "
                        f"ece {row['ece']:.3f}  (n={row['n']})")
            P.append(probs)
            Y.append(yte)
            G.append(np.full(len(yte), name))

        probs, y, _ = np.concatenate(P), np.concatenate(Y), np.concatenate(G)
        pred = probs.argmax(1)
        fold_mean = {k: float(np.mean([r[k] for r in per])) for k in ("acc", "kappa", "ece")}
        pooled = {"acc": metrics.Metrics.accuracy(y, pred), "kappa": metrics.Metrics.kappa(y, pred),
                  "ece": metrics.Metrics.ece_from_probs(probs, y),
                  "confusion": metrics.Metrics.confusion(y, pred, method.n_classes).tolist()}
        sp = diagnostics.Diagnostics.spread(per, "acc")
        logger.info(f"  {'MEAN':>6}  acc {fold_mean['acc']:.3f}  kappa {fold_mean['kappa']:.3f}  "
              f"ece {fold_mean['ece']:.3f}   (spread {sp['min']:.3f}-{sp['max']:.3f}, std {sp['std']:.3f})")
        return {"method": method.name, "regime": method.regime, "n_classes": method.n_classes, "n_folds": len(per),
                "per_fold": per, "fold_mean": fold_mean, "pooled": pooled, "acc_spread": sp}

    @staticmethod
    def run(
        method: Method,
        folds: list[tuple[Any, Any, Any]],
        *,
        tracking_cfg: TrackConfig | None = None,
        n_jobs: int = 1,
        backend: str = "threading",
    ) -> dict[str, Any]:
        """aggregate + log to MLflow (guarded), configured by `tracking_cfg` (see TrackConfig). `n_jobs` +
        `backend` parallelize folds (see aggregate — "loky" for heavy GIL-bound methods, "threading" else)."""
        tc = tracking_cfg or TrackConfig()
        models: list[tuple[str, Any]] | None = [] if tc.save_models else None
        res = Harness.aggregate(method, folds, models_out=models, n_jobs=n_jobs, backend=backend)
        fm, pooled = res["fold_mean"], res["pooled"]
        tags = {"method": method.name, "regime": method.regime, "dataset": (tc.params or {}).get("dataset", "")}
        with tracking.Tracking.run("mindscape", f"{method.name}_{method.regime}",
                          params=tc.params or {"method": method.name, "regime": method.regime},
                          tags=tags, run_dir=tc.run_dir):
            tracking.Tracking.metrics({"acc_mean": fm["acc"], "kappa_mean": fm["kappa"], "ece_mean": fm["ece"],
                              "acc_pooled": pooled["acc"], "kappa_pooled": pooled["kappa"],
                              "ece_pooled": pooled["ece"], "acc_std": res["acc_spread"]["std"],
                              "acc_min": res["acc_spread"]["min"], "acc_max": res["acc_spread"]["max"]})
            tracking.Tracking.per_group("acc_subject", {r["fold"]: r["acc"] for r in res["per_fold"]})
            tracking.Tracking.per_group("ece_subject", {r["fold"]: r["ece"] for r in res["per_fold"]})
            tracking.Tracking.artifact_json("aggregate.json", res)
            for fold_name, clf in (models or []):
                tracking.Tracking.save_model(clf, f"model_{method.name}_{fold_name}", run_dir=tc.run_dir)
        return res
