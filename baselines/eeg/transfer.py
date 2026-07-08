"""Cross-subject Riemannian transfer — the Riemannian Procrustes Analysis ladder as reusable functions.

Plain tangent-space + LR transfers badly across subjects: each subject's covariance cloud sits at a
different LOCATION on the SPD manifold (a domain shift, not a difference in the shared contrast). RPA
(Rodrigues 2019) aligns the domains in up to three steps; these functions are the *method* — the alignment +
classifier — with no experiment scaffolding. The runner (`tasks/motor_imagery/align.py`) owns the folds, the
covariance estimation, the calibration split, and the metrics; it just calls these.

  zero-shot  (no target labels — deployment-real): `zero_shot_predict(..., scale=)`
             re-center each domain to the identity (± re-scale dispersion), then tangent-space + LR.
  calibrated (a few labelled target trials, DISJOINT from test): `calibrated_predict(kind=)`
             full RPA (center+scale+rotate) or MDWM — the rotation/MDWM are supervised on the calib slice.
"""
from __future__ import annotations

import numpy as np
from pyriemann.tangentspace import TangentSpace, tangent_space
from pyriemann.transfer import (
    MDWM,
    TLCenter,
    TLClassifier,
    TLRotate,
    TLScale,
    encode_domains,
)
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

from core.features import recenter_covariances, scale_to_identity


def _tangent_lr():
    return make_pipeline(TangentSpace(metric="riemann"), LogisticRegression(max_iter=500, C=1.0))


def align_domains(Csrc, groups, Cte, *, scale: bool, target_groups=None):
    """Re-center each source subject and the target to the identity independently (± dispersion re-scale) —
    the unsupervised, per-domain transforms. Re-centering is PER DOMAIN, so if the target holds more than one
    subject (e.g. a k-fold test fold) pass `target_groups` to re-center each on its OWN mean — pooling them
    would align to a shared blob and undo the fix. `target_groups=None` treats the target as one domain (the
    LOSO case: a single held-out subject). Returns (source-aligned, target-aligned)."""
    def _align(C):
        rc = recenter_covariances(C)
        return scale_to_identity(rc) if scale else rc

    def _by_group(C, g):
        out = np.empty_like(C)
        for k in np.unique(g):
            out[g == k] = _align(C[g == k])
        return out

    Cs = _by_group(Csrc, groups)
    Ct = _align(Cte) if target_groups is None else _by_group(Cte, target_groups)
    return Cs, Ct


def recentered_tangent_features(C, groups) -> np.ndarray:
    """Re-center each subject's covariances to the identity (per-domain), then map to the tangent space at
    the identity -> a Euclidean feature vector `[n, d(d+1)/2]`. The feature-space view of the strong EEG
    decoder — for feature-level fusion, so its EEG side matches the re-centered covariance the probs side
    uses (not a crude log-variance). `groups` = subject per row (re-centering is per-domain)."""
    rc = np.empty_like(C)
    for g in np.unique(groups):
        rc[groups == g] = recenter_covariances(C[groups == g])
    return tangent_space(rc, np.eye(C.shape[-1]))           # tangent at I: the covariances are centred there


def zero_shot_predict(Csrc, ysrc, groups, Cte, *, scale: bool, target_groups=None) -> np.ndarray:
    """Zero-shot transfer: align source (per subject) + target (per subject if `target_groups` given, else as
    one domain), tangent-space + LR, return class probabilities `[n, C]` for ALL target trials (no labels)."""
    Cs, Ct = align_domains(Csrc, groups, Cte, scale=scale, target_groups=target_groups)
    clf = _tangent_lr().fit(Cs, ysrc)
    return np.asarray(clf.predict_proba(Ct), dtype=float)


def calibrated_predict(kind: str, Csrc, ysrc, Ccal, ycal, Cev, mdwm_lambda: float = 0.5) -> np.ndarray:
    """Calibrated transfer: fit on source + a labelled target CALIBRATION slice, predict the disjoint target
    eval set. `kind='rpa'` = full RPA (center+scale+rotate) then tangent-space LR; `kind='mdwm'` = Minimum
    Distance to Weighted Mean (source↔target class-mean blend, weight `mdwm_lambda`). Returns int labels for
    `Cev`. The caller guarantees Ccal and Cev are disjoint — no test labels enter here."""
    Xf = np.concatenate([Csrc, Ccal])
    yf = np.concatenate([ysrc, ycal]).astype(str)
    dom = np.array(["source"] * len(ysrc) + ["target"] * len(ycal))
    Xenc, yenc = encode_domains(Xf, yf, dom)
    Xev, _ = encode_domains(Cev, np.zeros(len(Cev), int).astype(str), np.array(["target"] * len(Cev)))

    if kind == "mdwm":
        model = MDWM(domain_tradeoff=mdwm_lambda, target_domain="target")
    else:                                                            # full RPA + tangent-space LR
        model = make_pipeline(TLCenter("target"), TLScale("target", centered_data=True),
                              TLRotate("target"), TLClassifier("target", _tangent_lr()))
    model.fit(Xenc, yenc)
    return model.predict(Xev).astype(int)
