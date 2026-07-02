"""EEG↔fNIRS fusion combiners — the output-space aggregators + feature fusion, and the diagnostics that
frame the result (complementarity, the aggregation sweep). Operate on the two modalities' per-block
probabilities (or features); the runner (`tasks/workload/run_fusion.py`) owns the fold loop, the aligned
gather, and the recording, and calls these. See the README fusion section for what they show (a rigorous
null: complementarity is real, no output-space combiner cashes it — confidence doesn't track correctness)."""
from __future__ import annotations

import numpy as np


def _log_bandpower(X: np.ndarray) -> np.ndarray:
    """Per-channel log variance = the cheap broadband EEG feature used on the fusion feature side."""
    return np.log(X.var(axis=2) + 1e-12)


def feature_fusion(Xe_tr, Xf_tr, y_tr, Xe_te, Xf_te) -> np.ndarray:
    """Feature-level fusion: concat EEG log-bandpower + fNIRS mean/slope/peak -> shrinkage-LDA. Test probs."""
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    from core.features import amplitude_features
    ftr = np.concatenate([_log_bandpower(Xe_tr), amplitude_features(Xf_tr)], axis=1)
    fte = np.concatenate([_log_bandpower(Xe_te), amplitude_features(Xf_te)], axis=1)
    clf = make_pipeline(StandardScaler(),
                        LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")).fit(ftr, y_tr)
    return clf.predict_proba(fte)


def smart_aggregators(eeg_fit, eeg_score, fn_fit, fn_score, Xe_tr, Xf_tr, y_tr, g_tr, pe, pf):
    """The learned/calibrated output-space combiners, fit WITHOUT touching test data: an inner GroupKFold(3)
    over the train subjects yields out-of-fold base probs, on which we fit (a) a stacking meta-LDA over the
    concatenated [eeg, fnirs] probs and (b) a per-modality temperature. Returns their test-set outputs
    (stacking probs, calibrated eeg probs, calibrated fnirs probs); falls back to the raw probs if a train
    subject group is too small to inner-split."""
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.model_selection import GroupKFold

    from neuroscan.evaluation.calibrate import TemperatureScaler

    n, C = pe.shape
    oof_e, oof_f = np.zeros((n, C)), np.zeros((n, C))
    try:
        for itr, iva in GroupKFold(n_splits=3).split(np.arange(n), groups=g_tr):
            oof_e[iva] = eeg_score(eeg_fit(Xe_tr[itr], y_tr[itr]), Xe_tr[iva])
            oof_f[iva] = fn_score(fn_fit(Xf_tr[itr], y_tr[itr]), Xf_tr[iva])
    except ValueError:                                       # too few groups for an inner split
        return (pe + pf) / 2.0, pe, pf

    def _logit(p):
        return np.log(p + 1e-12)

    def _softmax(z):
        z = z - z.max(1, keepdims=True)
        e = np.exp(z)
        return e / e.sum(1, keepdims=True)

    meta = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto").fit(
        np.concatenate([oof_e, oof_f], axis=1), y_tr)
    stk = meta.predict_proba(np.concatenate([pe, pf], axis=1))
    Te = TemperatureScaler().fit(_logit(oof_e), y_tr).T
    Tf = TemperatureScaler().fit(_logit(oof_f), y_tr).T
    return stk, _softmax(_logit(pe) / Te), _softmax(_logit(pf) / Tf)


def complementarity(mean: dict, ce: np.ndarray, cf: np.ndarray) -> dict:
    """Is fusion fundamentally hopeless, or just naive averaging? The oracle (either modality correct) is the
    upper bound ANY fusion could reach; near-zero error correlation means the two modalities fail on
    independent blocks, so a per-trial selector has headroom the mean can't touch. `ce`/`cf` = per-block
    correct masks (pooled over folds)."""
    return {
        "best_single": max(mean["eeg"], mean["fnirs"]),
        "oracle_either": float((ce | cf).mean()),
        "both_correct": float((ce & cf).mean()),
        "eeg_only": float((ce & ~cf).mean()), "fnirs_only": float((~ce & cf).mean()),
        "both_wrong": float((~ce & ~cf).mean()),
        "err_corr": float(np.corrcoef(ce.astype(float), cf.astype(float))[0, 1]),
    }


SWEEP_KEYS = ("mean", "product", "conf_weight", "maxconf_pick", "stacking", "cal_mean", "cal_conf_weight")


def aggregation_sweep(Pe, Pf, Stk, Pce, Pcf, y, ce, cf) -> dict:
    """Can any OUTPUT-space combiner cash the oracle headroom? Try every cheap→learned aggregator on the
    pooled probs. They all fail here because confidence does not track correctness (see the conf_gap) — the
    reliability signal a selector needs is not in the probabilities at all."""
    acc = lambda P: float((P.argmax(1) == y).mean())
    we, wf = Pe.max(1, keepdims=True), Pf.max(1, keepdims=True)
    return {
        "mean": acc((Pe + Pf) / 2), "product": acc(Pe * Pf),
        "conf_weight": acc(we * Pe + wf * Pf),
        "maxconf_pick": float((np.where(we >= wf, Pe.argmax(1, keepdims=True),
                                        Pf.argmax(1, keepdims=True)).ravel() == y).mean()),
        "stacking": acc(Stk),
        "cal_mean": acc((Pce + Pcf) / 2),
        "cal_conf_weight": acc(Pce.max(1, keepdims=True) * Pce + Pcf.max(1, keepdims=True) * Pcf),
        # does confidence predict correctness? gap = mean max-prob(correct) - max-prob(wrong), per modality
        "eeg_conf_gap": float(Pe.max(1)[ce].mean() - Pe.max(1)[~ce].mean()),
        "fnirs_conf_gap": float(Pf.max(1)[cf].mean() - Pf.max(1)[~cf].mean()),
    }
