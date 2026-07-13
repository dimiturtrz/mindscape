"""EEG↔fNIRS fusion combiners — the output-space aggregators + feature fusion, and the diagnostics that
frame the result (complementarity, the aggregation sweep). Operate on the two modalities' per-block
probabilities (or features); the runner (`tasks/workload/run_fusion.py`) owns the fold loop, the aligned
gather, and the recording, and calls these. See the README fusion section for what they show (a rigorous
null: complementarity is real, no output-space combiner cashes it — confidence doesn't track correctness)."""
from __future__ import annotations

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from baselines.fusion.base import FusionData, ModalityModels, PooledProbs
from core.features import Amplitude
from neuroscan.evaluation.calibrate import TemperatureScaler


def feature_fusion(Fe_tr, Xf_tr, y_tr, Fe_te, Xf_te) -> np.ndarray:
    """Feature-level fusion: concat the EEG feature `Fe` (the re-centered tangent-space vector — the strong
    EEG representation, so this is a fair test) + fNIRS mean/slope/peak -> shrinkage-LDA. Test probs. The
    caller supplies `Fe` (via `transfer.recentered_tangent_features`) so the EEG side matches the probs side."""
    ftr = np.concatenate([Fe_tr, Amplitude.amplitude_features(Xf_tr)], axis=1)
    fte = np.concatenate([Fe_te, Amplitude.amplitude_features(Xf_te)], axis=1)
    clf = make_pipeline(StandardScaler(),
                        LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")).fit(ftr, y_tr)
    return clf.predict_proba(fte)


def smart_aggregators(models: ModalityModels, train: FusionData,
                      eeg_test_probs: np.ndarray, fnirs_test_probs: np.ndarray):
    """The learned/calibrated output-space combiners, fit WITHOUT touching test data: an inner GroupKFold(3)
    over the train subjects yields out-of-fold base probs, on which we fit (a) a stacking meta-LDA over the
    concatenated [eeg, fnirs] probs and (b) a per-modality temperature. The EEG decoder is passed as a
    probability function (`models.eeg_probs`) so re-centering — which needs the subject groups — flows into
    the OOF too. Returns (stacking probs, calibrated eeg probs, calibrated fnirs probs); falls back to the
    raw probs if a train subject group is too small to inner-split."""
    n, n_classes = eeg_test_probs.shape
    oof_eeg, oof_fnirs = np.zeros((n, n_classes)), np.zeros((n, n_classes))
    try:
        for train_idx, val_idx in GroupKFold(n_splits=3).split(np.arange(n), groups=train.groups):
            oof_eeg[val_idx] = models.eeg_probs(train.eeg[train_idx], train.y[train_idx],
                                                train.groups[train_idx], train.eeg[val_idx], train.groups[val_idx])
            oof_fnirs[val_idx] = models.fnirs_score(models.fnirs_fit(train.fnirs[train_idx], train.y[train_idx]),
                                                    train.fnirs[val_idx])
    except ValueError:                                       # too few groups for an inner split
        return (eeg_test_probs + fnirs_test_probs) / 2.0, eeg_test_probs, fnirs_test_probs

    def _logit(p):
        return np.log(p + 1e-12)

    def _softmax(z):
        z = z - z.max(1, keepdims=True)
        e = np.exp(z)
        return e / e.sum(1, keepdims=True)

    meta = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto").fit(
        np.concatenate([oof_eeg, oof_fnirs], axis=1), train.y)
    stacking = meta.predict_proba(np.concatenate([eeg_test_probs, fnirs_test_probs], axis=1))
    temp_eeg = TemperatureScaler().fit(_logit(oof_eeg), train.y).T
    temp_fnirs = TemperatureScaler().fit(_logit(oof_fnirs), train.y).T
    return (stacking, _softmax(_logit(eeg_test_probs) / temp_eeg),
            _softmax(_logit(fnirs_test_probs) / temp_fnirs))


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


def aggregation_sweep(pooled: PooledProbs) -> dict:
    """Can any OUTPUT-space combiner capture the oracle headroom? Try every cheap→learned aggregator on the
    pooled probs. They all fail here because confidence does not track correctness (see the conf_gap) — the
    reliability signal a selector needs is not in the probabilities at all."""
    eeg, fnirs, y = pooled.eeg, pooled.fnirs, pooled.y
    cal_eeg, cal_fnirs = pooled.cal_eeg, pooled.cal_fnirs
    eeg_correct, fnirs_correct = pooled.eeg_correct, pooled.fnirs_correct

    def acc(P):
        return float((P.argmax(1) == y).mean())

    conf_eeg, conf_fnirs = eeg.max(1, keepdims=True), fnirs.max(1, keepdims=True)
    return {
        "mean": acc((eeg + fnirs) / 2), "product": acc(eeg * fnirs),
        "conf_weight": acc(conf_eeg * eeg + conf_fnirs * fnirs),
        "maxconf_pick": float((np.where(conf_eeg >= conf_fnirs, eeg.argmax(1, keepdims=True),
                                        fnirs.argmax(1, keepdims=True)).ravel() == y).mean()),
        "stacking": acc(pooled.stacking),
        "cal_mean": acc((cal_eeg + cal_fnirs) / 2),
        "cal_conf_weight": acc(cal_eeg.max(1, keepdims=True) * cal_eeg
                               + cal_fnirs.max(1, keepdims=True) * cal_fnirs),
        # does confidence predict correctness? gap = mean max-prob(correct) - max-prob(wrong), per modality
        "eeg_conf_gap": float(eeg.max(1)[eeg_correct].mean() - eeg.max(1)[~eeg_correct].mean()),
        "fnirs_conf_gap": float(fnirs.max(1)[fnirs_correct].mean() - fnirs.max(1)[~fnirs_correct].mean()),
    }
