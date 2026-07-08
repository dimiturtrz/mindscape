"""Fusion combiners/diagnostics — pure functions on per-block probabilities / correct-masks. Synthetic."""
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from baselines.fusion import combine
from baselines.fusion.base import FusionData, ModalityModels, PooledProbs


def test_complementarity_oracle_and_error_structure():
    #        block: 0    1    2    3
    ce = np.array([1, 1, 0, 0], bool)      # EEG correct on 0,1
    cf = np.array([1, 0, 1, 0], bool)      # fNIRS correct on 0,2 — they disagree on 1,2
    comp = combine.complementarity({"eeg": 0.5, "fnirs": 0.5}, ce, cf)
    assert comp["best_single"] == 0.5
    assert comp["oracle_either"] == 0.75           # either right on 0,1,2 -> 3/4
    assert comp["both_correct"] == 0.25 and comp["both_wrong"] == 0.25
    assert comp["eeg_only"] == 0.25 and comp["fnirs_only"] == 0.25


def test_aggregation_sweep_mean_and_confgap():
    y = np.array([0, 1, 0, 1])
    # a confident-and-right EEG on some blocks, fNIRS elsewhere
    Pe = np.array([[.9, .1], [.1, .9], [.4, .6], [.6, .4]])   # right on 0,1
    Pf = np.array([[.6, .4], [.6, .4], [.8, .2], [.2, .8]])   # right on 0,2,3
    ce = Pe.argmax(1) == y
    cf = Pf.argmax(1) == y
    agg = combine.aggregation_sweep(PooledProbs(eeg=Pe, fnirs=Pf, stacking=Pe, cal_eeg=Pe, cal_fnirs=Pf,
                                                y=y, eeg_correct=ce, fnirs_correct=cf))
    assert set(combine.SWEEP_KEYS) <= set(agg)                # every sweep key present
    assert 0.0 <= agg["mean"] <= 1.0
    # conf_gap = mean max-prob(correct) - mean max-prob(wrong); finite, sign as computed
    assert np.isfinite(agg["eeg_conf_gap"]) and np.isfinite(agg["fnirs_conf_gap"])


def _fnirs_epochs(rng, y, n_ch=4, n_t=40):
    """fNIRS-like epochs `[n, ch, t]` whose class rides in the amplitude (offset per class)."""
    return np.stack([rng.normal(scale=0.4, size=(n_ch, n_t)) + cls * 2.0 for cls in y])


def test_feature_fusion_concats_eeg_features_plus_fnirs_and_predicts():
    """Feature-level fusion concatenates the EEG tangent feature `Fe` with fNIRS mean/slope/peak -> LDA.
    A class-separable signal in BOTH sides must produce a valid probability matrix that decodes."""
    rng = np.random.default_rng(0)
    ytr = np.array([0, 1] * 20)
    yte = np.array([0, 1] * 8)
    Fe_tr = (ytr * 3.0)[:, None] + rng.normal(scale=0.5, size=(len(ytr), 5))   # EEG feature separable by class
    Fe_te = (yte * 3.0)[:, None] + rng.normal(scale=0.5, size=(len(yte), 5))
    Xf_tr, Xf_te = _fnirs_epochs(rng, ytr), _fnirs_epochs(rng, yte)
    probs = combine.feature_fusion(Fe_tr, Xf_tr, ytr, Fe_te, Xf_te)
    assert probs.shape == (len(yte), 2)
    assert np.allclose(probs.sum(1), 1.0, atol=1e-6)
    assert (probs.argmax(1) == yte).mean() > 0.8


def _lda_models():
    """ModalityModels whose eeg/fnirs decoders are plain LDAs on the (already-feature) inputs."""
    def eeg_probs(etr, ytr, _gtr, ete, _gte):
        return LinearDiscriminantAnalysis().fit(etr, ytr).predict_proba(ete)

    def fnirs_fit(ftr, ytr):
        return LinearDiscriminantAnalysis().fit(ftr, ytr)

    return ModalityModels(eeg_probs=eeg_probs, fnirs_fit=fnirs_fit,
                          fnirs_score=lambda m, fte: m.predict_proba(fte))


def _fusion_train(n_groups, rng, per=8):
    """FusionData with `n_groups` subjects, 2 classes each (features separable by class), plus per-modality
    probability matrices. `smart_aggregators` builds its OOF over `train` and sizes it from the probs, so the
    probs must have one row per train block (the inner OOF is over the train set)."""
    eeg, fnirs, y, g = [], [], [], []
    for s in range(n_groups):
        for cls in (0, 1):
            for _ in range(per):
                eeg.append((cls * 3.0) + rng.normal(scale=0.6, size=4))
                fnirs.append((cls * 2.5) + rng.normal(scale=0.6, size=4))
                y.append(cls)
                g.append(s)
    train = FusionData(np.asarray(eeg), np.asarray(fnirs), np.asarray(y), np.asarray(g))
    eeg_probs = LinearDiscriminantAnalysis().fit(train.eeg, train.y).predict_proba(train.eeg)
    fnirs_probs = LinearDiscriminantAnalysis().fit(train.fnirs, train.y).predict_proba(train.fnirs)
    return train, eeg_probs, fnirs_probs


def test_smart_aggregators_stacking_and_temperature_calibrate():
    """Enough train subjects for the inner GroupKFold(3): returns stacking probs + per-modality
    temperature-calibrated probs, each a proper probability matrix (no test data touched to fit them)."""
    rng = np.random.default_rng(1)
    train, eeg_te, fnirs_te = _fusion_train(6, rng)
    stacking, cal_eeg, cal_fnirs = combine.smart_aggregators(_lda_models(), train, eeg_te, fnirs_te)
    for P in (stacking, cal_eeg, cal_fnirs):
        assert P.shape == eeg_te.shape
        assert np.allclose(P.sum(1), 1.0, atol=1e-5)
    # temperature scaling does not change the argmax of the base probs (it only rescales confidence)
    assert np.array_equal(cal_eeg.argmax(1), eeg_te.argmax(1))


def test_smart_aggregators_falls_back_when_too_few_groups():
    """With fewer than 3 train subjects the inner GroupKFold raises -> fall back to the raw mean + raw probs
    (no stacking/temperature possible)."""
    rng = np.random.default_rng(2)
    train, eeg_te, fnirs_te = _fusion_train(2, rng)
    stacking, cal_eeg, cal_fnirs = combine.smart_aggregators(_lda_models(), train, eeg_te, fnirs_te)
    assert np.allclose(stacking, (eeg_te + fnirs_te) / 2.0)
    assert np.array_equal(cal_eeg, eeg_te) and np.array_equal(cal_fnirs, fnirs_te)
