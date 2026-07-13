"""Per-subject calibration ablation for EEG workload — the transfer lever the fusion investigation surfaced.

n-back workload band-power is subject-idiosyncratic in ABSOLUTE scale, so a zero-calibration cross-subject
decoder sits near the floor. Z-scoring each subject by its own feature statistics (unsupervised, no labels —
the EEG analog of the Riemannian re-centering that closes the motor-imagery gap) removes that offset. This
measures the effect rigorously and bounds the transductive optimism:

  - raw            : no per-subject normalization (the zero-calibration number)
  - transductive-z : each subject z-scored by ALL its own blocks' stats (uses the test blocks -> optimistic)
  - calib-half-z   : stats from a held-out CALIBRATION half of each test subject, scored on the other half
                     (the unbiased, deployment-real number — a short unlabeled calibration set)

Also reports the fusion picture on the z-scored features (EEG becomes the stronger modality; the oracle
grows) so the numbers the README cites are reproducible + recorded, not hand-typed.

    python -m neuroscan.tasks.workload.calibration_ablation
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import GroupKFold

from core.data import store
from core.data.eeg.base import EpochCfg
from core.data.fnirs.base import FnirsCfg
from core.features import Amplitude, BandPower, SubjectNorm
from neuroscan.evaluation import metrics, results

logger = logging.getLogger(__name__)

_EEG_CFG = EpochCfg(fmin=4, fmax=30, tmin=0.0, tmax=40.0, resample=100.0)
_K = 5
_SEED = 0


def _lda():
    return LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")


def _cv_raw_or_transductive(F, y, g, subs, zt):
    accs = []
    for tr, te in GroupKFold(_K).split(subs, groups=subs):
        mtr, mte = np.isin(g, subs[tr]), np.isin(g, subs[te])
        Fz = SubjectNorm.zscore_per_subject(F, g) if zt else F
        accs.append(metrics.Metrics.accuracy(y[mte], _lda().fit(Fz[mtr], y[mtr]).predict(Fz[mte])))
    return float(np.mean(accs))


def _cv_calib_half(F, y, g, subs, rng):
    """Leakage-free per-subject calibration: train z-scored on train subjects; for each test subject, fit stats on
    a random half of its blocks and score the other half."""
    accs = []
    for tr, te in GroupKFold(_K).split(subs, groups=subs):
        Ftr = SubjectNorm.zscore_per_subject(F, g)            # train side: per-subject z (train subjects only used)
        mtr = np.isin(g, subs[tr])
        clf = _lda().fit(Ftr[mtr], y[mtr])
        yt, yp = [], []
        for s in subs[te]:
            idx = np.where(g == s)[0]
            perm = rng.permutation(idx)
            h = len(perm) // 2
            mu, sd = F[perm[:h]].mean(0), F[perm[:h]].std(0)  # stats from the calibration half only
            ev = perm[h:]
            yp.extend(clf.predict((F[ev] - mu) / (sd + 1e-6)))
            yt.extend(y[ev])
        accs.append(metrics.Metrics.accuracy(np.array(yt), np.array(yp)))
    return float(np.mean(accs))


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    for lib_name in ("mne", "moabb", "braindecode"):
        logging.getLogger(lib_name).setLevel(logging.WARNING)
    rng = np.random.default_rng(_SEED)
    me = store.Store.load("shin2017_nback_eeg", _EEG_CFG)
    mf = store.Store.load("shin2017_nback", FnirsCfg())
    subs = np.array(sorted(set(me["subject"].unique().to_list()) & set(mf["subject"].unique().to_list())))
    qe = me.filter(me["subject"].is_in([str(s) for s in subs]))
    qf = mf.filter(mf["subject"].is_in([str(s) for s in subs]))
    Xe, y = store.Store.gather(qe)
    Xf, yf = store.Store.gather(qf)
    assert np.array_equal(y, yf), "EEG/fNIRS blocks misaligned"
    ge = qe["subject"].to_numpy()
    Fe, Ff = BandPower.band_powers(Xe, _EEG_CFG.resample), Amplitude.amplitude_features(Xf)

    out = {
        "eeg_raw": _cv_raw_or_transductive(Fe, y, ge, subs, zt=False),
        "eeg_ztrans": _cv_raw_or_transductive(Fe, y, ge, subs, zt=True),
        "eeg_zcalib": _cv_calib_half(Fe, y, ge, subs, rng),
        "fnirs_raw": _cv_raw_or_transductive(Ff, y, ge, subs, zt=False),
        "fnirs_ztrans": _cv_raw_or_transductive(Ff, y, ge, subs, zt=True),
    }
    # fusion picture on the z-scored (transductive) features: EEG becomes the strong modality; oracle grows
    Fez, Ffz = SubjectNorm.zscore_per_subject(Fe, ge), SubjectNorm.zscore_per_subject(Ff, ge)
    CE, CF, LATE = [], [], []
    for tr, te in GroupKFold(_K).split(subs, groups=subs):
        mtr, mte = np.isin(ge, subs[tr]), np.isin(ge, subs[te])
        pe = _lda().fit(Fez[mtr], y[mtr]).predict_proba(Fez[mte])
        pf = _lda().fit(Ffz[mtr], y[mtr]).predict_proba(Ffz[mte])
        CE.append(pe.argmax(1) == y[mte])
        CF.append(pf.argmax(1) == y[mte])
        LATE.append(((pe + pf) / 2).argmax(1) == y[mte])
    ce, cf, late = np.concatenate(CE), np.concatenate(CF), np.concatenate(LATE)
    out.update({
        "eeg_z_best": float(ce.mean()), "fnirs_z": float(cf.mean()),
        "late_z": float(late.mean()), "oracle_z": float((ce | cf).mean()),
        "err_corr_z": float(np.corrcoef(ce.astype(float), cf.astype(float))[0, 1]),
    })

    for k, v in out.items():
        logger.info(f"  {k:14s} {v:.3f}")
    logger.info(f"\n  EEG transfer: raw {out['eeg_raw']:.3f} -> calib-half {out['eeg_zcalib']:.3f} (unbiased) "
          f"/ transductive {out['eeg_ztrans']:.3f} (upper bound)")

    run_dir = Path("runs") / "calibration_ablation_shin2017_nback_eeg"
    run_dir.mkdir(parents=True, exist_ok=True)
    # record under the harness-schema key so results.record picks it up as one row of fields
    (run_dir / "aggregate.json").write_text(json.dumps(
        {"method": "calibration_ablation", "regime": "cross_subject_kfold", "n_classes": 3,
         "fold_mean": {"acc": out["eeg_zcalib"]}, "per_role_mean": out}, indent=2))
    results.Results.record(run_dir)
    logger.info(f"-> recorded {run_dir.name}")


if __name__ == "__main__":
    main()
