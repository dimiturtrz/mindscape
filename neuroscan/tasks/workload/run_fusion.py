"""Stage-3 EEG+fNIRS fusion on the Shin n-back — does combining the two modalities beat either alone?

The Shin set records EEG and fNIRS SIMULTANEOUSLY, so per subject the 27 workload blocks are the same
trials in both (verified: identical label sequence per subject). That lets us fuse them block-for-block.
Both modalities are weak alone here (~0.42, chance 0.333, tiny data — 702 blocks), so the honest question
is whether the complementary weak signals add. Even a null ("no gain on data this small") is a real result.

Two fusion levels, both kept dumb (tiny data → no room for a learned fusion model):
  - **late**    : fit each modality's decoder separately, AVERAGE their class-probabilities.
  - **feature** : concat per-block features (EEG log band-power + fNIRS mean/slope/peak) -> one LDA.

Reported under the matched 5-fold GroupKFold (benchmark-comparable), alongside the unimodal baselines.

    python -m neuroscan.tasks.workload.run_fusion --exp nback_fusion         # re-centered (strong) EEG
    python -m neuroscan.tasks.workload.run_fusion --exp nback_fusion_plain   # plain EEG (comparison)

Config (regime, plain-vs-recentered EEG) lives in experiments.yaml (task: fusion).
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np

from baselines.eeg import transfer
from baselines.fusion import combine
from core import config
from core.data import store
from core.data.eeg.base import EpochCfg
from core.data.fnirs.base import FnirsCfg
from neuroscan import models
from neuroscan.evaluation import metrics

logger = logging.getLogger(__name__)

_EEG, _FNIRS = "shin2017_nback_eeg", "shin2017_nback"
# the recipes each modality decodes best at (from the unimodal runs)
_EEG_CFG = EpochCfg(fmin=4, fmax=30, tmin=0.0, tmax=40.0, resample=100.0)
_FNIRS_CFG = FnirsCfg()


def _gather_aligned(meta_e, meta_f, subs):
    """Gather EEG + fNIRS epochs for `subs`, block-aligned. Returns (Xe, Xf, y) with a hard guard that the
    two label sequences match (catches any silent misalignment before it can fake a fusion gain)."""
    q_e = meta_e.filter(meta_e["subject"].is_in([str(s) for s in subs]))
    q_f = meta_f.filter(meta_f["subject"].is_in([str(s) for s in subs]))
    Xe, ye = store.gather(q_e)
    Xf, yf = store.gather(q_f)
    assert len(ye) == len(yf) and np.array_equal(ye, yf), "EEG/fNIRS blocks misaligned — fusion invalid"
    return Xe, Xf, ye


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    for _n in ("mne", "moabb", "braindecode"):
        logging.getLogger(_n).setLevel(logging.WARNING)
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--exp", default="nback_fusion", help="named fusion experiment in experiments.yaml")
    ap.add_argument("--set", dest="overrides", action="append", default=[], metavar="key=val",
                    help="ad-hoc override, e.g. --set regime=cross_subject --set params.plain_eeg=true")
    ap.add_argument("--out", default=None)
    ap.add_argument("--no-record", action="store_true")
    args = ap.parse_args()

    exp = config.load_experiment(args.exp, args.overrides)
    regime = exp.regime
    meta_e = store.load(_EEG, _EEG_CFG)
    meta_f = store.load(_FNIRS, _FNIRS_CFG)
    subs = sorted(set(meta_e["subject"].unique().to_list()) & set(meta_f["subject"].unique().to_list()))
    n_classes = int(meta_e["label_id"].max()) + 1
    recenter = not exp.params.get("plain_eeg", False)
    logger.info(f"fusion cloud: {len(subs)} paired subjects · {n_classes} classes · chance {1/n_classes:.3f} · "
          f"EEG {'re-centered' if recenter else 'plain'} Riemann")

    eeg_fit, eeg_score = models.get_method("riemann")        # the plain-EEG fallback (fusion EEG is Riemann)
    fn_fit, fn_score = models.get_method("fnirs_lda")

    def _cov(X):
        from pyriemann.estimation import Covariances
        return Covariances("oas").transform(X.astype(np.float64))

    def eeg_probs(Xtr, ytr, gtr, Xte, gte):
        """EEG decoder as a probability fn. Default = zero-shot RE-CENTERED Riemann (recenter each subject —
        train AND test — to the identity, unsupervised); needs the subject groups, so it can't be a plain
        get_method decoder. The `nback_fusion_plain` config (params.plain_eeg) falls back to plain Riemann."""
        if not recenter:
            return eeg_score(eeg_fit(Xtr, ytr), Xte)
        return transfer.zero_shot_predict(_cov(Xtr), ytr, gtr, _cov(Xte), scale=False, target_groups=gte)

    def eeg_feats(X, g):
        """EEG feature vector for feature-level fusion — the re-centered tangent-space rep, so its EEG side
        matches the strong probs-side EEG (not a crude log-variance)."""
        return transfer.recentered_tangent_features(_cov(X), g)

    # fold generator over the shared subject set (same split drives both modalities)
    if regime == "cross_subject_kfold":
        fold_subs = [([str(x) for x in te]) for _, _, _, te in _subject_folds(subs, k=5)]
    else:
        fold_subs = [[str(s)] for s in subs]

    rows = []
    CE, CF, Y = [], [], []                                   # pooled correct-masks for the oracle analysis
    PE, PF, STK, PEc, PFc = [], [], [], [], []               # pooled probs for the aggregation sweep
    for i, te_subs in enumerate(fold_subs):
        tr_subs = [s for s in map(str, subs) if s not in te_subs]
        Xe_tr, Xf_tr, y_tr = _gather_aligned(meta_e, meta_f, tr_subs)
        Xe_te, Xf_te, y_te = _gather_aligned(meta_e, meta_f, te_subs)
        g_tr = meta_e.filter(meta_e["subject"].is_in(tr_subs))["subject"].to_numpy()  # train row -> subject
        g_te = meta_e.filter(meta_e["subject"].is_in(te_subs))["subject"].to_numpy()  # test row -> subject

        # unimodal decoders — EEG re-centered by default (per-subject, zero-shot), fNIRS the amplitude LDA
        pe = eeg_probs(Xe_tr, y_tr, g_tr, Xe_te, g_te)
        pf = fn_score(fn_fit(Xf_tr, y_tr), Xf_te)
        # late fusion: average the two probability vectors
        p_late = (pe + pf) / 2.0
        # feature fusion: concat the re-centered tangent-space EEG feature + fNIRS features -> shrinkage-LDA
        p_feat = combine.feature_fusion(eeg_feats(Xe_tr, g_tr), Xf_tr, y_tr, eeg_feats(Xe_te, g_te), Xf_te)
        # smarter output-space aggregators (stacking meta-LDA + per-modality temperature), fit on inner
        # OOF probs from a GroupKFold over the TRAIN subjects so the meta/temperature never see test data
        stk, pec, pfc = combine.smart_aggregators(eeg_probs, fn_fit, fn_score,
                                                  Xe_tr, Xf_tr, y_tr, g_tr, pe, pf)

        CE.append(pe.argmax(1) == y_te)
        CF.append(pf.argmax(1) == y_te)
        Y.append(y_te)
        PE.append(pe)
        PF.append(pf)
        STK.append(stk)
        PEc.append(pec)
        PFc.append(pfc)
        rows.append({
            "fold": str(i), "n": int(len(y_te)),
            "eeg": metrics.accuracy(y_te, pe.argmax(1)),
            "fnirs": metrics.accuracy(y_te, pf.argmax(1)),
            "late": metrics.accuracy(y_te, p_late.argmax(1)),
            "feature": metrics.accuracy(y_te, p_feat.argmax(1)),
        })
        logger.info(f"  fold{i}: eeg {rows[-1]['eeg']:.3f} | fnirs {rows[-1]['fnirs']:.3f} | "
              f"late {rows[-1]['late']:.3f} | feature {rows[-1]['feature']:.3f}")

    mean = {k: float(np.mean([r[k] for r in rows])) for k in ("eeg", "fnirs", "late", "feature")}
    # complementarity: is fusion fundamentally hopeless, or just naive averaging? The oracle (either
    # modality correct) is the upper bound ANY fusion could reach; near-zero error correlation means the
    # two modalities fail on independent blocks, so a per-trial selector has headroom the mean can't touch.
    ce, cf = np.concatenate(CE), np.concatenate(CF)
    comp = combine.complementarity(mean, ce, cf)             # oracle headroom + error-independence
    y, Pe, Pf = np.concatenate(Y), np.concatenate(PE), np.concatenate(PF)
    Stk, Pce, Pcf = np.concatenate(STK), np.concatenate(PEc), np.concatenate(PFc)
    agg = combine.aggregation_sweep(Pe, Pf, Stk, Pce, Pcf, y, ce, cf)   # every output-space combiner
    _acc_keys = combine.SWEEP_KEYS
    comp["best_aggregator"] = float(max(agg[k] for k in _acc_keys))
    comp["oracle_gap_captured"] = comp["best_aggregator"] - comp["best_single"]
    logger.info(f"\n=== fusion · {regime} · shin n-back ({len(rows)} folds, chance {1/n_classes:.3f}) ===")
    for k in ("eeg", "fnirs", "late", "feature"):
        logger.info(f"  {k:>8}: {mean[k]:.3f}")
    best_uni = comp["best_single"]
    logger.info(f"  fusion vs best-unimodal: late {mean['late']-best_uni:+.3f} | feature {mean['feature']-best_uni:+.3f}")
    logger.info(f"  ORACLE (either correct) {comp['oracle_either']:.3f}  (+{comp['oracle_either']-best_uni:.3f} headroom) "
          f"| err-corr {comp['err_corr']:+.3f} | both-wrong {comp['both_wrong']:.3f}")
    logger.info("  aggregation sweep (all output-space combiners vs best-single "
          f"{best_uni:.3f}, oracle {comp['oracle_either']:.3f}):")
    for k in _acc_keys:
        logger.info(f"    {k:>16} {agg[k]:.3f}  ({agg[k]-best_uni:+.3f})")
    logger.info(f"    conf-gap (correct-wrong max-prob): eeg {agg['eeg_conf_gap']:+.3f} | fnirs {agg['fnirs_conf_gap']:+.3f}"
          "  <- ~0 => confidence does not predict correctness => output-space fusion cannot select")

    run_dir = Path(args.out) if args.out else Path("runs") / f"fusion_{regime}_shin2017_nback"
    run_dir.mkdir(parents=True, exist_ok=True)
    res = {"method": "fusion", "regime": regime, "n_classes": n_classes,
           "fold_mean": {"acc": mean["late"]}, "per_role_mean": mean,
           "complementarity": comp, "aggregation": agg, "per_fold": rows}
    (run_dir / "aggregate.json").write_text(json.dumps(res, indent=2))
    logger.info(f"-> {run_dir}/aggregate.json")


def _subject_folds(subs, k):
    """GroupKFold over the subject list (each subject in one test fold)."""
    from sklearn.model_selection import GroupKFold
    subs = list(map(str, subs))
    gkf = GroupKFold(n_splits=k)
    for i, (tr, te) in enumerate(gkf.split(list(range(len(subs))), groups=subs)):
        yield f"fold{i}", [subs[j] for j in tr], None, [subs[j] for j in te]


if __name__ == "__main__":
    main()
