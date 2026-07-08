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
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from pyriemann.estimation import Covariances
from sklearn.model_selection import GroupKFold

from baselines.eeg import transfer
from baselines.fusion import combine
from baselines.fusion.base import FusionData, ModalityModels, PooledProbs
from core import config
from core.data import store
from core.data.eeg.base import EpochCfg
from core.data.fnirs.base import FnirsCfg
from neuroscan import models
from neuroscan.evaluation import metrics

logger = logging.getLogger(__name__)


@dataclass
class _RunnerModels:
    """The four modality callables the fold loop needs: the EEG probability + tangent-feature functions
    (both re-centered, so both take subject groups) and the fNIRS fit/score."""
    eeg_probs: Callable
    eeg_feats: Callable
    fnirs_fit: Callable
    fnirs_score: Callable


@dataclass
class _Analysis:
    """The two fusion diagnostics reported together: complementarity/oracle-headroom + the aggregation sweep."""
    complementarity: dict
    aggregation: dict

_EEG, _FNIRS = "shin2017_nback_eeg", "shin2017_nback"
# the recipes each modality decodes best at (from the unimodal runs)
_EEG_CFG = EpochCfg(fmin=4, fmax=30, tmin=0.0, tmax=40.0, resample=100.0)
_FNIRS_CFG = FnirsCfg()


def _gather_aligned(meta_e, meta_f, subs) -> FusionData:
    """Gather EEG + fNIRS epochs for `subs`, block-aligned, as a FusionData (with per-block subject groups).
    Hard guard that the two label sequences match — catches any silent misalignment before it can fake a
    fusion gain."""
    q_e = meta_e.filter(meta_e["subject"].is_in([str(s) for s in subs]))
    q_f = meta_f.filter(meta_f["subject"].is_in([str(s) for s in subs]))
    eeg, y_eeg = store.gather(q_e)
    fnirs, y_fnirs = store.gather(q_f)
    assert len(y_eeg) == len(y_fnirs) and np.array_equal(y_eeg, y_fnirs), \
        "EEG/fNIRS blocks misaligned — fusion invalid"
    return FusionData(eeg=eeg, fnirs=fnirs, y=y_eeg, groups=q_e["subject"].to_numpy())


def _parse_args():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--exp", default="nback_fusion", help="named fusion experiment in experiments.yaml")
    ap.add_argument("--set", dest="overrides", action="append", default=[], metavar="key=val",
                    help="ad-hoc override, e.g. --set regime=cross_subject --set params.plain_eeg=true")
    ap.add_argument("--out", default=None)
    ap.add_argument("--no-record", action="store_true")
    return ap.parse_args()


def _run_folds(fold_subs, meta_e, meta_f, subs, models: _RunnerModels):
    """Run every fold: unimodal + fused predictions per fold. Returns (rows, PooledProbs) — the pooled probs
    hold the concatenated correct-masks and probability stacks used for the oracle + aggregation-sweep."""
    field_names = ("eeg", "fnirs", "stacking", "cal_eeg", "cal_fnirs", "y", "eeg_correct", "fnirs_correct")
    pooled = {name: [] for name in field_names}
    rows = []
    modality_models = ModalityModels(models.eeg_probs, models.fnirs_fit, models.fnirs_score)
    for fold, test_subs in enumerate(fold_subs):
        train_subs = [s for s in map(str, subs) if s not in test_subs]
        train = _gather_aligned(meta_e, meta_f, train_subs)
        test = _gather_aligned(meta_e, meta_f, test_subs)

        # unimodal decoders — EEG re-centered by default (per-subject, zero-shot), fNIRS the amplitude LDA
        eeg_probs = models.eeg_probs(train.eeg, train.y, train.groups, test.eeg, test.groups)
        fnirs_probs = models.fnirs_score(models.fnirs_fit(train.fnirs, train.y), test.fnirs)
        late = (eeg_probs + fnirs_probs) / 2.0
        # feature fusion: concat the re-centered tangent-space EEG feature + fNIRS features -> shrinkage-LDA
        feature = combine.feature_fusion(models.eeg_feats(train.eeg, train.groups), train.fnirs, train.y,
                                         models.eeg_feats(test.eeg, test.groups), test.fnirs)
        # smarter output-space aggregators (stacking meta-LDA + per-modality temperature), fit on inner
        # OOF probs from a GroupKFold over the TRAIN subjects so the meta/temperature never see test data
        stacking, cal_eeg, cal_fnirs = combine.smart_aggregators(modality_models, train, eeg_probs, fnirs_probs)

        for name, value in (("eeg", eeg_probs), ("fnirs", fnirs_probs), ("stacking", stacking),
                            ("cal_eeg", cal_eeg), ("cal_fnirs", cal_fnirs), ("y", test.y),
                            ("eeg_correct", eeg_probs.argmax(1) == test.y),
                            ("fnirs_correct", fnirs_probs.argmax(1) == test.y)):
            pooled[name].append(value)
        rows.append({
            "fold": str(fold), "n": int(len(test.y)),
            "eeg": metrics.accuracy(test.y, eeg_probs.argmax(1)),
            "fnirs": metrics.accuracy(test.y, fnirs_probs.argmax(1)),
            "late": metrics.accuracy(test.y, late.argmax(1)),
            "feature": metrics.accuracy(test.y, feature.argmax(1)),
        })
        logger.info(f"  fold{fold}: eeg {rows[-1]['eeg']:.3f} | fnirs {rows[-1]['fnirs']:.3f} | "
              f"late {rows[-1]['late']:.3f} | feature {rows[-1]['feature']:.3f}")

    return rows, PooledProbs(**{name: np.concatenate(values) for name, values in pooled.items()})


def _report(regime, n_classes, rows, mean, analysis: _Analysis):
    """Print the per-role means, fusion-vs-unimodal deltas, oracle headroom, and the aggregation sweep."""
    comp, agg = analysis.complementarity, analysis.aggregation
    logger.info(f"\n=== fusion · {regime} · shin n-back ({len(rows)} folds, chance {1/n_classes:.3f}) ===")
    for role in ("eeg", "fnirs", "late", "feature"):
        logger.info(f"  {role:>8}: {mean[role]:.3f}")
    best_uni = comp["best_single"]
    logger.info(f"  fusion vs best-unimodal: late {mean['late']-best_uni:+.3f} | feature {mean['feature']-best_uni:+.3f}")
    logger.info(f"  ORACLE (either correct) {comp['oracle_either']:.3f}  (+{comp['oracle_either']-best_uni:.3f} headroom) "
          f"| err-corr {comp['err_corr']:+.3f} | both-wrong {comp['both_wrong']:.3f}")
    logger.info("  aggregation sweep (all output-space combiners vs best-single "
          f"{best_uni:.3f}, oracle {comp['oracle_either']:.3f}):")
    for key in combine.SWEEP_KEYS:
        logger.info(f"    {key:>16} {agg[key]:.3f}  ({agg[key]-best_uni:+.3f})")
    logger.info(f"    conf-gap (correct-wrong max-prob): eeg {agg['eeg_conf_gap']:+.3f} | fnirs {agg['fnirs_conf_gap']:+.3f}"
          "  <- ~0 => confidence does not predict correctness => output-space fusion cannot select")


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    for lib_name in ("mne", "moabb", "braindecode"):
        logging.getLogger(lib_name).setLevel(logging.WARNING)
    args = _parse_args()

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

    models_bundle = _RunnerModels(eeg_probs, eeg_feats, fn_fit, fn_score)
    rows, pooled = _run_folds(fold_subs, meta_e, meta_f, subs, models_bundle)

    mean = {k: float(np.mean([r[k] for r in rows])) for k in ("eeg", "fnirs", "late", "feature")}
    # complementarity: is fusion fundamentally hopeless, or just naive averaging? The oracle (either
    # modality correct) is the upper bound ANY fusion could reach; near-zero error correlation means the
    # two modalities fail on independent blocks, so a per-trial selector has headroom the mean can't touch.
    comp = combine.complementarity(mean, pooled.eeg_correct, pooled.fnirs_correct)   # oracle + error-independence
    agg = combine.aggregation_sweep(pooled)                  # every output-space combiner
    comp["best_aggregator"] = float(max(agg[k] for k in combine.SWEEP_KEYS))
    comp["oracle_gap_captured"] = comp["best_aggregator"] - comp["best_single"]
    _report(regime, n_classes, rows, mean, _Analysis(comp, agg))

    run_dir = Path(args.out) if args.out else Path("runs") / f"fusion_{regime}_shin2017_nback"
    run_dir.mkdir(parents=True, exist_ok=True)
    res = {"method": "fusion", "regime": regime, "n_classes": n_classes,
           "fold_mean": {"acc": mean["late"]}, "per_role_mean": mean,
           "complementarity": comp, "aggregation": agg, "per_fold": rows}
    (run_dir / "aggregate.json").write_text(json.dumps(res, indent=2))
    logger.info(f"-> {run_dir}/aggregate.json")


def _subject_folds(subs, k):
    """GroupKFold over the subject list (each subject in one test fold)."""
    subs = list(map(str, subs))
    gkf = GroupKFold(n_splits=k)
    for i, (tr, te) in enumerate(gkf.split(list(range(len(subs))), groups=subs)):
        yield f"fold{i}", [subs[j] for j in tr], None, [subs[j] for j in te]


if __name__ == "__main__":
    main()
