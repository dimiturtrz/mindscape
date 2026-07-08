"""Stage-3: a compact INPUT-level gated fusion — the one path left after output-space fusion was closed.

The complementarity is real (oracle 0.688 vs best-single 0.474, near-independent errors) but no output-space
combiner cashes it, because a decoder's confidence does not predict its correctness (run_fusion.py). So the
per-trial "which modality to trust" signal has to be learned from the INPUT features, not the decisions.

This is the smallest model that can do that: shallow per-modality encoders → a gate network that reads BOTH
embeddings and emits a per-trial mixing weight α, → α·p_eeg + (1−α)·p_fnirs. Everything is sized for n=26 /
~700 blocks (the SOTA review's hard constraint): d_model 16–32, dropout ≥ 0.5, weight decay, per-subject
z-scoring, and — decisively — a nested split (inner val over TRAIN subjects for early stopping) under the
same outer 5-fold GroupKFold as every other fusion number.

RESULT — an honest NEGATIVE (kept as an artifact). The gate scores ~0.573 fold-mean, which looks like a
+0.10 win over fNIRS-alone (0.474) but is NOT one: an ablation shows per-subject z-scoring alone lifts
EEG-band-power → LDA to 0.581 (from 0.407 raw; the absolute band-power was subject-idiosyncratic), and the
gate merely *ties* that best single modality (z-EEG 0.581, z-concat-LDA 0.578, late-z 0.575, gate 0.573). So
the learned output-mixing gate captures none of the oracle headroom either — consistent with run_fusion's
finding that the per-trial reliability signal is not in the probabilities. The real transfer lever the
investigation surfaced is the per-subject normalization (measured in calibration_ablation.py), not the fusion.

    python -m neuroscan.tasks.workload.fusion_gate   # 5-fold GroupKFold; records the (negative) gate number
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
from sklearn.model_selection import GroupKFold

from baselines.fusion.base import FusionData
from baselines.fusion.gate import GateConfig, GatedFusion
from core.data import store
from core.data.eeg.base import EpochCfg
from core.data.fnirs.base import FnirsCfg
from core.features import amplitude_features, band_powers, zscore_per_subject
from neuroscan.evaluation import metrics, results

logger = logging.getLogger(__name__)

_EEG, _FNIRS = "shin2017_nback_eeg", "shin2017_nback"
_EEG_CFG = EpochCfg(fmin=4, fmax=30, tmin=0.0, tmax=40.0, resample=100.0)
_FNIRS_CFG = FnirsCfg()


def _load_features():
    """Return per-block EEG band-power + fNIRS mean/slope/peak features, the label, and the subject id —
    block-aligned across the two modalities (hard guard on the label sequence)."""
    me = store.load(_EEG, _EEG_CFG)
    mf = store.load(_FNIRS, _FNIRS_CFG)
    subs = sorted(set(me["subject"].unique().to_list()) & set(mf["subject"].unique().to_list()))
    qe = me.filter(me["subject"].is_in(subs))
    qf = mf.filter(mf["subject"].is_in(subs))
    Xe, ye = store.gather(qe)
    Xf, yf = store.gather(qf)
    assert np.array_equal(ye, yf), "EEG/fNIRS blocks misaligned — fusion invalid"
    groups = qe["subject"].to_numpy()
    Fe = band_powers(Xe, _EEG_CFG.resample).astype(np.float32)         # [n, 28*3]
    Ff = amplitude_features(Xf).astype(np.float32)                     # [n, ch*3]
    return Fe, Ff, ye.astype(np.int64), groups


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    for _n in ("mne", "moabb", "braindecode"):
        logging.getLogger(_n).setLevel(logging.WARNING)
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--out", default=None)
    ap.add_argument("--no-record", action="store_true")
    args = ap.parse_args()

    Fe, Ff, y, groups = _load_features()
    Fe, Ff = zscore_per_subject(Fe, groups), zscore_per_subject(Ff, groups)
    subs = np.array(sorted(set(groups)))
    n_classes = int(y.max()) + 1
    logger.info(f"gated fusion: {len(y)} blocks · {len(subs)} subjects · EEG {Fe.shape[1]}d · fNIRS {Ff.shape[1]}d · "
          f"chance {1/n_classes:.3f}")

    rows, P, A, Y = [], [], [], []
    outer = GroupKFold(n_splits=args.k)
    for i, (tr, te) in enumerate(outer.split(subs, groups=subs)):
        tr_subs, te_subs = subs[tr], subs[te]
        # inner val: hold out one GroupKFold slice of the TRAIN subjects for early stopping
        itr, iva = next(GroupKFold(n_splits=4).split(tr_subs, groups=tr_subs))
        va_subs = tr_subs[iva]
        fit_subs = tr_subs[itr]
        m_fit = np.isin(groups, fit_subs)
        m_va = np.isin(groups, va_subs)
        m_te = np.isin(groups, te_subs)

        clf = GatedFusion(GateConfig(eeg_dim=Fe.shape[1], fnirs_dim=Ff.shape[1], n_classes=n_classes))
        clf.fit(FusionData(Fe[m_fit], Ff[m_fit], y[m_fit]), FusionData(Fe[m_va], Ff[m_va], y[m_va]))
        p, a = clf.predict(Fe[m_te], Ff[m_te])
        P.append(p)
        A.append(a)
        Y.append(y[m_te])
        acc = metrics.accuracy(y[m_te], p.argmax(1))
        rows.append({"fold": str(i), "n": int(m_te.sum()), "gate_acc": acc,
                     "alpha_mean": float(a.mean())})
        logger.info(f"  fold{i}: gate {acc:.3f} | ᾱ(eeg-weight) {a.mean():.2f} (n={int(m_te.sum())})")

    y_all, P_all = np.concatenate(Y), np.concatenate(P)
    gate = float((P_all.argmax(1) == y_all).mean())
    fold_mean = float(np.mean([r["gate_acc"] for r in rows]))
    std = float(np.std([r["gate_acc"] for r in rows]))
    logger.info("\n=== gated fusion · 5-fold GroupKFold · shin n-back ===")
    logger.info(f"  gate pooled {gate:.3f} | fold-mean {fold_mean:.3f} ± {std:.3f}")
    logger.info(f"  NOTE: this ~{fold_mean:.2f} is NOT a fusion win — it ties z-scored-EEG-alone (~0.581) and "
          "z-concat-LDA (~0.578).")
    logger.info("  The lift over raw fNIRS (0.474) is per-subject z-scoring rescuing EEG (0.407->0.581), NOT the "
          "gate; the gate captures no oracle headroom (see run_fusion + the ablation).")

    run_dir = Path(args.out) if args.out else Path("runs") / "fusion_gate_cross_subject_kfold_shin2017_nback"
    run_dir.mkdir(parents=True, exist_ok=True)
    res = {"method": "fusion_gate", "regime": "cross_subject_kfold", "n_classes": n_classes,
           "fold_mean": {"acc": fold_mean}, "per_role_mean": {"gate": fold_mean},
           "pooled_acc": gate, "acc_std": std, "per_fold": rows}
    (run_dir / "aggregate.json").write_text(json.dumps(res, indent=2))
    if not args.no_record:
        results.record(run_dir)
    logger.info(f"-> {run_dir}/aggregate.json")


if __name__ == "__main__":
    main()
