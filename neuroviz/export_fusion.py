"""Export the EEG↔fNIRS fusion COMPLEMENTARITY view for neuroviz -> web/data/fusion.json.

The other two views animate a single modality's signal; this one shows the fusion result the numbers make:
on the same n-back blocks, EEG and fNIRS are each weak but fail on DIFFERENT blocks, so a per-block picture
(both-right / EEG-only / fNIRS-only / both-wrong) makes the complementarity — and why naive fusion can't cash
it — visible. Predictions are pooled out-of-fold over a subject-wise 5-fold GroupKFold (every block is a
held-out test block), the same protocol as tasks/workload/run_fusion.py.

    python -m neuroviz.export_fusion            # writes neuroviz/web/data/fusion.json (+ manifest flag)
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
from sklearn.model_selection import GroupKFold

from baselines.eeg import transfer
from baselines.fusion import combine
from core.data import store
from core.data.eeg.base import EpochCfg
from core.data.fnirs.base import FnirsCfg
from neuroscan.models import Methods
from neuroscan.tasks.workload.riemann import Riemann

logger = logging.getLogger(__name__)

_EEG, _FNIRS = "shin2017_nback_eeg", "shin2017_nback"
_EEG_CFG = EpochCfg(fmin=4, fmax=30, tmin=0.0, tmax=40.0, resample=100.0)
_OUT = Path(__file__).parent / "web" / "data" / "fusion.json"


def _gather(meta, subs):
    q = meta.filter(meta["subject"].is_in([str(s) for s in subs]))
    X, y = store.Store.gather(q)
    return X, y, q["subject"].to_numpy()


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    me = store.Store.load(_EEG, _EEG_CFG)
    mf = store.Store.load(_FNIRS, FnirsCfg())
    subs = np.array(sorted(set(me["subject"].unique().to_list()) & set(mf["subject"].unique().to_list())))
    classes = sorted(me["label"].unique().to_list())
    id2lab = dict(enumerate(sorted(me["label"].unique().to_list())))     # label_id -> name (matches gather order)
    n_classes = len(classes)
    ff, fs = Methods.get_method("fnirs_lda")

    blocks = []
    for tr, te in GroupKFold(n_splits=5).split(subs, groups=subs):
        Xe, y, ge = _gather(me, subs[tr]); Xf, yf, _ = _gather(mf, subs[tr])
        Xet, yt, gt = _gather(me, subs[te]); Xft, yft, _ = _gather(mf, subs[te])
        if not (np.array_equal(y, yf) and np.array_equal(yt, yft)):
            raise ValueError("EEG/fNIRS blocks misaligned — fusion invalid")
        # EEG = re-centered Riemann (per-subject, zero-shot) — the strong workload modality; fNIRS = amplitude LDA
        Ce = Riemann.cov(Xe)
        Cet = Riemann.cov(Xet)
        pe = transfer.zero_shot_predict(transfer.Domain(Ce, y, ge),
                                        transfer.Domain(Cet, groups=gt), scale=False).argmax(1)
        pf = fs(ff(Xf, y), Xft).argmax(1)
        blocks.extend({"subject": str(gt[i]), "truth": int(yt[i]), "eeg": int(pe[i]), "fnirs": int(pf[i])}
                      for i in range(len(yt)))

    ce = np.array([b["eeg"] == b["truth"] for b in blocks])
    cf = np.array([b["fnirs"] == b["truth"] for b in blocks])
    eeg_acc, fnirs_acc = float(ce.mean()), float(cf.mean())
    comp = combine.complementarity({"eeg": eeg_acc, "fnirs": fnirs_acc}, ce, cf)
    summary = {
        "eeg": eeg_acc, "fnirs": fnirs_acc,
        "oracle": comp["oracle_either"], "both_correct": comp["both_correct"],
        "eeg_only": comp["eeg_only"], "fnirs_only": comp["fnirs_only"],
        "both_wrong": comp["both_wrong"], "err_corr": comp["err_corr"],
        "chance": 1.0 / n_classes, "n": len(blocks),
    }
    # late fusion (avg prob) would need probs; we cite the recorded number so the view stays consistent
    try:
        rec = json.loads((Path(__file__).parents[1] / "results.json").read_text())["runs"]
        summary["late"] = rec["fusion_cross_subject_kfold_shin2017_nback"]["late"]
    except (FileNotFoundError, KeyError, json.JSONDecodeError):
        summary["late"] = None

    payload = {"classes": [c.replace(" ", "_") for c in classes], "id2label": id2lab,
               "summary": summary, "blocks": blocks}
    _OUT.write_text(json.dumps(payload))
    logger.info(f"wrote {_OUT.name}: {len(blocks)} blocks · EEG {summary['eeg']:.3f} fNIRS {summary['fnirs']:.3f} "
                f"oracle {summary['oracle']:.3f} (both-wrong {summary['both_wrong']:.3f})")

    # add a fusion flag to the manifest so the viewer shows the third mode (create it if the single-modality
    # exporters haven't run yet — they merge their own keys in later)
    man_path = _OUT.parent / "manifest.json"
    man = json.loads(man_path.read_text()) if man_path.exists() else {"modalities": {}}
    man["fusion"] = True
    man_path.write_text(json.dumps(man))
    logger.info(f"updated {man_path.name}: fusion=true")


if __name__ == "__main__":
    main()
