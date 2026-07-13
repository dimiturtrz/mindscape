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
from pathlib import Path

import numpy as np
from sklearn.model_selection import GroupKFold

from core.data import store
from core.data.eeg.base import EpochCfg
from core.data.fnirs.base import FnirsCfg
from neuroscan.models import Methods

_EEG, _FNIRS = "shin2017_nback_eeg", "shin2017_nback"
_EEG_CFG = EpochCfg(fmin=4, fmax=30, tmin=0.0, tmax=40.0, resample=100.0)
_OUT = Path(__file__).parent / "web" / "data" / "fusion.json"


def _gather(meta, subs):
    q = meta.filter(meta["subject"].is_in([str(s) for s in subs]))
    X, y = store.Store.gather(q)
    return X, y, q["subject"].to_numpy()


def main():
    me = store.Store.load(_EEG, _EEG_CFG)
    mf = store.Store.load(_FNIRS, FnirsCfg())
    subs = np.array(sorted(set(me["subject"].unique().to_list()) & set(mf["subject"].unique().to_list())))
    classes = sorted(me["label"].unique().to_list())
    id2lab = dict(enumerate(sorted(me["label"].unique().to_list())))     # label_id -> name (matches gather order)
    n_classes = len(classes)
    from pyriemann.estimation import Covariances

    from baselines.eeg import transfer
    ff, fs = Methods.get_method("fnirs_lda")

    blocks = []
    for tr, te in GroupKFold(n_splits=5).split(subs, groups=subs):
        Xe, y, ge = _gather(me, subs[tr]); Xf, yf, _ = _gather(mf, subs[tr])
        assert np.array_equal(y, yf)
        Xet, yt, gt = _gather(me, subs[te]); Xft, yft, _ = _gather(mf, subs[te])
        assert np.array_equal(yt, yft)
        # EEG = re-centered Riemann (per-subject, zero-shot) — the strong workload modality; fNIRS = amplitude LDA
        Ce = Covariances("oas").transform(Xe.astype(np.float64))
        Cet = Covariances("oas").transform(Xet.astype(np.float64))
        pe = transfer.zero_shot_predict(transfer.Domain(Ce, y, ge),
                                        transfer.Domain(Cet, groups=gt), scale=False).argmax(1)
        pf = fs(ff(Xf, y), Xft).argmax(1)
        for i in range(len(yt)):
            blocks.append({"subject": str(gt[i]), "truth": int(yt[i]),
                           "eeg": int(pe[i]), "fnirs": int(pf[i])})

    ce = np.array([b["eeg"] == b["truth"] for b in blocks])
    cf = np.array([b["fnirs"] == b["truth"] for b in blocks])
    summary = {
        "eeg": float(ce.mean()), "fnirs": float(cf.mean()),
        "oracle": float((ce | cf).mean()), "both_correct": float((ce & cf).mean()),
        "eeg_only": float((ce & ~cf).mean()), "fnirs_only": float((~ce & cf).mean()),
        "both_wrong": float((~ce & ~cf).mean()),
        "err_corr": float(np.corrcoef(ce.astype(float), cf.astype(float))[0, 1]),
        "chance": 1.0 / n_classes, "n": len(blocks),
    }
    # late fusion (avg prob) would need probs; we cite the recorded number so the view stays consistent
    try:
        rec = json.loads((Path(__file__).parents[1] / "results.json").read_text())["runs"]
        summary["late"] = rec["fusion_cross_subject_kfold_shin2017_nback"]["late"]
    except Exception:
        summary["late"] = None

    payload = {"classes": [c.replace(" ", "_") for c in classes], "id2label": id2lab,
               "summary": summary, "blocks": blocks}
    _OUT.write_text(json.dumps(payload))
    print(f"wrote {_OUT.name}: {len(blocks)} blocks · EEG {summary['eeg']:.3f} fNIRS {summary['fnirs']:.3f} "
          f"oracle {summary['oracle']:.3f} (both-wrong {summary['both_wrong']:.3f})")

    # add a fusion flag to the manifest so the viewer shows the third mode (create it if the single-modality
    # exporters haven't run yet — they merge their own keys in later)
    man_path = _OUT.parent / "manifest.json"
    man = json.loads(man_path.read_text()) if man_path.exists() else {"modalities": {}}
    man["fusion"] = True
    man_path.write_text(json.dumps(man))
    print(f"updated {man_path.name}: fusion=true")


if __name__ == "__main__":
    main()
