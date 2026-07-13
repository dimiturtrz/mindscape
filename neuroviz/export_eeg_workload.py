"""neuroviz export — the EEG view of the WORKLOAD task (Shin n-back), the third modality alongside fNIRS +
fusion. Distinct from export.py (which is EEG *motor imagery*, BCI-2a): here the signal is band-power at the
workload rhythms — frontal **theta** rises and parietal **alpha** suppresses with load — not mu/beta ERD.

Reads everything from the **processed store** (one format): epochs via `store.gather`, channel names via
`store.channels` (persisted by the adapter, not re-read from raw). Names → a standard-10-05 montage → 2D
topomap positions. Reuses the motor-imagery exporter's topomap / CSP / Riemann helpers on an MNE
`EpochsArray` rebuilt from the store, so the two EEG views share one rendering path.

    python -m neuroviz.export_eeg_workload --subject 1
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from core.data import store
from core.data.eeg.base import EpochCfg
from neuroviz.export import N_FRAMES, _csp_patterns, _pos2d, _riemann_patterns, _waveforms

# the workload EEG recipe (matches tasks/workload runs): broadband 4-30 Hz, full 40 s block, 100 Hz
_CFG = EpochCfg(fmin=4, fmax=30, tmin=0.0, tmax=40.0, resample=100.0)
_DATASET = "shin2017_nback_eeg"
THETA = (4.0, 7.0)            # Hz — frontal-midline theta rises with load
ALPHA = (8.0, 13.0)           # Hz — parietal alpha suppresses with load
BETA = (13.0, 30.0)           # Hz


def _bandpower_frames(ep, labels, fmin, fmax, n_frames=N_FRAMES):
    """Per-class SPATIAL band-power map over the block — the workload signal is *where* a rhythm concentrates
    and how that differs by load, NOT temporal change-from-onset (that's the motor-imagery ERD story). So:
    band-limited log-power per channel per time-bin, then **spatially demeaned** each frame (subtract the
    across-channel mean) → a zero-centred pattern the diverging colormap shows as hot/cold regions. Switch
    class (0/2/3-back) to see the load effect (frontal theta up, parietal alpha down). Returns
    ({class: [frame][ch]}, frame_times)."""
    band = ep.copy().filter(fmin, fmax, verbose="error")
    sf = ep.info["sfreq"]
    power = (band.get_data() * 1e6) ** 2                     # [n_epochs, ch, t]
    T = power.shape[2]
    edges = np.linspace(0, T, n_frames + 1).astype(int)
    widths = np.diff(edges)
    frames = {}
    for c in sorted(set(labels)):
        p = power[labels == c].mean(0)                       # [ch, t] mean over blocks of this load
        logp = np.log(np.add.reduceat(p, edges[:-1], axis=1) / widths + 1e-20)   # [ch, n_frames] log band-power
        logp = logp - logp.mean(0, keepdims=True)            # spatial demean -> pattern (which channels lead)
        frames[str(c)] = logp.T.tolist()                     # -> [n_frames][ch]
    ftimes = ((edges[:-1] + edges[1:]) / 2 / sf).tolist()
    return frames, ftimes


def _epochs(subject: int):
    """Rebuild an MNE EpochsArray for one subject from the processed store + its channel names/montage —
    so the motor-imagery exporter's helpers (which take MNE Epochs) run unchanged on this task."""
    import mne

    names = store.Store.channels(_DATASET, _CFG)
    if not names:
        raise SystemExit(f"no channels.json for {_DATASET} — run `python -m core.data.store --name {_DATASET}`")
    meta = store.Store.load(_DATASET, _CFG)
    q = meta.filter(meta["subject"] == str(subject))
    X, y = store.Store.gather(q)                                   # [n, 28, t] float32, y in {0,1,2}
    info = mne.create_info(list(names), _CFG.resample or 100.0, "eeg")
    ep = mne.EpochsArray(X.astype(np.float64) * 1e-6, info, tmin=_CFG.tmin, verbose="error")
    ep.set_montage(mne.channels.make_standard_montage("standard_1005"), match_case=False, on_missing="ignore")
    labels = np.asarray(q["label"].to_list())               # '0-back'/'2-back'/'3-back' (gather keeps q's row order)
    return ep, labels


def _predictions(subject: int):
    """Honest per-trial output: Riemann (tangent space) trained on the OTHER subjects (LOSO), predict THIS
    subject's blocks. Covariance methods read the workload band-power; chance is 1/3."""
    from neuroscan.models import get_method

    meta = store.Store.load(_DATASET, _CFG)
    fit, score = get_method("riemann")
    tr = meta.filter(meta["subject"] != str(subject))
    te = meta.filter(meta["subject"] == str(subject))
    Xtr, ytr = store.Store.gather(tr)
    Xte, yte = store.Store.gather(te)
    probs = np.asarray(score(fit(Xtr, ytr), Xte))
    pred = probs.argmax(1)
    id2lab = {r["label_id"]: r["label"] for r in te.select("label_id", "label").unique().to_dicts()}
    per = {}
    for c in sorted(np.unique(yte).tolist()):
        i = int((yte == c).argmax())
        per[id2lab[c]] = {"truth": id2lab[c], "pred": id2lab[int(pred[i])],
                          "probs": [round(float(p), 3) for p in probs[i]],
                          "correct": bool(pred[i] == c)}
    score_d = {"acc": round(float((pred == yte).mean()), 3), "chance": round(1 / 3, 3),
               "regime": "cross-subject (LOSO)", "decoder": "Riemann (tangent space)"}
    return per, score_d


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--subject", type=int, default=1)
    ap.add_argument("--out", default="neuroviz/web/data")
    args = ap.parse_args()

    ep, labels = _epochs(args.subject)
    print(f"eeg-workload subject {args.subject}: {len(ep)} blocks, {len(ep.ch_names)} ch, "
          f"classes {sorted(set(labels))}")

    # per-class spatial band-power maps (θ/α/β) — switch load class to see frontal-theta / parietal-alpha shift
    theta_fr, ftimes = _bandpower_frames(ep, labels, *THETA)
    alpha_fr, _ = _bandpower_frames(ep, labels, *ALPHA)
    beta_fr, _ = _bandpower_frames(ep, labels, *BETA)
    data = {
        "subject": str(args.subject),
        "sfreq": float(ep.info["sfreq"]),
        "channels": list(ep.ch_names),
        "pos": _pos2d(ep.info).tolist(),
        "classes": [str(c) for c in sorted(set(labels))],
        "frames": {"theta": theta_fr, "alpha": alpha_fr, "beta": beta_fr},
        "frame_times": ftimes,
        "csp_patterns": _csp_patterns(ep, labels),
        "riemann_patterns": _riemann_patterns(ep, labels),
        "waveforms": _waveforms(ep, labels),
    }
    per, score = _predictions(args.subject)
    data["predictions"] = per
    data["score"] = score

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"eegwl_subject{args.subject}.json").write_text(json.dumps(data))
    # manifest: register under a workload-EEG modality key (the viewer groups it into the workload task)
    subs = sorted(int(p.stem.replace("eegwl_subject", "")) for p in out.glob("eegwl_subject*.json"))
    mpath = out / "manifest.json"
    man = json.loads(mpath.read_text()) if mpath.exists() else {"modalities": {}}
    man.setdefault("modalities", {})["eeg_workload"] = subs
    mpath.write_text(json.dumps(man))
    print(f"-> {out}/eegwl_subject{args.subject}.json  (+ manifest, eeg_workload subjects {subs})")


if __name__ == "__main__":
    main()
