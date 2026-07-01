"""neuroviz fNIRS export — the hemodynamic view (Shin n-back workload), same JSON schema the web viewer loads.

The fNIRS counterpart to export.py: animated **HbO/HbR topomaps per workload class** (watch the hemodynamic
response build over ~5-8 s), the prefrontal optode montage, example waveforms, and per-class **LDA channel
weights** (what the amplitude-feature decoder reads). Shares the EEG viewer's schema (channels/pos/classes/
frames/waveforms) so one web app renders both modalities.

    python -m neuroviz.export_fnirs --subject 1
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

N_FRAMES = 40                    # hemodynamic-response animation frames
CLASS_NAMES = {0: "0-back", 1: "2-back", 2: "3-back"}


def _subject_epochs(subject: int):
    """(X [n,72,t] HbO|HbR, y, ch_names[36], pos2d[36,2], fs) for one subject via the adapter + montage."""
    import scipy.io as sio

    from core.config import raw_dir
    from core.data.fnirs.base import FnirsCfg
    from core.data.fnirs.shin2017 import adapter

    X, y, _ = adapter("nback").get_data([subject], FnirsCfg(tmax=20.0))
    d = raw_dir() / "shin2017" / f"VP{subject:03d}-NIRS"
    mnt = sio.loadmat(d / "mnt_nback.mat", struct_as_record=False, squeeze_me=True)["mnt_nback"]
    names = [str(c) for c in np.asarray(mnt.clab)][:36]
    pos = np.stack([np.asarray(mnt.x)[:36], np.asarray(mnt.y)[:36]], axis=1).astype(float)
    pos = pos - pos.mean(0)
    pos = pos / (np.abs(pos).max() + 1e-9)                  # normalize into [-1, 1]
    return X, y, names, pos, 10.0


def _frames(X, y, chan_slice, n_frames=N_FRAMES):
    """Per-class time-resolved HbO (or HbR) topomap: mean over trials, downsampled to n_frames.
    Returns ({class: [frame][ch]}, frame_times) — the hemodynamic response building over the trial."""
    T = X.shape[2]
    edges = np.linspace(0, T, n_frames + 1).astype(int)
    frames = {}
    for c in sorted(set(y.tolist())):
        m = X[y == c][:, chan_slice, :].mean(0)             # [36, t] mean HbO/HbR
        frames[CLASS_NAMES[c]] = [m[:, edges[i]:edges[i + 1]].mean(1).tolist() for i in range(n_frames)]
    ftimes = [float((edges[i] + edges[i + 1]) / 2 / 10.0 - 2.0) for i in range(n_frames)]  # tmin=-2 s
    return frames, ftimes


def _lda_patterns(X, y):
    """Per-class LDA weight on the HbO MEAN feature (what the decoder reads), one value per channel."""
    from baselines.fnirs_features import _features
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

    lda = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto").fit(_features(X), y)
    coef = np.atleast_2d(lda.coef_)                         # [n_class, 216] (mean|slope|peak × 72)
    classes = sorted(set(y.tolist()))
    if coef.shape[0] == 1 and len(classes) == 2:
        coef = np.vstack([-coef[0], coef[0]])
    out = {}
    for c, row in zip(classes, coef):
        w = np.asarray(row)[:36]                            # HbO mean-feature block
        out[CLASS_NAMES[c]] = (w / (np.abs(w).max() + 1e-9)).tolist()
    return out


def _waveforms(X, y, names, n_t=300):
    """One example trial per class — BOTH chromophores per optode (the raw data): {chan:{hbo,hbr}}.
    HbO = channels 0..35, HbR = 36..71 at the same optodes; showing both reveals the anti-correlation."""
    T = X.shape[2]
    step = max(1, T // n_t)
    ti = np.arange(0, T, step)
    t = (ti / 10.0 - 2.0).tolist()
    out = {}
    for c in sorted(set(y.tolist())):
        ei = np.where(y == c)[0][0]
        out[CLASS_NAMES[c]] = {names[i]: {"hbo": X[ei, i, ti].tolist(), "hbr": X[ei, i + 36, ti].tolist()}
                               for i in range(len(names))}
    return {"t": t, "trials": out, "chans": names}


def _predictions(subject: int, X, y):
    """Honest per-trial output: train the fNIRS decoder on the OTHER subjects (LOSO), predict THIS
    subject's trials. Returns ({class: {truth, pred, probs, correct}} for the shown example trial) and the
    subject's cross-subject fold accuracy — so the viewer shows ground truth vs prediction, not just signal."""
    import polars as pl

    from baselines import fnirs_features as ff
    from core.data import store
    from core.data.fnirs.base import FnirsCfg

    meta = store.load("shin2017_nback", FnirsCfg(tmax=20.0))
    Xtr, ytr = store.gather(meta.filter(pl.col("subject") != str(subject)))
    clf = ff.fit(Xtr, ytr)
    probs = ff.score(clf, X)
    pred = probs.argmax(1)
    per = {}
    for c in sorted(set(y.tolist())):
        i = int(np.where(y == c)[0][0])                    # the example trial shown for this class
        per[CLASS_NAMES[c]] = {"truth": CLASS_NAMES[c], "pred": CLASS_NAMES[int(pred[i])],
                               "probs": [round(float(p), 3) for p in probs[i]],
                               "correct": bool(pred[i] == c)}
    score = {"acc": round(float((pred == y).mean()), 3), "chance": round(1 / 3, 3),
             "regime": "cross-subject (LOSO)", "decoder": "fNIRS mean+slope+peak → LDA"}
    return per, score


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--subject", type=int, default=1)
    ap.add_argument("--out", default="neuroviz/web/data")
    args = ap.parse_args()

    X, y, names, pos, fs = _subject_epochs(args.subject)
    print(f"fNIRS subject {args.subject}: {len(X)} epochs, {len(names)} ch, classes {sorted(set(y.tolist()))}")

    hbo, ftimes = _frames(X, y, slice(0, 36))
    hbr, _ = _frames(X, y, slice(36, 72))
    data = {
        "modality": "fnirs",
        "subject": str(args.subject),
        "sfreq": fs,
        "channels": names,
        "pos": pos.tolist(),
        "classes": [CLASS_NAMES[c] for c in sorted(set(y.tolist()))],
        "frames": {"HbO": hbo, "HbR": hbr},                # signal maps (the chromophores)
        "frame_times": ftimes,
        "lda_patterns": _lda_patterns(X, y),               # decoder view (per-class HbO weight)
        "waveforms": _waveforms(X, y, names),
    }
    per, score = _predictions(args.subject, X, y)
    data["predictions"] = per                              # ground truth vs decoder prediction (per shown trial)
    data["score"] = score                                  # the honest cross-subject decoder accuracy
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"fnirs_subject{args.subject}.json").write_text(json.dumps(data))

    # modality-aware manifest: merge with any existing EEG entry
    mpath = out / "manifest.json"
    man = json.loads(mpath.read_text()) if mpath.exists() else {}
    if "modalities" not in man:                             # migrate old {subjects:[...]} = EEG
        man = {"modalities": {"eeg": man.get("subjects", [])}, }
    man["modalities"].setdefault("fnirs", [])
    subs = sorted({*man["modalities"]["fnirs"], int(args.subject)})
    man["modalities"]["fnirs"] = subs
    mpath.write_text(json.dumps(man))
    print(f"-> {out}/fnirs_subject{args.subject}.json  (+ manifest, fnirs subjects {subs})")


if __name__ == "__main__":
    main()
