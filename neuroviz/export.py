"""neuroviz data export — compute the field-standard motor-imagery views for one subject and write a
self-contained JSON the web viewer loads (a static export -> dependency-free in-browser viewer).

Views (all 2D, the conventions a neuro audience expects):
  - topomaps: mu (8-12 Hz) + beta (13-30 Hz) band power per class -> the contralateral ERD pattern
  - CSP spatial patterns: what the baseline decoder actually learns (should localize over C3/C4)
  - waveforms: example trials at C3/Cz/C4 per class

    python -m neuroviz.export --subject 1
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

# --- bands ---
MU = (8.0, 12.0)              # Hz — sensorimotor mu rhythm
BETA = (13.0, 30.0)           # Hz — sensorimotor beta rhythm

# --- preprocessing recipe (the broad band the viewer's power is computed in) ---
PROC_BAND = (4.0, 40.0)       # Hz — broadband for epoching before per-band power
SFREQ = 250.0                 # Hz — resample target
N_CLASSES = 4                 # left/right hand, feet, tongue

# --- view parameters ---
N_FRAMES = 50                 # ERD animation frames per trial
BASELINE_S = 0.5              # s — pre-imagery window for ERD baseline-normalization
N_CSP = 6                     # CSP spatial patterns to export (matches baselines/eeg/csp_lda.py n_components)
N_WAVE_T = 300                # downsampled time points for waveform display
PER_CLASS = 1                 # example trials per class in the waveform panel


def _load_epochs(subject: int):
    """MNE Epochs for a subject (broad 4-40 Hz band, montage set) via MOABB."""
    import mne
    from moabb.datasets import BNCI2014_001
    from moabb.paradigms import MotorImagery

    from core.config import Config
    Config.configure_moabb_download()
    para = MotorImagery(n_classes=N_CLASSES, fmin=PROC_BAND[0], fmax=PROC_BAND[1],
                        tmin=0.0, tmax=None, resample=SFREQ)
    ep, labels, _ = para.get_data(dataset=BNCI2014_001(), subjects=[subject], return_epochs=True)
    ep.set_montage(mne.channels.make_standard_montage("standard_1020"),
                   match_case=False, on_missing="ignore")
    return ep, np.asarray(labels)


def _pos2d(info):
    from mne.channels.layout import _find_topomap_coords
    pos = _find_topomap_coords(info, picks="eeg")          # sphere-projected 2D, the standard topo layout
    pos = pos - pos.mean(0)
    pos = pos / np.abs(pos).max()                          # normalize into [-1, 1]
    return pos


def _erd_frames(ep, labels, fmin, fmax, n_frames=N_FRAMES, baseline_s=BASELINE_S):
    """Time-resolved ERD per class: band-limited power over the trial, baseline-normalized to the first
    `baseline_s` (pre-imagery). Negative = event-related DESYNCHRONIZATION (the motor-imagery signature).
    Returns ({class: [frame][ch]}, frame_times) — averaged across epochs, downsampled to n_frames."""
    band = ep.copy().filter(fmin, fmax, verbose="error")
    sf = ep.info["sfreq"]
    X = band.get_data() * 1e6
    power = X ** 2                                          # [n_epochs, ch, t]
    T = power.shape[2]
    t = np.arange(T) / sf
    base_mask = t < baseline_s
    edges = np.linspace(0, T, n_frames + 1).astype(int)
    widths = np.diff(edges)                                 # samples per frame-bin (uneven; T need not divide)
    frames = {}
    for c in sorted(set(labels)):
        p = power[labels == c].mean(0)                     # [ch, t]
        base = p[:, base_mask].mean(1, keepdims=True) + 1e-20
        erd = (p - base) / base                            # ERD ratio per channel per time
        binned = np.add.reduceat(erd, edges[:-1], axis=1) / widths   # [ch, n_frames] mean per bin
        frames[str(c)] = binned.T.tolist()                 # -> [n_frames][ch]
    ftimes = ((edges[:-1] + edges[1:]) / 2 / sf).tolist()
    return frames, ftimes


def _csp_patterns(ep, labels, n=N_CSP):
    from mne.decoding import CSP
    X = ep.get_data() * 1e6
    csp = CSP(n_components=n, reg="ledoit_wolf", log=True)
    csp.fit(X.astype(np.float64), labels)
    pat = csp.patterns_                                    # [n_ch, n_components] (mne convention)
    pat = np.asarray(pat)[:, :n].T                         # -> [n_components, n_ch]
    return [(row / (np.abs(row).max() + 1e-9)).tolist() for row in pat]


def _riemann_patterns(ep, labels):
    """Per-class Riemannian discriminant channel weights — what the tangent-space classifier learned.

    Fits the tangent-space + logistic-regression baseline (baselines/eeg/riemann.py), then reads each class's
    weight vector over the (whitened-log) covariance entries and keeps its DIAGONAL: per-channel weight =
    how much that channel's own power drives the logit for the class. Parallel to CSP patterns, but here
    the feature is the covariance itself (no spatial filters). Returns {class: [w per channel]}, normalized.
    """
    from baselines.eeg.riemann import TangentSpace
    X = ep.get_data() * 1e6
    clf = TangentSpace().fit(X.astype(np.float64), np.asarray(labels))
    lr = clf.pipe_.named_steps["logisticregression"]
    coef = np.atleast_2d(lr.coef_)                         # [n_class, n_tri] over upper-triangular cov entries
    classes = [str(c) for c in lr.classes_]
    if coef.shape[0] == 1 and len(classes) == 2:           # binary LR: one row = class[1] vs class[0]
        coef = np.vstack([-coef[0], coef[0]])
    n_ch = len(ep.ch_names)
    iu = np.triu_indices(n_ch)                             # pyriemann vectorization order (row-major upper)
    diag = iu[0] == iu[1]                                  # the diagonal (per-channel) entries
    out = {}
    for c, row in zip(classes, coef):
        w = np.asarray(row)[diag]
        out[c] = (w / (np.abs(w).max() + 1e-9)).tolist()
    return out


def _waveforms(ep, labels, per_class=PER_CLASS, n_t=N_WAVE_T):
    """One example trial per class, ALL channels (downsampled to ~n_t points for display).
    The viewer colors each channel by its contribution to the selected view — no hardcoded highlight."""
    names = list(ep.ch_names)
    X = ep.get_data() * 1e6                                # [n, ch, t] microvolts
    T = X.shape[2]
    step = max(1, T // n_t)
    ti = np.arange(0, T, step)
    t = (ti / ep.info["sfreq"]).tolist()
    out = {}
    for c in sorted(set(labels)):
        ei = np.where(labels == c)[0][:per_class]
        out[str(c)] = {names[i]: X[ei[0], i, ti].tolist() for i in range(len(names))}
    return {"t": t, "trials": out, "chans": names}


def _predictions(subject: int):
    """Honest per-trial output: train CSP+LDA on the OTHER subjects (LOSO), predict THIS subject's trials.
    Returns ({class: {truth, pred, probs, correct}} for a shown example trial) + the subject's fold accuracy."""
    import polars as pl

    from baselines.eeg import csp_lda
    from core.data import store
    from core.data.eeg.base import CANONICAL_MI_NAMES, EpochCfg

    meta = store.Store.load("bnci2014_001", EpochCfg())
    Xtr, ytr = store.Store.gather(meta.filter(pl.col("subject") != str(subject)))
    Xte, yte = store.Store.gather(meta.filter(pl.col("subject") == str(subject)))
    clf = csp_lda.fit(Xtr, ytr)
    probs = np.asarray(csp_lda.score(clf, Xte))
    pred = probs.argmax(1)
    per = {}
    for c in sorted(np.unique(yte).tolist()):
        i = int((yte == c).argmax())
        per[CANONICAL_MI_NAMES[c]] = {"truth": CANONICAL_MI_NAMES[c], "pred": CANONICAL_MI_NAMES[int(pred[i])],
                                      "probs": [round(float(p), 3) for p in probs[i]],
                                      "correct": bool(pred[i] == c)}
    score = {"acc": round(float((pred == yte).mean()), 3), "chance": 0.25,
             "regime": "cross-subject (LOSO)", "decoder": "CSP+LDA"}
    return per, score


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--subject", type=int, default=1)
    ap.add_argument("--out", default="neuroviz/web/data")
    args = ap.parse_args()

    ep, labels = _load_epochs(args.subject)
    print(f"subject {args.subject}: {len(ep)} epochs, {len(ep.ch_names)} ch, classes {sorted(set(labels))}")

    mu_fr, ftimes = _erd_frames(ep, labels, *MU)
    beta_fr, _ = _erd_frames(ep, labels, *BETA)
    data = {
        "subject": str(args.subject),
        "sfreq": float(ep.info["sfreq"]),
        "channels": list(ep.ch_names),
        "pos": _pos2d(ep.info).tolist(),
        "classes": [str(c) for c in sorted(set(labels))],
        "frames": {"mu": mu_fr, "beta": beta_fr},
        "frame_times": ftimes,
        "csp_patterns": _csp_patterns(ep, labels),
        "riemann_patterns": _riemann_patterns(ep, labels),
        "waveforms": _waveforms(ep, labels),
    }
    per, score = _predictions(args.subject)
    data["predictions"] = per                              # ground truth vs decoder prediction (per shown trial)
    data["score"] = score                                  # the honest cross-subject decoder accuracy
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"subject{args.subject}.json").write_text(json.dumps(data))
    # modality-aware manifest (the viewer's EEG/fNIRS switch reads it); preserve any fNIRS entry
    eeg_subs = sorted(int(p.stem.replace("subject", "")) for p in out.glob("subject*.json")
                      if not p.stem.startswith("fnirs"))
    mpath = out / "manifest.json"
    man = json.loads(mpath.read_text()) if mpath.exists() else {}
    if "modalities" not in man:                            # migrate legacy {subjects:[...]}
        man = {"modalities": {}}
    man["modalities"]["eeg"] = eeg_subs
    mpath.write_text(json.dumps(man))
    print(f"-> {out}/subject{args.subject}.json  (+ manifest)")


if __name__ == "__main__":
    main()
