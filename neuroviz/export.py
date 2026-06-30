"""neuroviz data export — compute the field-standard motor-imagery views for one subject and write a
self-contained JSON the web viewer loads. The EEG analogue of the siblings' export_web (cardioview).

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

MU = (8.0, 12.0)
BETA = (13.0, 30.0)
KEY_CHANS = ["C3", "Cz", "C4"]


def _load_epochs(subject: int):
    """MNE Epochs for a subject (broad 4-40 Hz band, montage set) via MOABB."""
    import mne
    from moabb.datasets import BNCI2014_001
    from moabb.paradigms import MotorImagery

    from core.config import configure_moabb_download
    configure_moabb_download()
    para = MotorImagery(n_classes=4, fmin=4.0, fmax=40.0, tmin=0.0, tmax=None, resample=250.0)
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


def _erd_frames(ep, labels, fmin, fmax, n_frames=50, baseline_s=0.5):
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
    frames = {}
    for c in sorted(set(labels)):
        p = power[labels == c].mean(0)                     # [ch, t]
        base = p[:, base_mask].mean(1, keepdims=True) + 1e-20
        erd = (p - base) / base                            # ERD ratio per channel per time
        fr = [erd[:, edges[i]:edges[i + 1]].mean(1).tolist() for i in range(n_frames)]
        frames[str(c)] = fr
    ftimes = [float((edges[i] + edges[i + 1]) / 2 / sf) for i in range(n_frames)]
    return frames, ftimes


def _csp_patterns(ep, labels, n=4):
    from mne.decoding import CSP
    X = ep.get_data() * 1e6
    csp = CSP(n_components=n, reg="ledoit_wolf", log=True)
    csp.fit(X.astype(np.float64), labels)
    pat = csp.patterns_                                    # [n_ch, n_components] (mne convention)
    pat = np.asarray(pat)[:, :n].T                         # -> [n_components, n_ch]
    return [(row / (np.abs(row).max() + 1e-9)).tolist() for row in pat]


def _waveforms(ep, labels, chans, per_class=1):
    names = ep.ch_names
    idx = [names.index(c) for c in chans if c in names]
    X = ep.get_data() * 1e6                                # [n, ch, t] microvolts
    t = (np.arange(X.shape[2]) / ep.info["sfreq"]).tolist()
    out = {}
    for c in sorted(set(labels)):
        ei = np.where(labels == c)[0][:per_class]
        out[str(c)] = {names[i]: X[ei[0], i, :].tolist() for i in idx}
    return {"t": t, "trials": out, "chans": [names[i] for i in idx]}


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
        "waveforms": _waveforms(ep, labels, KEY_CHANS),
    }
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"subject{args.subject}.json").write_text(json.dumps(data))
    # a tiny manifest the viewer reads to list available subjects
    subs = sorted(int(p.stem.replace("subject", "")) for p in out.glob("subject*.json"))
    (out / "manifest.json").write_text(json.dumps({"subjects": subs}))
    print(f"-> {out}/subject{args.subject}.json  (+ manifest)")


if __name__ == "__main__":
    main()
