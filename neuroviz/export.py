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


def _bandpower(ep, labels, fmin, fmax):
    """Per-class mean log band power per channel, z-scored across channels (so the spatial pattern,
    i.e. the ERD lateralization, is what's visible rather than absolute scale)."""
    psd = ep.compute_psd(fmin=fmin, fmax=fmax, verbose="error")
    p = psd.get_data().mean(axis=2)                        # [n_epochs, n_ch] mean power in band
    p = np.log(p + 1e-20)
    out = {}
    for c in sorted(set(labels)):
        v = p[labels == c].mean(0)
        out[str(c)] = ((v - v.mean()) / (v.std() + 1e-9)).tolist()
    return out


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

    data = {
        "subject": str(args.subject),
        "sfreq": float(ep.info["sfreq"]),
        "channels": list(ep.ch_names),
        "pos": _pos2d(ep.info).tolist(),
        "classes": [str(c) for c in sorted(set(labels))],
        "bandpower": {"mu": _bandpower(ep, labels, *MU), "beta": _bandpower(ep, labels, *BETA)},
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
