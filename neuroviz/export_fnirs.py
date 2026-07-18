"""neuroviz fNIRS export — the hemodynamic view (Shin n-back workload), same JSON schema the web viewer loads.

The fNIRS counterpart to export.py: animated **HbO/HbR topomaps per workload class** (watch the hemodynamic
response build over ~5-8 s), the prefrontal optode montage, example waveforms, and per-class **LDA channel
weights** (what the amplitude-feature decoder reads). Shares the EEG viewer's schema (channels/pos/classes/
frames/waveforms) so one web app renders both modalities.

    python -m neuroviz.export_fnirs --subject 1
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, cast

import numpy as np
import polars as pl
import scipy.io as sio
from jaxtyping import Float, Int
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from baselines.fnirs import features as ff
from core.config import Config
from core.data import store
from core.data.fnirs.base import FnirsCfg
from core.data.fnirs.shin2017 import Shin2017NirsAdapter
from core.features import Amplitude
from neuroviz.manifest import Manifest
from neuroviz.viewdata import Decode, ViewData

logger = logging.getLogger(__name__)

_BINARY = 2                      # binary-LR special case: one coef row covers the two classes
N_FRAMES = 200                   # animation frames ≈ native 10 Hz over the 22 s window (~30 fps at 3× speed)
CLASS_NAMES = {0: "0-back", 1: "2-back", 2: "3-back"}


def _subject_epochs(subject: int):
    """(X [n,72,t] HbO|HbR, y, ch_names[36], pos2d[36,2], fs) for one subject via the adapter + montage."""
    X, y, _ = Shin2017NirsAdapter.adapter("nback").get_data([subject], FnirsCfg(tmax=20.0))
    d = Config.raw_dir() / "shin2017" / f"VP{subject:03d}-NIRS"
    mnt = sio.loadmat(d / "mnt_nback.mat", struct_as_record=False, squeeze_me=True)["mnt_nback"]
    names = [str(c) for c in np.asarray(mnt.clab)][:36]
    pos = np.stack([np.asarray(mnt.x)[:36], np.asarray(mnt.y)[:36]], axis=1).astype(float)
    pos = pos - pos.mean(0)
    pos = pos / (np.abs(pos).max() + 1e-9)                  # normalize into [-1, 1]
    return X, y, names, pos, 10.0


def _frames(X: Float[np.ndarray, "n ch t"], y: Int[np.ndarray, "n"], chan_slice: slice,
            n_frames: int = N_FRAMES) -> tuple[dict[str, list[Any]], list[float]]:
    """Per-class time-resolved HbO (or HbR) topomap: mean over trials, downsampled to n_frames.
    Returns ({class: [frame][ch]}, frame_times) — the hemodynamic response building over the trial."""
    T = X.shape[2]
    edges = np.linspace(0, T, n_frames + 1).astype(int)
    widths = np.diff(edges)                                  # samples per frame-bin (uneven)
    frames: dict[str, list[Any]] = {}
    for c in sorted(np.unique(y).tolist()):
        m = X[y == c][:, chan_slice, :].mean(0)             # [36, t] mean HbO/HbR
        frames[CLASS_NAMES[c]] = (np.add.reduceat(m, edges[:-1], axis=1) / widths).T.tolist()
    ftimes = ((edges[:-1] + edges[1:]) / 2 / 10.0 - 2.0).tolist()   # tmin=-2 s
    return frames, ftimes


def _lda_patterns(X: Float[np.ndarray, "n ch t"], y: Int[np.ndarray, "n"]) -> dict[str, list[float]]:
    """Per-class LDA weight on the HbO MEAN feature (what the decoder reads), one value per channel."""
    lda = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto").fit(Amplitude.amplitude_features(X), y)
    coef = np.atleast_2d(lda.coef_)                         # [n_class, 216] (mean|slope|peak × 72)
    classes = sorted(np.unique(y).tolist())
    if coef.shape[0] == 1 and len(classes) == _BINARY:
        coef = np.vstack([-coef[0], coef[0]])
    out: dict[str, list[float]] = {}
    for c, row in zip(classes, coef, strict=True):
        w = np.asarray(row)[:36]                            # HbO mean-feature block
        out[CLASS_NAMES[c]] = (w / (np.abs(w).max() + 1e-9)).tolist()
    return out


def _waveforms(X: Float[np.ndarray, "n ch t"], y: Int[np.ndarray, "n"], names: list[str],
               n_t: int = 300) -> dict[str, Any]:
    """One example trial per class — BOTH chromophores per optode (the raw data): {chan:{hbo,hbr}}.
    HbO = channels 0..35, HbR = 36..71 at the same optodes; showing both reveals the anti-correlation."""
    T = X.shape[2]
    step = max(1, T // n_t)
    ti = np.arange(0, T, step)
    t = (ti / 10.0 - 2.0).tolist()
    out: dict[str, Any] = {}
    for c in sorted(np.unique(y).tolist()):
        ei = int((y == c).argmax())
        out[CLASS_NAMES[c]] = {names[i]: {"hbo": X[ei, i, ti].tolist(), "hbr": X[ei, i + 36, ti].tolist()}
                               for i in range(len(names))}
    return {"t": t, "trials": out, "chans": names}


def _predictions(subject: int, X: Float[np.ndarray, "n ch t"],
                 y: Int[np.ndarray, "n"]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Honest per-trial output: train the fNIRS decoder on the OTHER subjects (LOSO), predict THIS
    subject's trials. Returns ({class: {truth, pred, probs, correct}} for the shown example trial) and the
    subject's cross-subject fold accuracy — so the viewer shows ground truth vs prediction, not just signal."""
    meta = store.Store.load("shin2017_nback", cast(store.EpochCfg, FnirsCfg(tmax=20.0)))
    Xtr, ytr = store.Store.gather(meta.filter(pl.col("subject") != str(subject)))
    clf = ff.fit(Xtr, ytr)
    probs = ff.score(clf, X)
    pred = probs.argmax(1)
    return ViewData.prediction_report(CLASS_NAMES, Decode(y, pred, probs, 1 / 3, "fNIRS mean+slope+peak → LDA"))


def main():
    args = ViewData.subject_args(__doc__)
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    X, y, names, pos, fs = _subject_epochs(args.subject)
    logger.info(f"fNIRS subject {args.subject}: {len(X)} epochs, {len(names)} ch, "
                f"classes {sorted(np.unique(y).tolist())}")

    hbo, ftimes = _frames(X, y, slice(0, 36))
    data = {
        "modality": "fnirs",
        "subject": str(args.subject),
        "sfreq": fs,
        "channels": names,
        "pos": pos.tolist(),
        "classes": [CLASS_NAMES[c] for c in sorted(np.unique(y).tolist())],
        "frames": {"response": hbo},                       # HbO activation topomap (HbR lives in the waveforms)
        "frame_times": ftimes,
        "lda_patterns": _lda_patterns(X, y),               # decoder view (per-class HbO weight)
        "waveforms": _waveforms(X, y, names),
    }
    per, score = _predictions(args.subject, X, y)
    data["predictions"] = per                              # ground truth vs decoder prediction (per shown trial)
    data["score"] = score                                  # the honest cross-subject decoder accuracy
    out = Path(args.out)
    subs = Manifest.publish(out, args.subject, "fnirs_subject", "fnirs", data)
    logger.info(f"-> {out}/fnirs_subject{args.subject}.json  (+ manifest, fnirs subjects {subs})")


if __name__ == "__main__":
    main()
