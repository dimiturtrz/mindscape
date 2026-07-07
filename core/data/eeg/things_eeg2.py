"""THINGS-EEG2 (Gifford et al. 2022) — the EEG->image visual-semantic spine (Stage 3).

10 subjects viewing THINGS images under RSVP (100 ms + 100 ms blank), with repeats (train 4x, test 80x) ->
high SNR, which is why it's the field's reconstruction benchmark. Clean, disjoint split by design:
    train: 1,654 concepts x 10 exemplars x 4 reps
    test:  200 concepts x 1 exemplar x 80 reps   (concepts disjoint from train)
Labels are THINGS concepts (a curated 1,854-concept hierarchy with human-similarity data) -> map cleanly to
CLIP / semantic embeddings. See the dataset landscape + the honest split plan (image-disjoint + cross-subject
LOSO; cross-dataset vs THINGS-EEG1 concept-level) in
`research/deep_dives/2026-07-07_eeg_visual_semantic_decoding_datasets.md`.

THINGS-EEG2 on OSF (`3jk45`) is split into components; the raw EEG (`crxs4`) and images (`y63gw`) live on
external add-ons (figshare / Google Drive), which osfclient can't clone — but OSF's WaterButler serves each
file by download-link, so we enumerate the component's files via the OSF API and stream them. Raw is ~110 GB
(10 subjects x ~11 GB zips), so `download()` streams + unzips per subject and skips any already present
(resumable). Run once, in the background.

Fetch (needs the `data` extra -> `uv sync --extra data`):
    python -c "from core.data.eeg.things_eeg2 import download; download()"   # raw EEG (~110 GB) + image set
Lands in `<data>/raw/things_eeg2/{raw/sub-XX/, images/}`. The decoder/get_data is deferred until the split
design lands (bd qoa) — this module currently provides the fetch + an on-disk subject index only, so it is
NOT yet registered.
"""
from __future__ import annotations

import json
import urllib.request
import zipfile

import numpy as np
import polars as pl

from core.config import raw_dir

_ROOT = "things_eeg2"
_N_EEG = 63                # 63 EEG + 1 trailing 'stim' trigger channel (raw_eeg_data is [64, T] @ 1000 Hz)
_FS_RAW = 1000.0
_RAW_NODE = "crxs4"      # "Raw EEG data" component — sub-01.zip..sub-10.zip on the figshare add-on
_RAW_PROVIDER = "figshare"
_IMG_NODE = "y63gw"      # "Image set" component — osfstorage, ~0.66 GB
_API = "https://api.osf.io/v2/nodes/{node}/files/{prov}/?page[size]=100"


def _provider_files(node: str, prov: str) -> list[tuple[str, str]]:
    """[(filename, download_url)] for a component's storage provider, via the OSF API (public -> no auth)."""
    out: list[tuple[str, str]] = []
    url = _API.format(node=node, prov=prov)
    while url:
        d = json.load(urllib.request.urlopen(url, timeout=60))
        for f in d["data"]:
            a = f["attributes"]
            if a["kind"] == "file":
                out.append((a["name"], f["links"]["download"]))
        url = d["links"].get("next")
    return out


def _stream(url: str, dest, chunk: int = 1 << 20) -> None:
    """Stream a URL to `dest` (atomic via .part), following OSF's redirect to files.osf.io."""
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(url, timeout=120) as r, open(tmp, "wb") as fh:
        while True:
            b = r.read(chunk)
            if not b:
                break
            fh.write(b)
    tmp.rename(dest)


def download(*, raw: bool = True) -> None:
    """Fetch THINGS-EEG2 raw EEG (~110 GB) + the image set into `<data>/raw/things_eeg2/`.

    Streams each subject zip from OSF (figshare-backed) and unzips it; skips subjects already extracted, so a
    killed run resumes cleanly. Large — run in the background.
    """
    base = raw_dir() / _ROOT
    rawdir = base / "raw"
    rawdir.mkdir(parents=True, exist_ok=True)

    # --- raw EEG: 10 per-subject zips ---
    for name, url in sorted(_provider_files(_RAW_NODE, _RAW_PROVIDER)):
        sub = name.replace(".zip", "")
        if (rawdir / sub).is_dir():
            print(f"[things_eeg2] {sub} already present — skip")
            continue
        zpath = rawdir / name
        print(f"[things_eeg2] downloading {name} ...")
        _stream(url, zpath)
        print(f"[things_eeg2] extracting {name} ...")
        with zipfile.ZipFile(zpath) as z:
            z.extractall(rawdir)
        zpath.unlink()
        print(f"[things_eeg2] {sub} done")

    # --- image set (osfstorage, ~0.66 GB) ---
    imgdir = base / "images"
    imgdir.mkdir(parents=True, exist_ok=True)
    for name, url in _provider_files(_IMG_NODE, "osfstorage"):
        dest = imgdir / name
        if dest.exists():
            continue
        print(f"[things_eeg2] image file {name} ...")
        _stream(url, dest)
        if name.endswith(".zip"):
            with zipfile.ZipFile(dest) as z:
                z.extractall(imgdir)
    print(f"[things_eeg2] all done -> {base}")


def _index() -> dict[int, object]:
    """{subject int -> its dir}, discovered on disk under `<data>/raw/things_eeg2/raw/` (naming-robust)."""
    import re

    out: dict[int, object] = {}
    for d in sorted((raw_dir() / _ROOT / "raw").glob("sub-*")):
        m = re.search(r"sub-0*(\d+)", d.name)
        if m and d.is_dir():
            out[int(m.group(1))] = d
    return out


def subjects() -> list[int]:
    return sorted(_index())


# ── image-metadata mapping (global; the stim code is a 1-based index into these ordered lists) ──
_META = None


def _meta() -> dict:
    global _META
    if _META is None:
        _META = np.load(raw_dir() / _ROOT / "images" / "image_metadata.npy", allow_pickle=True).item()
    return _META


def _concept_idx(concept_str: str) -> int:
    """'00001_aardvark' -> 0 — the leading number IS the 1-based concept id (matches clip_targets' sort)."""
    return int(str(concept_str)[:5]) - 1


def _labels_for(split: str) -> tuple[np.ndarray, np.ndarray]:
    """(concept_idx per code, img_file per code) as arrays indexed by code-1, for 'training' | 'test'."""
    m = _meta()
    key = "train" if split == "training" else "test"
    concepts = np.asarray(m[f"{key}_img_concepts"])
    files = np.asarray(m[f"{key}_img_files"])
    cidx = np.array([_concept_idx(c) for c in concepts], dtype=np.int64)
    return cidx, files


# ── epoching ──

def _session_epochs(path, split: str, tmin: float, tmax: float, resample: float,
                    fmin: float | None, fmax: float | None):
    """One session .npy -> (X [n,63,t] float32, concept[n], img_file[n]). Stim code -> image via metadata."""
    d = np.load(path, allow_pickle=True).item()
    raw = np.asarray(d["raw_eeg_data"])
    eeg, stim = raw[:_N_EEG], raw[_N_EEG]
    fs = float(np.asarray(d["sfreq"]))

    if fmin is not None or fmax is not None:
        from core.data.signal import bandpass
        eeg = bandpass(eeg, fmin or 0.1, fmax or (fs / 2 - 1), fs)

    onset = np.where((stim[1:] != 0) & (stim[:-1] == 0))[0] + 1
    codes = stim[onset].astype(int)                                  # 1-based image id
    cidx_map, files_map = _labels_for(split)
    valid = (codes >= 1) & (codes <= len(cidx_map))                 # drop target/catch trials out of range
    onset, codes = onset[valid], codes[valid]

    a, b = int(round(tmin * fs)), int(round(tmax * fs))
    keep = (onset + a >= 0) & (onset + b <= eeg.shape[1])
    onset, codes = onset[keep], codes[keep]

    X = np.stack([eeg[:, o + a:o + b] for o in onset]).astype(np.float32)   # [n,63,t]
    # per-channel z-score: EEG is in volts (~1e-5), which leaves BatchNorm's running variance
    # ill-conditioned -> eval-mode embeddings collapse to chance. Standardizing to O(1) fixes it.
    X = (X - X.mean(axis=2, keepdims=True)) / (X.std(axis=2, keepdims=True) + 1e-7)
    if resample and resample != fs:
        from scipy.signal import resample as _rs
        X = _rs(X, int(round(X.shape[2] * resample / fs)), axis=2).astype(np.float32)
    return X, cidx_map[codes - 1], files_map[codes - 1]


def get_epochs(subjects_: list[int] | None = None, *, split: str = "training",
               tmin: float = 0.0, tmax: float = 1.0, resample: float = 250.0,
               fmin: float | None = None, fmax: float | None = None
               ) -> tuple[np.ndarray, np.ndarray, np.ndarray, pl.DataFrame]:
    """Epoch THINGS-EEG2 for the EEG->image task (our own preprocessing off the raw).

    Returns (X [n,63,t] float32, concept [n] int in [0,1653]/[0,199], img_file [n] str, meta {subject,session}).
    `split` = 'training' (16,540 concepts-images) or 'test' (200 held-out concepts). The concept index matches
    `neuroscan/tasks/visual/clip_targets.py` (sorted concept order), so retrieval lines up.
    """
    idx = _index()
    subs = subjects_ or sorted(idx)
    Xs, cs, fsl, subj, sess = [], [], [], [], []
    for sub in subs:
        for spath in sorted(idx[sub].glob(f"ses-*/raw_eeg_{split}.npy")):
            X, c, f = _session_epochs(spath, split, tmin, tmax, resample, fmin, fmax)
            Xs.append(X); cs.append(c); fsl.append(f)
            n = len(c)
            subj += [str(sub)] * n
            sess += [spath.parent.name] * n
    X = np.concatenate(Xs).astype(np.float32)
    return (X, np.concatenate(cs), np.concatenate(fsl),
            pl.DataFrame({"subject": subj, "session": sess}))


# NOTE: not registered in core/data/registry.py — the EEG->image retrieval paradigm has its own
# (X, concept, image) signature + runner (neuroscan/tasks/visual/), distinct from the MI/workload class-fold
# harness. Registering here would imply the class-decoding contract, which this isn't.
