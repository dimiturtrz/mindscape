"""Consolidated epoch store: every dataset's epochs + metadata as one homogeneous, query-able thing.

Mirrors the siblings' store: turn the raw adapters into a cached, common-schema cloud.

    processed/<dataset>/<epochkey>/
        data/sub<N>.npz   # X [n,ch,t] float32, y [n] canonical int, session [n], run [n]
        meta.csv          # one row per EPOCH, common schema (read with polars)

`epochkey` = the EpochCfg recipe (band/window/resample) so two recipes never collide. Each processed
dataset is self-contained: concatenating the meta.csv of the datasets you ask for *is* the data cloud.
Splits are queries over it (see data/splits.py); `gather` pulls the actual X/y for a split.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import polars as pl

from core.config import processed_dir
from core.data.eeg.base import EpochCfg
from core.data.registry import get_adapter

logger = logging.getLogger(__name__)

_SCHEMA = {
    "dataset": pl.Utf8, "subject": pl.Utf8, "session": pl.Utf8, "run": pl.Utf8,
    "label_id": pl.Int64, "label": pl.Utf8, "epoch": pl.Int64, "file": pl.Utf8,
}


def dataset_dir(name: str, cfg: EpochCfg) -> Path:
    return processed_dir() / name / cfg.key()


def _rows_for_subject(name: str, sub: int, npz: Path, label_names: dict[int, str]) -> list[dict]:
    """Build per-epoch meta rows from a subject npz WITHOUT loading X (npz is lazy per-key).
    `label_names` (id->name) comes from the dataset's own adapter, so the label column is correct for
    any modality (MI classes, n-back load levels, …) — no hardcoded convention."""
    z = np.load(npz, allow_pickle=True)
    y, sess, run = z["y"], z["session"], z["run"]
    return [{"dataset": name, "subject": str(sub), "session": str(sess[i]), "run": str(run[i]),
             "label_id": int(y[i]), "label": label_names.get(int(y[i]), str(int(y[i]))),
             "epoch": i, "file": npz.name} for i in range(len(y))]


def build(name: str, cfg: EpochCfg, *, rebuild: bool = False) -> Path:
    """Consolidate one dataset into processed/<name>/<epochkey>/ (per-subject npz + meta.csv).

    Process-if-missing: epochs each subject only if its npz is absent; re-emits meta.csv each call.
    """
    out = dataset_dir(name, cfg)
    data = out / "data"
    data.mkdir(parents=True, exist_ok=True)
    adapter = get_adapter(name)
    label_names = {v: k for k, v in adapter.label_map.items()}   # id -> name, per dataset

    # channel names are dataset-level metadata — persist them into processed/ so the cache is
    # self-describing (one format), instead of downstream readers reaching back into the raw files.
    names_fn = getattr(adapter, "channels", None)
    ch_names = names_fn() if callable(names_fn) else None
    if ch_names:
        (out / "channels.json").write_text(json.dumps(list(ch_names)))

    rows: list[dict] = []
    for sub in adapter.subjects():
        npz = data / f"sub{sub}.npz"
        if rebuild or not npz.exists():
            X, y, m = adapter.get_data([sub], cfg)
            np.savez_compressed(npz, X=X, y=y,
                                session=m["session"].to_numpy(), run=m["run"].to_numpy())
        rows.extend(_rows_for_subject(name, sub, npz, label_names))

    pl.DataFrame(rows, schema=_SCHEMA).write_csv(out / "meta.csv")
    return out


def load(names: list[str] | str, cfg: EpochCfg) -> pl.DataFrame:
    """Ensure each dataset is consolidated, then return ONE polars frame over all of them (the cloud
    for this recipe). Adds an absolute `path` column pointing at each subject npz."""
    names = [names] if isinstance(names, str) else list(names)
    frames = []
    for name in names:
        out = dataset_dir(name, cfg)
        if not (out / "meta.csv").exists():
            build(name, cfg)
        df = pl.read_csv(out / "meta.csv", schema_overrides=_SCHEMA)
        df = df.with_columns((pl.lit(str(out / "data")) + "/" + pl.col("file")).alias("path"))
        frames.append(df)
    return pl.concat(frames, how="vertical_relaxed")


def channels(name: str, cfg: EpochCfg) -> list[str] | None:
    """The dataset's channel names, from the processed cache (built by `build`). None if the adapter
    doesn't expose them. Ensures the dataset is consolidated first, so the channels.json exists."""
    out = dataset_dir(name, cfg)
    if not (out / "meta.csv").exists():
        build(name, cfg)
    p = out / "channels.json"
    return json.loads(p.read_text()) if p.exists() else None


def gather(df: pl.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Pull the actual epochs for a split frame -> (X [n,ch,t], y [n]), in the frame's row order.
    Groups by subject npz so each file is read once."""
    if df.is_empty():
        raise ValueError("gather() got an empty split")
    d = df.with_row_index("_i")
    parts = []
    for (path,), g in d.group_by(["path"], maintain_order=True):
        z = np.load(path, allow_pickle=True)
        idx = g["epoch"].to_numpy()
        parts.append((g["_i"].to_numpy(), z["X"][idx], g["label_id"].to_numpy()))

    n = len(d)
    tail = parts[0][1].shape[1:]
    X = np.empty((n, *tail), dtype=np.float32)
    y = np.empty(n, dtype=np.int64)
    for i, Xp, yp in parts:
        X[i] = Xp
        y[i] = yp
    return X, y


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    for lib_name in ("mne", "moabb", "braindecode"):
        logging.getLogger(lib_name).setLevel(logging.WARNING)
    import argparse

    ap = argparse.ArgumentParser(description="consolidate a dataset into processed/<ds>/<epochkey>/")
    ap.add_argument("--name", default="bnci2014_001")
    args = ap.parse_args()
    df = load(args.name, EpochCfg())
    logger.info(f"\n=== cloud: {len(df)} epochs over {df['subject'].n_unique()} subjects ===")
    logger.info(df.group_by("dataset", "label").agg(pl.len().alias("n")).sort("dataset", "label"))
    logger.info(df.group_by("subject").agg(pl.len().alias("n")).sort("subject"))
