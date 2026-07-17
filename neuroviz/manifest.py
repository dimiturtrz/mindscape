"""One home for writing a neuroviz view JSON + updating the modality-aware manifest.

The three subject-exporters (`export`, `export_fnirs`, `export_eeg_workload`) each write a
`<prefix><subject>.json` and register the subject under a modality key in `manifest.json` (the
viewer's EEG/fNIRS/workload switch reads it). Same write-json + read-or-init-manifest + set-modality
shape everywhere — extracted here so there is a single source of truth for the on-disk contract.
"""
from __future__ import annotations

import json
from pathlib import Path


class Manifest:
    """Neuroviz view-data writer: one subject JSON + the modality-aware manifest merge."""

    @staticmethod
    def publish(out: Path, subject: int, prefix: str, modality: str, data: dict) -> list[int]:
        """Write `<prefix><subject>.json` under `out`, then set `manifest.json`'s
        `modalities[modality]` to every subject exported for this prefix (globbed from disk, so
        re-running a single subject keeps the others). Returns the sorted subject list."""
        out.mkdir(parents=True, exist_ok=True)
        (out / f"{prefix}{subject}.json").write_text(json.dumps(data))
        subs = sorted(int(p.stem.removeprefix(prefix)) for p in out.glob(f"{prefix}*.json"))
        mpath = out / "manifest.json"
        man = json.loads(mpath.read_text()) if mpath.exists() else {"modalities": {}}
        man.setdefault("modalities", {})[modality] = subs
        mpath.write_text(json.dumps(man))
        return subs
