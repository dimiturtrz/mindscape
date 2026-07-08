"""One data root, everything derived — mirrors the siblings' core/config.py.

Source of truth: `paths.yaml` at the repo root (gitignored). Copy `paths.example.yaml` ->
`paths.yaml` and set the single `data:` line. Override with the env var MINDSCAPE_DATA (e.g. in CI).

Under <data> the layout is:
    <data>/raw/        MOABB/MNE download cache (you don't touch it; MOABB fills it)
    <data>/processed/  epoched preprocess cache (created on first run)

We also point MOABB/MNE at <data>/raw so downloads land inside the one root, not ~/mne_data.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path, PurePosixPath, PureWindowsPath

import mne
from moabb.datasets import download as _dl
from moabb.utils import set_download_dir
from omegaconf import OmegaConf
from pydantic import BaseModel

logger = logging.getLogger(__name__)

REPO = Path(__file__).resolve().parent.parent   # repo root — the one place that computes it

_DRIVE_PREFIX_LEN = 2   # a Windows drive prefix is 2 chars ("D:")
_MNT_MIN_PARTS = 3      # a WSL mount path "/mnt/<x>/..." has at least 3 path parts ("/", "mnt", "<x>")


# ─────────────────────────── experiment registry ───────────────────────────
# Named runs live in experiments.yaml (config-as-data, like reference.yaml); entrypoints take `--exp <name>`
# instead of a flag per parameter. See prefer-config-files-over-argv: argv sparingly.

class Experiment(BaseModel):
    """One named run from experiments.yaml. `recipe` -> EpochCfg/FnirsCfg kwargs; `params` = method knobs."""
    task: str                                    # decode | fnirs | align | fusion
    dataset: str | None = None
    method: str | None = None
    regime: str | None = None
    test_session: str | None = None
    recipe: dict = {}
    params: dict = {}


def _experiments_doc():
    path = REPO / "experiments.yaml"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found — the named-experiment registry (see experiments.yaml)")
    return OmegaConf.load(path)


def experiment_names() -> list[str]:
    """All registered experiment names (for --exp choices / error messages)."""
    return sorted(_experiments_doc().experiments.keys())


def load_experiment(name: str, overrides: list[str] | None = None) -> Experiment:
    """Resolve a named experiment, applying any `--set key=val` dotlist overrides (e.g. `recipe.fmin=4`).
    Unknown name -> a listing SystemExit, so the CLI fails helpfully instead of KeyError-ing."""
    doc = _experiments_doc()
    if name not in doc.experiments:
        raise SystemExit(f"unknown --exp {name!r}; known: {experiment_names()}")
    node = doc.experiments[name]
    if overrides:
        node = OmegaConf.merge(node, OmegaConf.from_dotlist(overrides))
    return Experiment(**OmegaConf.to_container(node, resolve=True))


def to_native_path(path_str: str) -> str:
    """Translate a configured path to the current platform so ONE paths.yaml works on Windows + WSL.

    Windows drive 'X:/...' <-> POSIX/WSL '/mnt/x/...'. A bare Windows drive path on POSIX would
    otherwise be treated as relative and leak the data tree into the repo; a POSIX
    mount path on Windows would be unresolvable. There's no solid pip lib for this (wslpath is WSL-only),
    so this small, tested mapping is the dependency-free fix. The default WSL mount is /mnt/<drive>;
    MINDSCAPE_DATA can always override with an explicit native path.
    """
    # Parse with PurePath (robust to '\\', mixed slashes, 'D:' w/o slash); only the /mnt mapping is
    # explicit (no lib does the WSL drive<->mount convention).
    drive = PureWindowsPath(path_str).drive              # 'D:' for a drive path, '' otherwise
    if len(drive) == _DRIVE_PREFIX_LEN and drive.endswith(":"):
        rest = "/".join(PureWindowsPath(path_str).parts[1:])
        if os.name == "nt":
            return f"{drive}/{rest}".rstrip("/")
        return f"/mnt/{drive[0].lower()}/{rest}".rstrip("/")
    parts = PurePosixPath(path_str).parts                # /mnt/<x>/... on Windows -> drive
    if os.name == "nt" and len(parts) >= _MNT_MIN_PARTS and parts[1] == "mnt" and len(parts[2]) == 1:
        return f"{parts[2].upper()}:/" + "/".join(parts[3:])
    return path_str


def data_root(sub: str | None = None) -> Path:
    """The single data root (platform-translated), or a named subdir under it (`raw` / `processed`)."""
    env = os.environ.get("MINDSCAPE_DATA")
    if env:
        raw = env
    else:
        cfg = REPO / "paths.yaml"
        if not cfg.exists():
            raise FileNotFoundError(
                f"{cfg} not found — copy paths.example.yaml -> paths.yaml and set `data:` "
                f"(or set the MINDSCAPE_DATA env var)."
            )
        raw = str(OmegaConf.load(cfg).data)
    root = Path(to_native_path(raw))
    return root / sub if sub else root


def raw_dir() -> Path:
    return data_root("raw")


def processed_dir() -> Path:
    return data_root("processed")


def configure_moabb_download() -> Path:
    """Point MOABB/MNE's download cache at <data>/raw so recordings stay inside the one root (never the
    repo). Idempotent; returns the cache dir. Call before any MOABB dataset access.

    The path is resolved to an ABSOLUTE native path and asserted — a relative value here is how raw
    downloads once leaked into the repo as a stray `D-/` dir (a drive-letter mangle). We also persist
    it to MNE's own config (set_config), not just the env, so a child process can't fall back."""
    cache = raw_dir().resolve()
    assert cache.is_absolute(), f"data root must be absolute, got {cache!r} (fix paths.yaml)"
    cache.mkdir(parents=True, exist_ok=True)
    native = os.fspath(cache)                          # native Windows path (no forward-slash mangling)
    os.environ["MNE_DATA"] = native                   # overwrite, don't setdefault — be authoritative
    os.environ["MOABB_RESULTS"] = os.fspath(processed_dir() / "moabb_results")
    try:
        mne.set_config("MNE_DATA", native, set_env=True)
    except OSError as exc:
        logger.debug(f"mne.set_config: {exc}")
    try:
        set_download_dir(native)
    except OSError as exc:
        logger.debug(f"moabb.set_download_dir: {exc}")
    if os.name == "nt":
        # ONLY native Windows needs this — a Windows absolute path has a drive colon that MOABB's
        # buggy sanitizer strips. Under WSL/POSIX the root is '/mnt/<drive>/...' (no colon) so MOABB works
        # unpatched; that's the zero-patch native path (see to_native_path).
        _patch_moabb_drive_colon()
    return cache


def _patch_moabb_drive_colon() -> None:
    """Compat shim for a MOABB *Windows* bug (no upstream fix as of 1.5.0; no config flag avoids it,
    and no colon-free absolute Windows path exists). MOABB's `_sanitize_path` translates ':' -> '-' over
    the WHOLE path, clobbering the drive colon ('<drive>:\\...' -> '<drive>-\\...'), so downloads go RELATIVE and
    leak into the repo cwd (the recurring `D-/`) + re-download every time. We restore a leading drive and
    sanitize only the rest (behavior-preserving otherwise). Idempotent. TODO: file/track upstream PR."""
    if getattr(_dl._sanitize_path, "_mindscape_patched", False):
        return
    _bad = ':*?"<>|'

    def _safe(path):
        s = str(path)
        if len(s) >= _DRIVE_PREFIX_LEN and s[1] == ":" and s[0].isalpha():        # 'D:...' -> keep 'D:', clean rest
            drive, rest = s[:2], s[2:]
            return Path(drive + rest.translate({ord(c): "-" for c in _bad}))
        return Path(s.translate({ord(c): "-" for c in _bad}))

    _safe._mindscape_patched = True
    _dl._sanitize_path = _safe
