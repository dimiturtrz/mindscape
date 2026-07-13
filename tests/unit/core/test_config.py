"""Data-root guard — a Windows drive path on POSIX must fail loud, not leak into the repo."""
import os

import pytest

from core import config


def test_windows_drive_translates_to_mount_on_posix(monkeypatch):
    monkeypatch.setattr(os, "name", "posix")
    monkeypatch.setenv("MINDSCAPE_DATA", "X:/eeg/bnci")
    assert str(config.Config.data_root()) == "/mnt/x/eeg/bnci"


def test_mount_translates_to_drive_on_windows(monkeypatch):
    monkeypatch.setattr(os, "name", "nt")
    assert config.Config.to_native_path("/mnt/x/eeg/bnci").replace("\\", "/") == "X:/eeg/bnci"


def test_posix_path_unchanged_on_posix(monkeypatch):
    monkeypatch.setattr(os, "name", "posix")
    assert config.Config.to_native_path("/srv/eeg/bnci") == "/srv/eeg/bnci"


def test_windows_path_unchanged_on_windows(monkeypatch):
    monkeypatch.setattr(os, "name", "nt")
    assert config.Config.to_native_path("X:/eeg/bnci") == "X:/eeg/bnci"


def test_drive_colon_sanitizer_preserves_drive():
    # the MOABB-bug workaround: 'X:\\a\\b' must keep 'X:' (only the rest is sanitized)
    config.Config._patch_moabb_drive_colon()
    from moabb.datasets import download as dl
    out = str(dl._sanitize_path("X:\\eeg\\bnci\\raw"))
    assert out.replace("/", "\\").startswith("X:\\")


# ── experiment registry ──────────────────────────────────────────────────────

def test_experiment_names_nonempty_and_sorted():
    names = config.Config.experiment_names()
    assert names and names == sorted(names)
    assert "mi_csp_within" in names


def test_load_experiment_resolves_registry_entry():
    exp = config.Config.load_experiment("nback_eeg_riemann_cross")
    assert exp.task == "decode" and exp.method == "riemann" and exp.regime == "cross_subject"
    assert exp.recipe["fmin"] == 4 and exp.recipe["resample"] == 100.0


def test_load_experiment_applies_dotlist_overrides():
    # --set feeds an OmegaConf dotlist: base config merges the override, leaving the file untouched
    exp = config.Config.load_experiment("mi_csp_within", ["method=riemann", "recipe.resample=250"])
    assert exp.method == "riemann" and exp.recipe["resample"] == 250
    assert config.Config.load_experiment("mi_csp_within").method == "csp_lda"   # base unchanged


def test_load_experiment_unknown_name_lists_options():
    with pytest.raises(SystemExit) as ei:
        config.Config.load_experiment("does_not_exist")
    assert "does_not_exist" in str(ei.value)
