"""Data-root guard — a Windows drive path on POSIX must fail loud, not leak into the repo."""
import os

import pytest

from core import config


def test_windows_drive_translates_to_mount_on_posix(monkeypatch):
    monkeypatch.setattr(os, "name", "posix")
    monkeypatch.setenv("MINDSCAPE_DATA", "X:/eeg/bnci")
    assert str(config.data_root()) == "/mnt/x/eeg/bnci"


def test_mount_translates_to_drive_on_windows(monkeypatch):
    monkeypatch.setattr(os, "name", "nt")
    assert config.to_native_path("/mnt/x/eeg/bnci").replace("\\", "/") == "X:/eeg/bnci"


def test_posix_path_unchanged_on_posix(monkeypatch):
    monkeypatch.setattr(os, "name", "posix")
    assert config.to_native_path("/srv/eeg/bnci") == "/srv/eeg/bnci"


def test_windows_path_unchanged_on_windows(monkeypatch):
    monkeypatch.setattr(os, "name", "nt")
    assert config.to_native_path("X:/eeg/bnci") == "X:/eeg/bnci"


def test_drive_colon_sanitizer_preserves_drive():
    # the MOABB-bug workaround: 'X:\\a\\b' must keep 'X:' (only the rest is sanitized)
    config._patch_moabb_drive_colon()
    from moabb.datasets import download as dl
    out = str(dl._sanitize_path("X:\\eeg\\bnci\\raw"))
    assert out.replace("/", "\\").startswith("X:\\")
