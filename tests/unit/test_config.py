"""Data-root guard — a Windows drive path on POSIX must fail loud, not leak into the repo."""
import os

import pytest

from core import config


def test_windows_drive_translates_to_mount_on_posix(monkeypatch):
    monkeypatch.setattr(os, "name", "posix")
    monkeypatch.setenv("MINDSCAPE_DATA", "D:/data/neural")
    assert str(config.data_root()) == "/mnt/d/data/neural"


def test_mount_translates_to_drive_on_windows(monkeypatch):
    monkeypatch.setattr(os, "name", "nt")
    assert config.to_native_path("/mnt/d/data/neural").replace("\\", "/") == "D:/data/neural"


def test_posix_path_unchanged_on_posix(monkeypatch):
    monkeypatch.setattr(os, "name", "posix")
    assert config.to_native_path("/data/neural") == "/data/neural"


def test_windows_path_unchanged_on_windows(monkeypatch):
    monkeypatch.setattr(os, "name", "nt")
    assert config.to_native_path("D:/data/neural") == "D:/data/neural"


def test_drive_colon_sanitizer_preserves_drive():
    # the MOABB-bug workaround: 'D:\\x\\y' must keep 'D:' (only the rest is sanitized)
    config._patch_moabb_drive_colon()
    from moabb.datasets import download as dl
    out = str(dl._sanitize_path("D:\\data\\neural\\raw"))
    assert out.replace("/", "\\").startswith("D:\\")
