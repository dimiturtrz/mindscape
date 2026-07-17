"""Manifest.publish — writes one view JSON + merges the modality-aware manifest."""
import json

from neuroviz.manifest import Manifest


def test_publish_writes_json_and_registers_subject(tmp_path):
    subs = Manifest.publish(tmp_path, 1, "subject", "eeg", {"hello": "world"})
    assert subs == [1]
    assert json.loads((tmp_path / "subject1.json").read_text()) == {"hello": "world"}
    man = json.loads((tmp_path / "manifest.json").read_text())
    assert man["modalities"]["eeg"] == [1]


def test_publish_globs_all_prior_subjects_of_the_prefix(tmp_path):
    Manifest.publish(tmp_path, 3, "subject", "eeg", {})
    subs = Manifest.publish(tmp_path, 1, "subject", "eeg", {})       # re-run keeps subject 3
    assert subs == [1, 3]
    man = json.loads((tmp_path / "manifest.json").read_text())
    assert man["modalities"]["eeg"] == [1, 3]


def test_publish_prefixes_are_disjoint_and_merge_into_one_manifest(tmp_path):
    Manifest.publish(tmp_path, 1, "subject", "eeg", {})
    Manifest.publish(tmp_path, 2, "fnirs_subject", "fnirs", {})      # must NOT be caught by the "subject" glob
    man = json.loads((tmp_path / "manifest.json").read_text())
    assert man["modalities"] == {"eeg": [1], "fnirs": [2]}
