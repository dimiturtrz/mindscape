"""Model profiler returns sane params/FLOPs."""
import pytest

pytest.importorskip("braindecode")


def test_profile_reports_params():
    from braindecode.models import EEGNetv4

    from neuroscan.models import profile
    r = profile.Profile.profile(EEGNetv4)          # takes the class, labels by cls.__name__
    assert r["model"] == "EEGNetv4"
    assert r["params"] > 0
    # FLOPs is None if fvcore absent, else a positive count
    assert r["flops"] is None or r["flops"] > 0


def test_fmt_units():
    from neuroscan.models import profile
    assert profile.Profile._fmt(3700) == "3.70K"
    assert profile.Profile._fmt(2_790_000) == "2.79M"
    assert profile.Profile._fmt(None) == "—"
