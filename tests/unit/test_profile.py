"""Model profiler returns sane params/FLOPs."""
import pytest

pytest.importorskip("braindecode")


def test_profile_reports_params():
    from neuroscan.models import profile
    r = profile.profile("EEGNetv4")
    assert r["model"] == "EEGNetv4"
    assert r["params"] > 0
    # FLOPs is None if fvcore absent, else a positive count
    assert r["flops"] is None or r["flops"] > 0


def test_fmt_units():
    from neuroscan.models import profile
    assert profile._fmt(3700) == "3.70K"
    assert profile._fmt(2_790_000) == "2.79M"
    assert profile._fmt(None) == "—"
