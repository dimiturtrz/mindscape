"""Reference-ceiling lookup is wired and reads sensibly."""
from core.reference import Reference


def test_ceilings_known_dataset():
    cs = Reference.ceilings("bnci2014_001", "within_subject")
    assert "atcnet" in cs and "sota" in cs
    assert 0 < cs["atcnet"]["acc"] <= 1.0


def test_ceilings_unknown_is_empty():
    assert Reference.ceilings("nope", "within_subject") == {}


def test_compare_reports_gap_and_source():
    s = Reference.compare(0.60, "bnci2014_001", "within_subject", method="fbcsp")
    assert "fbcsp" in s and "Ang" in s


def test_compare_no_reference_recorded():
    s = Reference.compare(0.60, "nope", "within_subject")
    assert "no reference recorded" in s and "0.600" in s
