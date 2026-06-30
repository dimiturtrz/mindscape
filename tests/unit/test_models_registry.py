"""Method registry — unified lookup for baseline + braindecode decoders."""
import pytest

from neuroscan import models


def test_method_names_include_baseline_and_nets():
    names = models.method_names()
    assert "csp_lda" in names and "riemann" in names
    assert "atcnet" in names and "eegnet" in names


def test_get_method_returns_fit_score_callables():
    for name in ("csp_lda", "riemann", "eegnet"):
        fit, score = models.get_method(name)
        assert callable(fit) and callable(score)


def test_unknown_method_raises():
    with pytest.raises(KeyError):
        models.get_method("nope")
