"""README number injection (`sync_numbers`) — the pure marker/expression logic, no README/results files.

`_lookup` reads a `<run>.<field>` from a runs dict (raising on bad term / missing run / missing field),
`_render` formats a single value or a signed `a-b` difference, `_markers` extracts every marker from text.
"""
import pytest

from neuroscan.evaluation.sync_numbers import _README, _SKIP_DIRS, _doc_files, _lookup, _markers, _render

_RUNS = {"csp_lda": {"acc": 0.598, "kappa": 0.464}, "eegnet": {"acc": 0.512}}


def test_doc_files_root_first_and_skips_vendored():
    files = _doc_files()
    assert files[0] == _README                                  # landing page synced first
    assert all(f.name == "README.md" for f in files)            # only READMEs
    assert not any(_SKIP_DIRS & set(f.parts) for f in files[1:])  # no .venv/external/generated trees


def test_render_single_value():
    assert _render(_RUNS, "csp_lda.acc") == "0.598"


def test_render_signed_difference():
    assert _render(_RUNS, "csp_lda.acc-eegnet.acc") == "+0.086"    # within->cross gap, unicode-safe sign
    assert _render(_RUNS, "eegnet.acc-csp_lda.acc").startswith("−")   # negative -> unicode minus


def test_lookup_raises_on_bad_term_missing_run_and_field():
    with pytest.raises(KeyError):
        _lookup(_RUNS, "no_dot_field")
    with pytest.raises(KeyError):
        _lookup(_RUNS, "ghost.acc")
    with pytest.raises(KeyError):
        _lookup(_RUNS, "csp_lda.ece")                       # run exists, field absent


def test_markers_extracts_expr_current_and_rendered():
    text = "acc **<!--r:csp_lda.acc-->0.111<!--/r-->** end"
    marks = _markers(_RUNS, text)
    assert len(marks) == 1
    _full, expr, current, rendered = marks[0]
    assert expr == "csp_lda.acc" and current == "0.111" and rendered == "0.598"
