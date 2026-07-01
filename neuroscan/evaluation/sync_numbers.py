"""Inject canonical numbers from results.json into the README, so prose can't drift from measurement.

Wrap any run-derived number in the README with a marker:

    | within-subject | **<!--r:csp_lda_within_bnci2014_001.acc-->0.598<!--/r-->** | ...

The text between the markers is REGENERATED from results.json on every sync; the comments render invisibly
on GitHub. An expression is either `<run_name>.<field>` (field = acc|kappa|ece, plus fusion per-role fields
eeg|fnirs|late|feature) or a difference of two such terms `a.acc-b.acc` (for the within→cross gap),
formatted with an explicit signed unicode minus.

    uv run python -m neuroscan.evaluation.sync_numbers            # rewrite README in place
    uv run python -m neuroscan.evaluation.sync_numbers --check    # exit 1 if anything is stale (CI gate)

Numbers that are NOT run outputs — FLOPs / params / latency (profiler) and the published literature
ceilings — stay hand-authored and unmarked; this only governs measured accuracy/kappa/ece cells.
"""
from __future__ import annotations

import argparse
import json
import re
import sys

from core.config import REPO

_RESULTS = REPO / "results.json"
_README = REPO / "README.md"
_DP = 3          # decimals shown in the README (snapshot keeps more; see results._PRECISION)

_MARKER = re.compile(r"<!--r:([^>]+?)-->(.*?)<!--/r-->")
_TERM = re.compile(r"^([\w.]+?)\.([a-z_]+)$")     # <run>.<field>; field validated by presence in the row


def _lookup(runs: dict, term: str) -> float:
    m = _TERM.match(term.strip())
    if not m:
        raise KeyError(f"bad term {term!r} (want <run_name>.acc|kappa|ece)")
    name, field = m.groups()
    if name not in runs:
        raise KeyError(f"no run {name!r} in results.json")
    v = runs[name].get(field)
    if v is None:
        raise KeyError(f"run {name!r} has no {field}")
    return float(v)


def _render(runs: dict, expr: str) -> str:
    expr = expr.strip()
    if "-" in expr and not expr.startswith("-"):          # a.field-b.field -> signed within→cross gap
        a, b = expr.split("-", 1)
        gap = _lookup(runs, a) - _lookup(runs, b)
        return f"{'−' if gap < 0 else '+'}{abs(gap):.{_DP}f}"     # unicode minus, matches README
    return f"{_lookup(runs, expr):.{_DP}f}"


def _markers(runs: dict, text: str) -> list[tuple[str, str, str, str]]:
    """(full_match, expr, current_text, rendered) for every marker in document order."""
    return [(m[0], m[1], m[2], _render(runs, m[1])) for m in _MARKER.finditer(text)]


def sync(check: bool = False) -> int:
    runs = json.loads(_RESULTS.read_text())["runs"]
    text = _README.read_text(encoding="utf-8")
    marks = _markers(runs, text)
    stale = [f"{expr}: {cur!r} -> {new!r}" for _old, expr, cur, new in marks if cur != new]
    if check:
        for s in stale:
            print(f"  {s}")
        print(f"{'STALE' if stale else 'ok'} — {len(stale)}/{len(marks)} marker(s) out of sync")
        return 1 if stale else 0
    for old, expr, _cur, new in marks:
        text = text.replace(old, f"<!--r:{expr}-->{new}<!--/r-->", 1)
    _README.write_text(text, encoding="utf-8")
    print(f"synced {len(marks)} marker(s); updated {len(stale)}")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="report staleness, don't write (CI gate)")
    sys.exit(sync(check=ap.parse_args().check))
