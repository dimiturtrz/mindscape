"""Inject canonical numbers from results.json into the README, so prose can't drift from measurement.

Wrap any run-derived number in the README with a marker:

    | within-subject | **<!--r:csp_lda_within_bnci2014_001.acc-->0.598<!--/r-->** | ...

The text between the markers is REGENERATED from results.json on every sync; the comments render invisibly
on GitHub. An expression is either `<run_name>.<field>` (field = acc|kappa|ece) or a difference of two such
terms `a.acc-b.acc` (for the within→cross gap), formatted with an explicit signed unicode minus.

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
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
_RESULTS = _ROOT / "results.json"
_README = _ROOT / "README.md"

_MARKER = re.compile(r"<!--r:([^>]+?)-->(.*?)<!--/r-->")
_TERM = re.compile(r"^([\w.]+?)\.(acc|kappa|ece)$")


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
    if "-" in expr and not expr.startswith("-"):
        a, b = expr.split("-", 1)
        gap = _lookup(runs, a) - _lookup(runs, b)
        return f"{'−' if gap < 0 else '+'}{abs(gap):.3f}"   # signed, unicode minus (matches README)
    return f"{_lookup(runs, expr):.3f}"


def sync(check: bool = False) -> int:
    runs = json.loads(_RESULTS.read_text())["runs"]
    text = _README.read_text(encoding="utf-8")
    stale: list[str] = []

    def repl(m: re.Match) -> str:
        expr, cur = m.group(1), m.group(2)
        new = _render(runs, expr)
        if new != cur:
            stale.append(f"{expr}: {cur!r} -> {new!r}")
        return f"<!--r:{expr}-->{new}<!--/r-->"

    new_text = _MARKER.sub(repl, text)
    n = len(_MARKER.findall(text))
    if check:
        if stale:
            print(f"STALE ({len(stale)}/{n} markers out of date):")
            for s in stale:
                print(f"  {s}")
            return 1
        print(f"ok — {n} marker(s) in sync")
        return 0
    if new_text != text:
        _README.write_text(new_text, encoding="utf-8")
    print(f"synced {n} marker(s); updated {len(stale)}")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="report staleness, don't write (CI gate)")
    sys.exit(sync(check=ap.parse_args().check))
