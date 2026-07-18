"""Inject canonical numbers from results.json into the README, so prose can't drift from measurement.

Wrap any run-derived number in the README with a marker:

    | within-subject | **<!--r:csp_lda_within_bnci2014_001.acc-->0.598<!--/r-->** | ...

The text between the markers is REGENERATED from results.json on every sync; the comments render invisibly
on GitHub. An expression is either `<run_name>.<field>` (field = acc|kappa|ece, plus fusion per-role fields
eeg|fnirs|late|feature) or a difference of two such terms `a.acc-b.acc` (for the within→cross gap),
formatted with an explicit signed unicode minus.

    uv run python -m neuroscan.evaluation.sync_numbers            # rewrite docs in place
    uv run python -m neuroscan.evaluation.sync_numbers --check    # exit 1 if anything is stale (CI gate)

Scans the root README **and every sub-`README.md`** (so a deep result table can live in the task README it
belongs to, not only the landing page) — vendored / generated trees are skipped. A file with no markers is
left untouched.

Numbers that are NOT run outputs — FLOPs / params / latency (profiler) and the published literature
ceilings — stay hand-authored and unmarked; this only governs measured accuracy/kappa/ece cells.
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from pathlib import Path
from typing import Any

from core.config import REPO
from neuroscan.tasks.cli import Cli

logger = logging.getLogger(__name__)

_RESULTS = REPO / "results.json"
_README = REPO / "README.md"
_DP = 3          # decimals shown in the README (snapshot keeps more; see results._PRECISION)
_SKIP_DIRS = {".venv", "external", "node_modules", ".git", "mlruns", "runs", ".pytest_cache", ".beads", "docs"}
_MARKER = re.compile(r"<!--r:([^>]+?)-->(.*?)<!--/r-->")
_TERM = re.compile(r"^([\w.]+?)\.([a-z_]+)$")     # <run>.<field>; field validated by presence in the row


class SyncNumbers:
    @staticmethod
    def _doc_files() -> list[Path]:
        """Root README first, then every sub-README.md outside vendored/generated trees."""
        subs = [p for p in sorted(REPO.rglob("README.md"))
                if p != _README and not _SKIP_DIRS & set(p.relative_to(REPO).parts)]
        return [_README, *subs]

    @staticmethod
    def _lookup(runs: dict[str, Any], term: str) -> float:
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

    @staticmethod
    def _render(runs: dict[str, Any], expr: str) -> str:
        expr = expr.strip()
        if "-" in expr and not expr.startswith("-"):          # a.field-b.field -> signed within→cross gap
            a, b = expr.split("-", 1)
            gap = SyncNumbers._lookup(runs, a) - SyncNumbers._lookup(runs, b)
            return f"{'−' if gap < 0 else '+'}{abs(gap):.{_DP}f}"     # unicode minus, matches README
        return f"{SyncNumbers._lookup(runs, expr):.{_DP}f}"

    @staticmethod
    def _markers(runs: dict[str, Any], text: str) -> list[tuple[str, str, str, str]]:
        """(full_match, expr, current_text, rendered) for every marker in document order."""
        return [(m[0], m[1], m[2], SyncNumbers._render(runs, m[1])) for m in _MARKER.finditer(text)]

    @staticmethod
    def sync(*, check: bool = False) -> int:
        runs = json.loads(_RESULTS.read_text())["runs"]
        total = stale_total = 0
        for path in SyncNumbers._doc_files():
            text = path.read_text(encoding="utf-8")
            marks = SyncNumbers._markers(runs, text)
            if not marks:
                continue
            stale = [(expr, cur, new) for _old, expr, cur, new in marks if cur != new]
            total += len(marks)
            stale_total += len(stale)
            rel = path.relative_to(REPO)
            if check:
                for expr, cur, new in stale:
                    logger.info(f"  {rel}: {expr}: {cur!r} -> {new!r}")
                continue
            for old, expr, _cur, new in marks:
                text = text.replace(old, f"<!--r:{expr}-->{new}<!--/r-->", 1)
            path.write_text(text, encoding="utf-8")
        if check:
            logger.info(f"{'STALE' if stale_total else 'ok'} — {stale_total}/{total} marker(s) out of sync")
            return 1 if stale_total else 0
        logger.info(f"synced {total} marker(s); updated {stale_total}")
        return 0


if __name__ == "__main__":
    Cli.setup_logging()
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="report staleness, don't write (CI gate)")
    sys.exit(SyncNumbers.sync(check=ap.parse_args().check))
