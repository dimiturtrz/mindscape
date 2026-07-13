"""The shared coverage-omit primitive (devtools/omit.py) — the 'not logic' shell set the arch-fitness
gates reuse. Pure glob-matching + a tomllib read, no grimp/networkx."""
from devtools.omit import coverage_omit, matches_omit


def test_matches_omit_globs_the_coverage_patterns():
    pats = ["core/data/*/registry.py", "neuroscan/tasks/cli.py", "neuroscan/tasks/**"]
    assert matches_omit("core/data/eeg/registry.py", pats)          # single-* matches one segment
    assert matches_omit("neuroscan/tasks/cli.py", pats)             # exact
    assert matches_omit("neuroscan/tasks/workload/run.py", pats)    # ** matches across segments
    assert not matches_omit("core/features/eeg/bandpower.py", pats)  # logic module — not omitted
    assert not matches_omit("core/data/registry.py", pats)          # single-* needs the middle segment


def test_matches_omit_normalizes_backslashes():
    assert matches_omit("neuroscan\\tasks\\cli.py", ["neuroscan/tasks/cli.py"])   # windows path normalized


def test_coverage_omit_reads_the_section(tmp_path):
    (tmp_path / "pyproject.toml").write_text('[tool.coverage.run]\nomit = ["a/b.py", "c/**"]\n')
    assert coverage_omit(str(tmp_path / "pyproject.toml")) == ["a/b.py", "c/**"]


def test_coverage_omit_absent_is_empty(tmp_path):
    assert coverage_omit(str(tmp_path / "nope.toml")) == []                        # missing file -> []
    (tmp_path / "pyproject.toml").write_text("[tool.ruff]\nline-length = 120\n")
    assert coverage_omit(str(tmp_path / "pyproject.toml")) == []                    # no coverage section -> []
