"""Import-graph diagnostic ranking (devtools/graph.py, bd 2r9). Tests the pure ranking + cycle
detection on a hand-built graph — no grimp / no repo scan. networkx-guarded so CI (dev extra, no
networkx) skips; runs under the devtools extra."""
import pytest

pytest.importorskip("grimp")            # devtools extra; networkx is present transitively, grimp is not
nx = pytest.importorskip("networkx")

from devtools.graph import (  # noqa: E402
    _DEFAULTS,
    _god_modules,
    _matches_omit,
    _oversized,
    _top,
    assert_fitness,
    load_structure_cfg,
    report,
    unmirrored,
)


def test_top_ranks_descending_and_caps():
    assert _top([("a", 1), ("b", 5), ("c", 3)], 2) == [("b", 5), ("c", 3)]


def test_report_surfaces_fan_in_and_cycles():
    g = nx.DiGraph()
    g.add_edges_from([("x", "hub"), ("y", "hub"), ("z", "hub"),   # hub: fan-in 3 (load-bearing)
                      ("a", "b"), ("b", "a")])                     # a<->b: an import cycle (SCC>1)
    r = report(g, top=5)
    assert "fan-in (load-bearing)" in r and "hub" in r
    assert "import cycles (SCC>1): 1" in r


def _god_graph(degree: int) -> nx.DiGraph:
    """A single module with fan-in AND fan-out both > `degree` (a god-module) plus enough leaves."""
    g = nx.DiGraph()
    for i in range(degree + 1):
        g.add_edge(f"in{i}", "god")       # fan-in  = degree+1 > degree
        g.add_edge("god", f"out{i}")      # fan-out = degree+1 > degree
    return g


def test_god_module_flagged_over_degree_clean_under():
    deg = 3
    assert len(_god_modules(_god_graph(deg), deg)) == 1          # both fan-in & fan-out exceed -> flagged
    assert _god_modules(_god_graph(deg), degree=deg + 5) == []   # under threshold -> clean


def test_oversized_blocks_over_ceiling_only():
    files = [("big.py", 801), ("ok.py", 200)]
    over = _oversized(files, mx=750)
    assert len(over) == 1 and "big.py" in over[0]        # only the file over the ceiling
    assert _oversized(files, mx=1000) == []              # both under -> clean


def test_assert_fitness_blocks_god_module_and_god_file():
    g = _god_graph(_DEFAULTS["bottleneck_degree"])
    files = [("huge.py", _DEFAULTS["file_max"] + 1)]
    blocking, _ = assert_fitness(g, files, _DEFAULTS)
    assert any("god-module" in b for b in blocking)
    assert any("god-file" in b for b in blocking)


def test_assert_fitness_clean_graph_no_blocking():
    g = nx.DiGraph([("a", "b"), ("b", "c")])                     # tiny chain, no cycles / god-modules
    blocking, _ = assert_fitness(g, [("small.py", 40)], _DEFAULTS)
    assert blocking == []


def test_line_floor_off_by_default_is_advisory_only():
    g = nx.DiGraph([("a", "b")])
    _, advisory = assert_fitness(g, [("tiny.py", 3)], _DEFAULTS)   # file_min=0 -> no undersized entries
    assert not any("earn its keep" in a for a in advisory)


def test_load_structure_cfg_defaults_when_absent():
    cfg = load_structure_cfg("does-not-exist.toml")
    assert cfg == _DEFAULTS


def _mirror_tree(root, *, mirror_bar=False, omit_bar=False):
    """A tiny source pkg + tests/unit under `root`: foo.py always mirrored, bar.py mirrored only if asked."""
    (root / "pkg").mkdir(parents=True)
    (root / "pkg" / "__init__.py").write_text("")           # plumbing — always exempt
    (root / "pkg" / "foo.py").write_text("X = 1\n")
    (root / "pkg" / "bar.py").write_text("Y = 2\n")
    tdir = root / "tests" / "unit" / "pkg"
    tdir.mkdir(parents=True)
    (tdir / "test_foo.py").write_text("def test_x(): assert True\n")
    if mirror_bar:
        (tdir / "test_bar.py").write_text("def test_y(): assert True\n")
    if omit_bar:
        (root / "pyproject.toml").write_text('[tool.coverage.run]\nomit = ["pkg/bar.py"]\n')


def test_unmirrored_flags_only_the_module_without_a_strict_mirror(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _mirror_tree(tmp_path)                                   # foo mirrored, bar not, __init__ exempt
    gaps = unmirrored(["pkg"])
    assert len(gaps) == 1 and "pkg/bar.py" in gaps[0]        # only bar (no test_bar.py) blocks
    assert "test_bar.py" in gaps[0]                          # message names the strict mirror path expected


def test_unmirrored_clean_when_every_module_mirrored(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _mirror_tree(tmp_path, mirror_bar=True)                  # both foo and bar have their mirror
    assert unmirrored(["pkg"]) == []


def test_unmirrored_exempts_coverage_omit_shells(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _mirror_tree(tmp_path, omit_bar=True)                    # bar unmirrored BUT declared an omit shell
    assert unmirrored(["pkg"]) == []                         # the omit carve exempts it — no false gap


def test_matches_omit_globs_the_coverage_patterns():
    pats = ["core/data/*/registry.py", "neuroscan/tasks/cli.py", "neuroscan/tasks/**"]
    assert _matches_omit("core/data/eeg/registry.py", pats)          # single-* matches one segment
    assert _matches_omit("neuroscan/tasks/cli.py", pats)             # exact
    assert _matches_omit("neuroscan/tasks/workload/run.py", pats)    # ** matches across segments
    assert not _matches_omit("core/features/eeg/bandpower.py", pats)  # logic module — not omitted
    assert not _matches_omit("core/data/registry.py", pats)          # single-* needs the middle segment
