"""Unit tests for the namespace-class detector (bd cardiac-seg-h7vy.4): the shared-state signal, and the
two skip rules (already-stateful __init__, CLI command classes) that keep it from false-flagging."""
import ast
import textwrap

from devtools.state_candidates import analyze, scan, shared_state


def _cls(src: str) -> ast.ClassDef:
    return next(n for n in ast.walk(ast.parse(textwrap.dedent(src))) if isinstance(n, ast.ClassDef))


def test_shared_param_across_methods_is_latent_state():
    """A param carried by >=half (and >=2) of the staticmethods surfaces as shared state, counted."""
    shared = shared_state(_cls("""
        class Inference:
            @staticmethod
            def a(model, vol): ...
            @staticmethod
            def b(model, vol, size): ...
            @staticmethod
            def c(model, size): ...
    """))
    assert shared == {"model": 3, "vol": 2, "size": 2}  # 3 methods -> threshold 2; all three params qualify


def test_per_call_only_param_below_threshold_dropped():
    """A param in just one method (below half) is NOT latent state — it's a genuine per-call arg."""
    shared = shared_state(_cls("""
        class M:
            @staticmethod
            def a(spacing): ...
            @staticmethod
            def b(spacing): ...
            @staticmethod
            def c(spacing): ...
            @staticmethod
            def d(other): ...
    """))
    assert shared == {"spacing": 3}


def test_stateful_class_with_init_is_skipped():
    """A class that already has __init__ is done-right — never a candidate."""
    assert shared_state(_cls("""
        class Normalizer:
            def __init__(self, root): self.root = root
            @staticmethod
            def a(x, root): ...
            @staticmethod
            def b(x, root): ...
    """)) == {}


def test_cli_command_class_is_skipped():
    """add_args + run = a dispatcher command, legitimately stateless — skipped."""
    assert shared_state(_cls("""
        class Cmd:
            @staticmethod
            def add_args(ap): ...
            @staticmethod
            def run(args): ...
    """)) == {}


def test_pydantic_config_class_is_skipped():
    """A pydantic config (base BaseModel) whose staticmethods thread declared FIELDS is not latent
    state — the params live in __init__ via the fields already. False positive (N4Cfg), skipped."""
    assert shared_state(_cls("""
        class Foo(BaseModel):
            shrink: int = 4
            fwhm: float = 0.15
            @staticmethod
            def a(vol, shrink, fwhm): ...
            @staticmethod
            def b(vol, shrink, fwhm): ...
    """)) == {}


def test_autograd_function_is_skipped():
    """A torch.autograd.Function threads `ctx` by the framework API (forward/backward) — a contract, not
    promotable instance state -> skipped like the pydantic config."""
    assert shared_state(_cls("""
        class GradReverse(Function):
            @staticmethod
            def forward(ctx, x, lambd): ...
            @staticmethod
            def backward(ctx, grad): ...
    """)) == {}


def test_scan_skips_coverage_omit_shells(tmp_path, monkeypatch):
    """scan() reuses [tool.coverage] omit — a runner/adapter shell's shared params are its data, not object
    identity, so an omitted file is not flagged; a non-omitted logic file with the same shape is."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "pyproject.toml").write_text('[tool.coverage.run]\nomit = ["pkg/runner.py"]\n')
    pkg = tmp_path / "pkg"
    pkg.mkdir()
    shape = "class C:\n    @staticmethod\n    def a(model, x): ...\n    @staticmethod\n    def b(model, y): ...\n"
    (pkg / "runner.py").write_text(shape)     # declared an omit shell -> skipped
    (pkg / "logic.py").write_text(shape)      # logic module -> flagged
    files = [r[2].replace("\\", "/") for r in scan(["pkg"])]
    assert any("logic.py" in f for f in files)
    assert not any("runner.py" in f for f in files)


def test_single_method_class_has_no_shared_state():
    """One method can't share anything across the class."""
    assert shared_state(_cls("""
        class One:
            @staticmethod
            def only(a, b, c): ...
    """)) == {}


def test_analyze_scores_and_names_the_class(tmp_path):
    """analyze returns (score, class, n_methods, shared) for a candidate in a real file."""
    f = tmp_path / "m.py"
    f.write_text(textwrap.dedent("""
        class Runner:
            @staticmethod
            def a(model, x): ...
            @staticmethod
            def b(model, y): ...
    """))
    (score, name, n, shared), = analyze(f)
    assert name == "Runner"
    assert n == 2
    assert shared == {"model": 2}
    assert score == 2
