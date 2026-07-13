"""nox task runner (bd kvo) — one command reproduces the CI gate suite locally.

`nox` runs everything; `nox -s lint test fitness dup` picks sessions. Each session shells the SAME pinned
tools CI uses (single source for the gate commands, so local == CI). venv backend is `none`: tools run via
uv/uvx/npx which handle their own isolation, so no per-session venv is built.
"""
import nox

nox.options.default_venv_backend = "none"
nox.options.sessions = ["lint", "test", "fitness", "dup"]   # `dup` (jscpd) needs Node; drop it if absent

_RUFF = "ruff@0.15.13"
_VULTURE = "vulture@2.16"
_SELECT = "W,F,I,B,T20,E7,PLR2004,PLC0415,FBT,C901,PLR0912,PLR0915,BLE001,S110,PLR0913,RUF100"
_PKGS = ("core", "neuroscan")


@nox.session
def lint(session):
    """Static gates with no project env: ruff (enforced) + vulture + import-linter + ast-grep."""
    session.run("uvx", _RUFF, "check", "core", "neuroscan", "baselines", "--select", _SELECT, external=True)
    session.run("uvx", _VULTURE, "--min-confidence", "80", external=True)
    session.run("uvx", "--from", "import-linter", "lint-imports", external=True)
    session.run("uvx", "--from", "ast-grep-cli", "ast-grep", "scan", "-c", "devtools/sgconfig.yml", *_PKGS,
                external=True)


@nox.session
def test(session):
    """Test + coverage floor (needs the synced env)."""
    session.run("uv", "run", "--extra", "dev", "--extra", "devtools", "pytest", "tests/unit", "--cov", "-q",
                external=True)
    session.run("uv", "run", "coverage", "report", "--fail-under=80", external=True)


@nox.session
def fitness(session):
    """Architecture fitness gate (grimp+networkx; needs the devtools extra)."""
    session.run("uv", "run", "--extra", "devtools", "python", "-m", "devtools.graph", "--assert", external=True)


@nox.session
def dup(session):
    """Duplication — enforced, blocks >1% (jscpd; needs Node)."""
    session.run("npx", "--yes", "jscpd@4", *_PKGS, "--config", "devtools/jscpd.json", external=True)
