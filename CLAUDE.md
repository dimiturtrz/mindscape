# Project Instructions for AI Agents

This file provides instructions and context for AI coding agents working on this project.

<!-- BEGIN BEADS INTEGRATION v:1 profile:minimal hash:3216161c -->
## Beads Issue Tracker

This project uses **bd (beads)** for issue tracking. Run `bd prime` to see full workflow context and commands.

### Quick Reference

```bash
bd ready              # Find available work
bd show <id>          # View issue details
bd update <id> --claim  # Claim work
bd close <id>         # Complete work
```

### Rules

- Use `bd` for ALL task tracking — do NOT use TodoWrite, TaskCreate, or markdown TODO lists
- Run `bd prime` for detailed command reference and session close protocol
- Use `bd remember` for persistent knowledge — do NOT use MEMORY.md files

## Session Completion

**When ending a work session**, you MUST complete ALL steps below. Work is NOT complete until `git push` succeeds.

**MANDATORY WORKFLOW:**

1. **File issues for remaining work** - Create issues for anything that needs follow-up
2. **Run quality gates** (if code changed) - Tests, linters, builds
3. **Update issue status** - Close finished work, update in-progress items
4. **PUSH TO REMOTE** - This is MANDATORY:
   ```bash
   git pull --rebase
   bd dolt push
   git push
   git status  # MUST show "up to date with origin"
   ```
5. **Clean up** - Clear stashes, prune remote branches
6. **Verify** - All changes committed AND pushed
7. **Hand off** - Provide context for next session

**CRITICAL RULES:**
- Work is NOT complete until `git push` succeeds
- NEVER stop before pushing - that leaves work stranded locally
- NEVER say "ready to push when you are" - YOU must push
- If push fails, resolve and retry until it succeeds
<!-- END BEADS INTEGRATION -->


## Build & Test

_Add your build and test commands here_

```bash
# Example:
# npm install
# npm test
```

## Architecture Overview

_Add a brief overview of your project architecture_

## Scaffolding

Guardrails provisioned by **sdlc-scaffold** via copier — `.copier-answers.yml` pins the version. The gate
config and the nox/CI/pre-commit runners are **template-owned**, and the guardrail analyzers now install as
the **`sdlc-devtools`** package (a git-dep pinned by scaffold tag via `devtools_ref` in `.copier-answers.yml`,
no vendored `devtools/` source — sgconfig/jscpd configs ship *inside* the package, located by
`python -m devtools.config <name>`): don't hand-edit them to pass a gate — fix upstream in the scaffold and
`uvx copier update` (or bump `devtools_ref` for an engine-only update), or edit only within `# >>> LOCAL-SLOT`
regions. `copier update` pulls scaffold improvements as reviewable steps. (mindscape carries a few
documented not-slotted local mods — the ML dep tree, the uv cu130 index, extra vulture `ignore_names` — and
keeps its own README/CLAUDE/AGENTS content.)

## Conventions & Patterns

### Code quality — ruff ratchet (bd mindscape-y63)

CI has an **enforced** lint gate (`ruff check core neuroscan --select <union>`) over the house **union
select** — the maximal superset the scaffold ships (bd o70; landed wholesale via sdlc-scaffold v0.14–v0.16,
no longer grown per-family here): `N` case, `S101`+`S`-family, `RUF0xx`, `PLR`, `SLF001`, `PTH`, `PERF`, … .
Plus **advisory** steps (full-tree `--statistics` + `ruff format --check`). Config in `[tool.ruff]`
(line-length 120). `N` **is** selected now, with this repo's idiom vocab in the `pep8-naming` LOCAL-SLOT
(`X*`/`F*` design + feature matrices, single-cap linalg `A/M/P/R/T/W`…); `RUF001/002/003` (intentional
`→ ≈ ×` unicode) stay ignored, plus `F722` (jaxtyping shape strings). `SLF001` is carved in the
`per-file-ignores` LOCAL-SLOT where the mandated op-namespace pattern forces `Cls._helper` access — a
scaffold-owned rule conflict (sdlc-scaffold-8ex), not a dodge. Working rules:

- **Bare `# noqa: CODE`** only — no prose (RUF100 enforces it suppresses a real hit). Prefer fixing.
- **Imports at top** — break circulars by extraction, never lazy imports.
- **Config objects, not param lists** — collapse too-many-args to a pydantic/dataclass config.
- **No blind `except`** — a specific exception, or let it crash.
- **Minimal comments** — self-documenting names.
- Keep **CLAUDE.md ↔ AGENTS.md in sync**.

### Architecture layers — import-linter contract (bd o92)

Three tiers, imports point **down only**: `core` (clean kernel) < `neuroscan` (trainer) < `neuroviz`
(viewer). Enforced statically in CI (`uvx --from import-linter lint-imports`, contracts in
`[tool.importlinter]`). What's forbidden is **upward drift**: `core` importing `neuroscan`/`neuroviz`, or
`neuroscan` importing `neuroviz`. The viewer reusing the trainer's model registry
(`neuroviz → neuroscan.models.get_method`) is an **intended downward edge** — allowed, not drift. Adding an
upward import breaks CI; if a layer genuinely needs a symbol from above, the symbol is in the wrong layer —
push it down (into `core`), don't invert the arrow.

### Architecture fitness — `graph.py --assert` gate (bd 3nn)

`devtools.graph` (bd 2r9/3nn, from the installed `sdlc-devtools` package, `--extra devtools`) is both the **explorer** (`python -m devtools.graph` —
fan-in/out / bottleneck / betweenness / cycles via grimp+networkx) **and** a CI **fitness gate**
(`python -m devtools.graph --assert`) — the *metric* arch axis import-linter's categorical contracts can't
express. **Blocks** on a god-module (fan-in AND fan-out both > `bottleneck_degree`), an import cycle (SCC>1),
a god-file (> `file_max`), or a **test-mirror** gap: a logic module `<pkg>/<path>/foo.py` with no strict
path-mirror test at `tests/unit/<path>/test_foo.py` (one home per module — a same-purpose test under a
different name does not count; rename it to the mirror path). Coverage-**omitted** shells (runners/adapters/
GPU/download/viz glue) are exempt — a non-unit-testable shell has no meaningful mirror and forcing a stub
violates no-stubs, so the same "not logic" set the coverage gate omits is exempt here too (`unmirrored()`
reads `[tool.coverage] omit`). **Advisory** (logged, never blocks): line-floor + betweenness chokepoint + Martin main-sequence distance
(`main_sequence_max`, off at 0.0). Thresholds
live in `[tool.structure]`, chosen clean against today's graph — they **ratchet only tighter, never relax**.
Runs in the CI `tests` job (needs the `devtools` extra).

### Class-shape smell explorers — advisory (bd 0hl)

`devtools.{state_candidates,data_clumps,lcom}` — pure-AST detectors run **advisory** in CI (never block):
namespace-class with latent shared state (→ `__init__`), param data-clumps (→ Introduce Parameter Object),
LCOM4 disjoint-state classes (→ split). Known-legit patterns are **excluded** so the radar stays actionable:
the sklearn `fit`/`transform` contract (lcom), and coverage-`omit` shells + `torch.autograd.Function`
(state_candidates — it shares `devtools.omit` with the test-mirror rule, so the gates agree on what a shell
is). The residual is the stateless op-namespace **floor** — methods sharing a per-call data/knob param
(`X`/`y`/`grid`/…); genuine latent state (a class threading the SAME identity/config through every method, like
a montage operator) stands out against it and is the promote signal. A **regression radar**, never a gate.
`python -m devtools.<name> core neuroscan`.

### Module shape — ast-grep gate (bd ylq)

Semantic AST rules ruff's token linters can't express (packaged `sgconfig.yml` → `sg-rules/`, shipped inside
`sdlc-devtools`; enforced in CI: `ast-grep scan -c "$(uv run -q --extra devtools python -m devtools.config sgconfig)" core neuroscan`, severity `error` blocks). Two rules:
**`py-top-level-function`** — everything meaningful is a method: a top-level `def` must live on the class that
owns it (`main`/`_main` exempt). The whole `core`+`neuroscan` tree was migrated (269 free funcs → 0), so any
NEW top-level function fails CI. **`py-top-level-side-effect`** — no import-time call statements (move them into
a method / lazy-populate, as the registries do; `matplotlib.use()` exempt). Fix by refactoring, never a `# noqa`.
Constants, `logger`, dataclasses, pydantic models, enums, and `nn.Module`/`Dataset` subclasses stay top-level —
only plain `def`s move.

### Duplication — jscpd gate (bd apl)

Packaged `jscpd.json` (Node, config located via `python -m devtools.config jscpd`, run in CI + `nox`) — the DRY axis. **Blocks** when python duplication
exceeds 1% (currently 0.7% after the runner boilerplate was DRY-extracted into `Cli.setup_logging` +
`Riemann.cross_subject_decode`). Fix a regression by extracting the shared logic, not by raising the threshold —
it ratchets **down** as dup settles. Genuinely-distinct-but-similar-shaped runners are left un-merged (don't
over-couple to chase the number).

### Magic literals — enforced ratchet (sdlc-scaffold)

`devtools.magic_literals` — recurring string vocab + repeated dict schemas (StrEnum / record candidates),
the non-comparison axis ruff `PLR2004` can't see. **Blocks** over the `[tool.magic_literals]` ceiling — a
per-repo FACT frozen at mindscape's DataFrame-schema + metric-key floor (`max_strings=33`, `max_key_sets=10`:
column names like `{run,session,subject}`, metric keys `{acc,ece,kappa}`). A NEW literal migrates to a
StrEnum/constant or bumps the ceiling with a documented reason; ratchets **down** as the schema vocab settles.

### Shape contracts — advisory (sdlc-scaffold)

`devtools.shape_contracts` — a public array/tensor boundary should carry a **jaxtyping** shape
(`Float[Tensor, "b c h w"]`, live at runtime via a `@shapecheck` beartype wrapper), not a bare
`np.ndarray`/`Tensor`; repo array aliases in `[tool.shape_contracts]`. **Advisory** (report-only) until the
tree is boundary-clean, then the scaffold graduates it to `--assert` (sdlc-scaffold-vip.4).

### Local gate runners — nox + pre-commit (bd kvo/dno)

`nox` (`noxfile.py`, `devtools` extra) reproduces the CI gate suite locally from one command: `nox` runs
`lint` (ruff+vulture+import-linter+ast-grep) + `test` (pytest+coverage floor) + `fitness` (graph.py --assert);
`nox -s dup` adds advisory jscpd. Sessions shell the SAME pinned tools CI uses, so local == CI.
`.pre-commit-config.yaml` runs the fast static gates (all but test/coverage) before each commit — enable with
`pre-commit install` (keep it separate from the beads `.beads/hooks/pre-commit`), or one-shot `pre-commit run
--all-files`.
