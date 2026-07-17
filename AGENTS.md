# Agent Instructions

This project uses **bd** (beads) for issue tracking. Run `bd prime` for full workflow context.

## Quick Reference

```bash
bd ready              # Find available work
bd show <id>          # View issue details
bd update <id> --claim  # Claim work atomically
bd close <id>         # Complete work
bd dolt push          # Push beads data to remote
```

## Non-Interactive Shell Commands

**ALWAYS use non-interactive flags** with file operations to avoid hanging on confirmation prompts.

Shell commands like `cp`, `mv`, and `rm` may be aliased to include `-i` (interactive) mode on some systems, causing the agent to hang indefinitely waiting for y/n input.

**Use these forms instead:**
```bash
# Force overwrite without prompting
cp -f source dest           # NOT: cp source dest
mv -f source dest           # NOT: mv source dest
rm -f file                  # NOT: rm file

# For recursive operations
rm -rf directory            # NOT: rm -r directory
cp -rf source dest          # NOT: cp -r source dest
```

**Other commands that may prompt:**
- `scp` - use `-o BatchMode=yes` for non-interactive
- `ssh` - use `-o BatchMode=yes` to fail instead of prompting
- `apt-get` - use `-y` flag
- `brew` - use `HOMEBREW_NO_AUTO_UPDATE=1` env var

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
   bd dolt push        # full push; also heals a stale per-write auto-push (see below)
   git push
   git status  # MUST show "up to date with origin"
   ```
   > **Dolt auto-push `git ref not found: refs/dolt/remotes/.../data/<uuid>` (bd nhi):** bd's per-write
   > auto-push is INCREMENTAL, keyed on `.beads/push-state.json` (gitignored). If that data ref is GC'd from
   > the local dolt store the delta can't resolve and every `bd close/create/update` prints the error. It is
   > cosmetic — the git-tracked `.beads/issues.jsonl` is the source of truth and is never at risk. Heal:
   > `bd dolt push` (full push, above) OR `rm .beads/push-state.json` — the next write re-establishes a fresh
   > full push. Upstream bd fragility, not a mindscape-source bug.
5. **Clean up** - Clear stashes, prune remote branches
6. **Verify** - All changes committed AND pushed
7. **Hand off** - Provide context for next session

**CRITICAL RULES:**
- Work is NOT complete until `git push` succeeds
- NEVER stop before pushing - that leaves work stranded locally
- NEVER say "ready to push when you are" - YOU must push
- If push fails, resolve and retry until it succeeds
<!-- END BEADS INTEGRATION -->

## Scaffolding

Guardrails provisioned by **sdlc-scaffold** via copier — `.copier-answers.yml` pins the version. The gate
config and the nox/CI/pre-commit runners are **template-owned**, and the guardrail analyzers now install as
the **`sdlc-devtools`** package (a git-dep pinned by scaffold tag via `devtools_ref` in `.copier-answers.yml`,
no vendored `devtools/` source — sgconfig/jscpd configs ship *inside* the package, located by
`python -m devtools.config <name>`): don't hand-edit them to pass a gate — fix upstream in the scaffold and
`uvx copier update` (or bump `devtools_ref` for an engine-only update), or edit only within `# >>> LOCAL-SLOT`
regions. `copier update` pulls scaffold improvements as reviewable steps. (mindscape carries a few
documented not-slotted local mods — the ML dep tree, the uv cu130 index, extra vulture `ignore_names`, and the
`lint_paths`/`jscpd_paths` widening of the ruff + jscpd gates onto `neuroviz` (the viewer tree; hand-applied to
`LINT_LAYERS`/`JSCPD_LAYERS` + the CI/pre-commit runner lines because same-version copier can't re-render the
answer — a cross-version `copier update` regenerates the identical lines, so it's zero-drift; upstream
proper-fix tracked in bd oeen) — and keeps its own README/CLAUDE/AGENTS content.)

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

### Magic literals & complexity — advisory explorers (sdlc-scaffold)

`devtools.magic_literals` (recurring string vocab + repeated dict schemas → StrEnum / record candidates, the
non-comparison axis ruff `PLR2004` can't see) and `devtools.complexity` (radon cyclomatic ranking) run
**advisory** — ranked reports, no gate, no config. The **fixed** complexity gate stays ruff `C901` (CC>10).
There is no honest universal magic-literal ceiling (0 too strict, N arbitrary), so it stays a report — the old
enforced `[tool.magic_literals]` ratchet was **retired** in the v1.5.0 scaffold update; if a repo later needs a
budget, a config knob is added on real need, not pre-emptively.

### Dependency hygiene — deptry gate (sdlc-scaffold)

`deptry` (`uv run --with deptry==0.25.1 deptry .`, CI + `nox`) — imported-but-undeclared (DEP001),
declared-but-unused (DEP002), transitive-imported (DEP003). **Blocks the merge.** Config in `[tool.deptry]`:
`noxfile.py` excluded (imports the `nox` runner, deliberately not a project dep); per-rule ignores in the
`deptry-unused` LOCAL-SLOT — DEP002 the test/tooling deps + the optional-extra deps whose only consumers are
coverage-omitted shells (onnx/onnxscript/osfclient/openneuro-py/nibabel), DEP001 the `external/` git-repo
checkouts (EEGPT_mcae/models/benchnirs, never pip deps). Fix a real hit by **declaring** the dep (as joblib +
matplotlib were promoted transitive→direct), not by widening the ignore.

### Architecture diagrams — archmap (advisory doc-gen, sdlc-scaffold)

`devtools.archmap core neuroscan` regenerates `docs/architecture/` — a committed, diffable **`graph.json`**
(nodes + weighted module→module import edges; arrow weight = count of imports crossing the pair — the
architecture-erosion record) plus a self-contained interactive **cytoscape `index.html`** viewer (gitignored,
~700KB vendored libs, regenerated locally / by the Pages workflow) that **folds/expands packages to any depth**
and focuses a module's neighbourhood. **Commit the `graph.json` diff**; the pre-commit hook regenerates it, CI
runs `--check` **advisory** (warns on stale, never blocks). Regenerate with `nox -s archmap`. (v1.6.0's
any-depth viewer supersedes the old one-tier mermaid `.md` tree and the earlier 2-level-nesting request.)

### Known-CVE scan — pip-audit (nightly, sdlc-scaffold)

`.github/workflows/audit.yml` — `pip-audit --skip-editable` over the resolved dep closure against the PyPA
advisory DB. **Nightly + manual dispatch, NOT a per-PR gate** (advisories change under you — a new CVE can land
with no code change on your side), so it goes red on a schedule and notifies rather than blocking a merge.
`--skip-editable` drops the git/local installs (sdlc-devtools) that carry no PyPI release to look up.

### Shape contracts — advisory (sdlc-scaffold)

`devtools.shape_contracts` — a public array/tensor boundary should carry a **jaxtyping** shape
(`Float[Tensor, "b c h w"]`, live at runtime via a `@shapecheck` beartype wrapper), not a bare
`np.ndarray`/`Tensor`; repo array aliases in `[tool.shape_contracts]`. **Advisory** (report-only) until the
tree is boundary-clean, then the scaffold graduates it to `--assert` (sdlc-scaffold-vip.4).

### Local gate runners — nox + pre-commit (bd kvo/dno)

`nox` (`noxfile.py`, `devtools` extra) reproduces the CI gate suite locally from one command: `nox` runs
`lint` (ruff check + ruff format --check + vulture + import-linter + arch-fitness `graph.py --assert` +
ast-grep + jscpd — all folded into one session as of scaffold v1.5.0) + `test` (pytest) + `cov`
(pytest+coverage; **enforced floor 80**, 95 advisory-warn). `nox -s archmap` regenerates the arch diagrams,
`nox -s audit` runs the pip-audit CVE scan. Sessions shell the SAME pinned tools CI uses, so local == CI.
`.pre-commit-config.yaml` runs the fast static gates (all but test/coverage) before each commit — enable with
`pre-commit install` (keep it separate from the beads `.beads/hooks/pre-commit`), or one-shot `pre-commit run
--all-files`.

