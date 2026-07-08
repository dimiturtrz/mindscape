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

## Conventions & Patterns

### Code quality — ruff ratchet (bd mindscape-y63)

CI has an **enforced** lint gate (`ruff check core neuroscan baselines --select <graduated>`) that grows one
family per PR — fix-all-first, then graduate the family into the enforced `--select`. Plus **advisory** steps
(full curated backlog + `ruff format --check`). Config lives in `[tool.ruff]` (line-length 120). `N` (the
`X`/`y` tensor idiom) and `RUF001/002/003` (intentional `→ ≈ ×` unicode in docstrings) are deliberately not
selected. Working rules:

- **Bare `# noqa: CODE`** only — no prose (RUF100 enforces it suppresses a real hit). Prefer fixing.
- **Imports at top** — break circulars by extraction, never lazy imports.
- **Config objects, not param lists** — collapse too-many-args to a pydantic/dataclass config.
- **No blind `except`** — a specific exception, or let it crash.
- **Minimal comments** — self-documenting names.
- Keep **CLAUDE.md ↔ AGENTS.md in sync**.
