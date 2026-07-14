# interpretations/

Where we make sense of **our own artifacts** — the runs, numbers, and checkpoints this project produced —
grounded in what [`../research/`](../research/) taught us about the field. Three knowledge layers, one home
each:

- [`../research/`](../research/) — **theirs**: external / literature synthesis (cited raw material in).
- [`../learning/`](../learning/) — **the ramp**: the study curriculum + foundations to stay conversant.
- **`interpretations/`** — **ours**: what our results *mean* — the reasoning that ties artifacts + research
  into a claim, kept honest against the code and the numbers.

An interpretation is not the artifact (that's `runs/`, mlflow, a `bd` memory, git history) and not the field's
knowledge (that's `research/`) — it's the *sense-making over both*.

## Split by task, converge across them

```
interpretations/
  <task>/                 one folder per task — sense-making coupled to that task's artifacts
    <date>_<topic>.md     e.g. visual/2026-07-14_perception_continuous_metrics.md
  converging/             cross-task threads — a claim that spans tasks (added when one is written)
```

- **`<task>/`** (`visual/`, `motor_imagery/`, `workload/`, …) — an interpretation of *one task's* runs lives
  next to its siblings. Coupled to that task's numbers.
- **`converging/`** — where a thread runs *across* tasks (e.g. the cross-subject single-trial gap recurring in
  motor imagery, workload, and perception). Created when the first cross-task doc is actually written — no
  speculative empty folders.

Naming: `<date>_<topic>.md`, underscores (matches `research/deep_dives/`). One home per interpretation; the
portfolio READMEs carry the *result* and **link here** for the reasoning (no duplicated prose).
