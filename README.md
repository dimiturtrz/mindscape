# mindscape

**Personal learning project — non-invasive neural decoding: reading intent and meaning from brain
signals, evaluated honestly, with an efficient on-device bias. Edge-deployable.**

> Working name `mindscape` (the terrain of the mind). Sibling in name and philosophy to
> [synthscape/mirage](https://github.com/dimiturtrz/synthscape) — the `-scape` is deliberate.

mindscape decodes **non-invasive neural signals** (EEG / MEG, public data) — training decoders and,
as in the sibling projects, treating the **honest evaluation** as the contribution rather than a
leaderboard number. The second through-line is **deployability**: an *efficient* decoder — the
edge-inference discipline (quantization / distillation) the field tends to underweight.

It's also how I'm ramping into neural decoding. The signal-processing, time-series modeling,
calibration / evaluation discipline, and edge-inference work carry over from prior
acoustic-detection work; the **neuroscience and the decoding methods I learn as I go.** Sibling
projects, same engine/eval-first, honest-where-it-fails philosophy, different domain:
[systole](https://github.com/dimiturtrz/cardiac-seg) (cardiac MRI → ejection fraction) and
[mirage](https://github.com/dimiturtrz/synthscape) (3D surface-defect anomaly).

## Status — just started (2026-06-30)
This is the **initialization**: the thesis and the shape are set; the first decoding task, the
evaluation harness, and the datasets land next. **No results yet** — when there are, they'll be
reported the way the siblings report theirs (the *measured contrast* + where it fails), not as a
leaderboard chase. This is a genuine direction I want to build in, started now and expanded in the
open from here.

## The shape (spine first, then the angle)
The full staged plan will live in `docs/PLAN.md` once the spine lands. The intended arc:

1. **Warm-up — reproduce a known result on real data.** Stand up a standard non-invasive decoding
   task (e.g. motor-imagery on a public benchmark) through a *verified* eval harness — prove the
   spine before anything novel. Reproducing and extending methods from papers is the mode the work
   runs in.
2. **Toward communication / semantic decoding.** Move from the warm-up toward the harder, more
   meaningful target — decoding intended communication from non-invasive signal — where the
   real-world, uncontrolled-signal evaluation traps live (a decoder that scores well on its own
   recordings can mean little out of distribution; honest eval + calibration under shift is the
   recurring contribution).
3. **The differentiator — efficient, on-device decoding.** Quantization / distillation toward a
   real-time decoder that runs on commodity edge hardware — the deployable angle, carried from prior
   edge-inference work, that the field underweights.

## Data
Public non-invasive neural-decoding benchmarks (EEG / MEG), kept **outside the repo** (size +
licensing) and pointed at via a one-root path config — the same convention as the sibling projects.
The first dataset lands with the warm-up; the choice + adapter will be documented when it does.

## How it's built
Agent-driven build, human-owned judgment — coding agents scaffold the plumbing; the modeling
decisions, the measurement correctness, and the evaluation are mine. Signal-processing and
evaluation discipline carry over from prior ML work; the neuroscience and decoding specifics I learn
as I go (a `learning/` track will track the ramp, as in the siblings).

## License
TBD (code will be permissively licensed). Any datasets used are **not** included and carry their own
licenses — obtained from their providers.
