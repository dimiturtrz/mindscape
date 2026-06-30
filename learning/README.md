# learning/

My study material + quiz logs for mindscape — the ramp into neural decoding. One file per topic: the
**theory writeup** (grounded in the pipeline we actually built), then a dated **quiz log** (questions,
my answers, honest assessment) appended when I ask to be quizzed.

Crash-course scope: enough to be **conversant and honest** about every claim the project makes — the task,
the signal, the methods, the eval — not a full degree. The [field-map](field-map.md) is the yardstick for
"how much is enough."

## The circuit (how each topic is learned)
1. **Research** — ground the topic (knowledge + web) → `../research/` (cited raw material).
2. **Theory** — the study writeup here → `NN_<topic>.md`, lessons tied to our code + numbers.
3. **Quiz (on demand)** — open-form questions on that theory when I say *"quiz me."*
4. **Log** — my answers + honest assessment + score appended to the topic file.
5. **Sharpen** — set the next concrete step.

**Honest by rule:** domain facts are source-grounded, not asserted from memory; wrong answers stay in the
log — the ramp is real, the self-eval is honest.

## Topics
1. [`01_motor-imagery-and-erd.md`](01_motor-imagery-and-erd.md) — the task, the motor system, imagery vs
   execution, ERD (the signal we decode).
2. [`02_signal-processing-and-csp.md`](02_signal-processing-and-csp.md) — how EEG arises, frequency bands,
   spectral power, spatial filters / CSP.
3. [`03_decoding-and-honest-eval.md`](03_decoding-and-honest-eval.md) — CSP+LDA → EEGNet/ATCNet, the
   cross-subject gap, calibration, the efficiency angle.

[glossary.md](glossary.md) — terms, one line each.
