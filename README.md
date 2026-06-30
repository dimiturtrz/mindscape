# mindscape

**Personal learning project — non-invasive neural decoding: reading intent and meaning from brain
signals, evaluated honestly, with an efficient on-device bias. Edge-deployable.**

> Working name `mindscape` (the terrain of the mind).

mindscape decodes **non-invasive neural signals** (EEG / MEG, public data) — training decoders and
treating the **honest evaluation** as the contribution rather than a leaderboard number. The second
through-line is **deployability**: an *efficient* decoder — the edge-inference discipline (quantization /
distillation) the field tends to underweight.

It's also how I'm ramping into neural decoding. The signal-processing, time-series modeling,
calibration / evaluation discipline, and edge-inference work carry over from prior ML projects of mine
(same eval-first, honest-where-it-fails philosophy, different domain); the **neuroscience and the
decoding methods I learn as I go.**

## See the signal the decoder reads
![neuroviz — mu/beta ERD animated over a motor-imagery trial; the contralateral motor cortex desynchronizes, the C3↔C4 hot spot flipping by imagined hand](neuroviz/docs/media/demo.gif)

Imagining one hand **desynchronizes the opposite motor cortex** (mu/beta power drop over C3↔C4) — the
motor-imagery signature, animated over the trial. [`neuroviz/`](neuroviz/): field-standard 2D topomaps +
CSP patterns + waveforms, dependency-free.

## The honest question
A motor-imagery decoder that scores well on a subject's *own* recordings — how far does it fall on a
subject it never saw? That gap, not the headline accuracy, is what this measures.

## The number that matters first
Stage 0, BCI Competition IV-2a (4-class motor imagery, 9 subjects), CSP+LDA, through a verified
harness with the standard train-session → eval-session protocol:

| regime | accuracy | kappa | ECE |
|---|---|---|---|
| within-subject | **0.598** | 0.463 | 0.139 |
| **cross-subject (leave-one-subject-out)** | **0.382** | 0.176 | 0.135 |
| **generalization gap** | **−0.216** | | |

A decoder at ~60% on its own data drops to **38%** on an unseen subject (chance = 25%). Several
subjects land **at or below chance** cross-subject. *That* is the contribution — measured, not
asserted.

## Where it fails (stratified, the siblings' standard)
- **Per-subject spread is huge** — within-subject accuracy ranges ~0.34–0.84 across the 9 subjects;
  the mean hides a 2× range.
- **Cross-subject collapse is subject-specific** — the "hard" subjects (low-SNR) drop near chance
  while clean subjects hold ~0.5; the harness reports per-subject accuracy + ECE every run.

## Honest limits (measured, not assumed)
- **Below published SOTA on raw decode**, deliberately: this reproduces *standard* methods (single-band
  CSP, commodity nets) through an honest protocol — not a fully-tuned SOTA recipe. The strong-decoder
  reproduction (ATCNet via the braindecode continuous-EMS pipeline) matches published numbers on clean
  subjects (~0.84) but underperforms on the hard ones; the gap analysis is written up in
  [`research/`](research/deep_dives/2026-06-30_2a_sota_recipe.md). The decoder is commodity — said so.
- **Neuroscience is a ramp** — the signal-processing/eval discipline carries from prior work; the
  decoding methods and neuroscience I'm learning as I go.
- **Edge claim is measured, not production** — Stage 2 exports to ONNX with a parity gate and benchmarks
  INT8 vs fp32; for these tiny nets (~26 KB) the honest finding is they're *already* edge-sized, with
  fp32 CPU inference ~0.4 ms — INT8 adds overhead rather than saving at this scale.

## What's here (Stage 0)
- **Eval harness** — every decoder is a `(fit_fn, score_fn)` pair fed through one spine; the evaluation
  *regime* (within / cross-subject / cross-session) is a criteria filter over the data cloud, so a run
  self-documents what it held out. Accuracy, Cohen's κ, ECE/Brier, per-subject diagnostics.
- **Decoders** — CSP+LDA baseline (the quarantine ceiling) + braindecode nets (EEGNet → ATCNet,
  EEGConformer) behind one GPU trainer (crop augmentation, exponential-moving standardization,
  seed-averaging).
- **Calibration under shift** — temperature scaling, ECE transfer from in-session val to cross-session
  test.
- **Reference ceilings** — published numbers as cited config to quarantine against, not chase.
- **Stage-2 tail** — ONNX export + INT8 quantization, parity-gated, with size/latency benchmarks.
- **Tracking** — guarded MLflow (local sqlite); per-run model cards.

## Data
Public non-invasive benchmarks (EEG/MEG), kept **outside the repo** (size + licensing) and pointed at
via a one-root path config — the sibling convention. Stage 0 uses **BCI Competition IV-2a** via
[MOABB](https://moabb.neurotechx.com/); the path config is cross-platform (Windows ↔ WSL auto-translated).

## Quickstart
```bash
uv sync                                  # + --extra track for MLflow, --extra export for ONNX
cp paths.example.yaml paths.yaml         # set the one data root (downloads land under <root>/raw)
uv run python -m neuroscan.experiments.run --method csp_lda --regime within --test-session 1test
uv run python -m neuroscan.experiments.run --method csp_lda --regime cross_subject   # the OOD gap
uv run pytest -q                         # the eval logic is unit + integration tested
```

## How it's built
Agent-driven build, human-owned judgment — coding agents scaffold the plumbing; the modeling
decisions, the measurement correctness, and the evaluation are mine. Architecture (two-layer engine +
science, split-as-criteria, dataset-adapter registry) is carried from a mature prior project of mine;
see [`docs/STRUCTURE.md`](docs/STRUCTURE.md). The neuroscience and decoding specifics I learn as I go.

## License
TBD (code will be permissively licensed). Any datasets used are **not** included and carry their own
licenses — obtained from their providers.
