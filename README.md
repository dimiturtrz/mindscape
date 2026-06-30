# mindscape — honest, efficient non-invasive neural decoding (EEG)

**What this is.** mindscape decodes **motor imagery** from non-invasive EEG and asks the question most
demos skip: *a decoder that scores ~60% on a subject's own recordings — how far does it fall on a person
it never saw?* On **BCI Competition IV-2a** (4-class motor imagery, 9 subjects, the standard benchmark),
a CSP+LDA decoder hits **0.598 within-subject** but collapses to **0.382 leave-one-subject-out** — a
**22-point generalization gap** (chance = 25%). That measured gap, and the calibration under it, is the
contribution — not the headline accuracy. The second through-line is **deployability**: the decoders
export to ONNX and run at sub-millisecond CPU latency.

It's also how **I'm** ramping into neural decoding: built on public data, the signal-processing /
time-series / calibration / edge-inference discipline carried from prior ML work, the **neuroscience and
the decoding methods learned as I go.** Status: **Stage 0** (a reproduced decode + a verified eval
harness) is done; Stage 1 (toward communication / semantic decoding) and Stage 2 (efficient on-device)
build on it. Full plan → **[docs/PLAN.md](docs/PLAN.md)**.

## See the signal the decoder reads — [neuroviz](neuroviz/)
![neuroviz — mu/beta ERD animated over a motor-imagery trial; the contralateral motor cortex desynchronizes, the C3↔C4 hot spot flipping by imagined hand](neuroviz/docs/media/demo.gif)

Imagining one hand **desynchronizes the opposite motor cortex** (mu/beta power drop over C3↔C4) — the
motor-imagery signature, animated over the trial. The viewer (dependency-free 2D topomaps + CSP spatial
patterns + C3/Cz/C4 waveforms) shows the signal the decoder consumes, the way the field actually looks
at it. → **[neuroviz/](neuroviz/)**

## The contribution — the eval, the gap, the calibration
The science layer is **signal → preprocess → decode → evaluate**, and the *evaluation regime* is the
point. Every decoder is one `(fit_fn, score_fn)` pair fed through a single harness; the **regime** —
within-subject, cross-subject (leave-one-subject-out), cross-session — is a **criteria filter over the
data cloud**, so each run self-documents exactly what it held out. That's what separates a real
generalization number from an inflated one.

### The headline — within → cross-subject (CSP+LDA, honest train-session → eval-session protocol)
| regime | accuracy | kappa | ECE |
|---|---|---|---|
| within-subject | **0.598** | 0.463 | 0.139 |
| **cross-subject (leave-one-subject-out)** | **0.382** | 0.176 | 0.135 |
| **generalization gap** | **−0.216** | −0.287 | |

A decoder at ~60% on its own data drops to **38%** on an unseen subject. The mean understates it:
**per subject, cross-subject accuracy spans 0.24–0.54**, and subjects A02/A05/A06 land **at or below
chance** when tested on a person they never saw. A "working" motor-imagery BCI is near-useless on several
unseen users — the trap the field underreports and any real-world deployment hits first.

### Calibration under shift
Temperature scaling fit on an in-session validation split, ECE measured before/after on the cross-session
test (ATCNet): test ECE **0.113 → 0.084**. We report the *transfer* honestly — whether an in-session
calibration fix survives the session shift — rather than a single in-distribution ECE.
([`neuroscan/evaluation/calibrate.py`](neuroscan/evaluation/calibrate.py))

## Decoders — ours vs published (reproduce, don't chase)
The plan's first rule: **don't chase the leaderboard** — reproduce a standard method, contribute the eval
rigor + the efficient deployable. The decoder is commodity; we say so.

### Deployable vs near-SOTA — both ours, measured
Params + FLOPs at the real 2a input (22 ch × 1125 samples, batch 1; FLOPs via fvcore, latency torch CPU
single-thread — `python -m neuroscan.models.profile`):

| model | params | FLOPs | CPU latency | within-subj acc | kappa |
|---|---|---|---|---|---|
| CSP+LDA (baseline) | — | — | — | 0.598 | 0.463 |
| **EEGNet** (edge-deployable) | **3.7K** | 13.7M | 1.5 ms | 0.606 | 0.475 |
| **ATCNet** (near-SOTA) | 114K | **2.8M** | 4.2 ms | **0.619** | 0.492 |
| EEGConformer | 871K | 72M | 4.2 ms | — | — |

**The 3.7K-parameter EEGNet ties the 30×-larger ATCNet** (0.606 vs 0.619) on our honest protocol — the
edge-deployable model gives up ~nothing in accuracy, at ~26 KB ONNX and sub-ms inference (ONNX runtime).
ATCNet carries more parameters but is the most FLOP-efficient (2.8M, its sliding-window TCN).

### vs published ceilings (cited, not chased)
| FBCSP | EEGNet | ShallowConvNet | ATCNet | transformer SOTA | cross-subject SOTA |
|---|---|---|---|---|---|
| 0.65 / κ0.57 | 0.70 / κ0.61 | 0.74 / κ0.65 | 0.81 / κ0.76 | 0.88 / κ0.84 | 0.74 |
| Ang 2012 | Lawhern 2018 | Schirrmeister 2017 | Altaheri 2023 | Sci. Rep. 2025 | EEGEncoder 2024 |

**We sit ~19 points under the published ATCNet, transparently.** The reproduction is honest about *why*:
matching the real recipe (continuous-signal exponential-moving standardization vs per-epoch, `StandardScaler`
not EMS, the 1.5–6 s / 1125-sample window, ATCNet's own internal sliding-window augmentation, batch 64)
lifted our ATCNet **0.567 → 0.619** and confirmed the architecture matches braindecode's. The residual
gap is concentrated on the hard subjects and traces to single-run-vs-their-10-run-mean + Adam-vs-AdamW —
diminishing returns we deliberately did not chase (the contribution is the eval, not the peak). Full
analysis, grounded in primary sources → [`research/deep_dives/2026-06-30_2a_sota_recipe.md`](research/deep_dives/2026-06-30_2a_sota_recipe.md).

## Efficient deploy — the edge angle ([Stage 2](core/export_onnx.py))
Trained decoders export to ONNX with a **parity gate** (fp32 ONNX must match torch < 1e-3 before the
quantized model is trusted) and benchmark INT8 vs fp32 on CPU. The honest finding: these EEG nets are
**already minuscule** (~26 KB), running at **~0.38 ms/inference** on a single CPU thread — so dynamic INT8
*adds* overhead rather than saving at this scale. The deployable story here isn't "shrink it," it's
"already edge-sized, measured." ([`neuroscan/experiments/quantize.py`](neuroscan/experiments/quantize.py))

## Honest limits (measured, not assumed)
Competent on a public benchmark, **not** a finished system:
- **Below published SOTA on raw decode, deliberately.** Best within-subject ~0.62 vs published 0.81–0.88;
  cross-subject 0.382 vs 0.74. We reproduce *standard* methods through an *honest* protocol (the harder
  train→eval-session split, not pooled within-session CV) — the contribution is the measured OOD gap +
  calibration + efficiency, all of which SOTA papers underreport.
- **Reproduction is partial.** Clean subjects reproduce (A03 ~0.79 vs published peak ~0.85); hard subjects
  lag ~0.15 — documented, not hidden.
- **Neuroscience is a ramp.** The signal-processing / eval discipline carries from prior work; the
  decoding methods and neuroscience are learned as I go.
- **Not a device.** Public research data only; no real-time online BCI, no clinical or prospective validation.

## Data
**BCI Competition IV-2a** — 9 subjects, 4-class motor imagery (left/right hand, feet, tongue), 22 EEG
channels @ 250 Hz, 2 sessions × 288 trials — pulled via **[MOABB](https://moabb.neurotechx.com/)** and
kept **outside the repo** (size + licensing). Per-dataset adapters remap to a canonical schema and cache
epochs to a recipe-keyed store; splits are queries over that cloud. One data root, set once:
```bash
cp paths.example.yaml paths.yaml      # then: data: <abs path to a data dir outside the repo>
```
Downloads land under `<root>/raw/`; the epoch cache under `<root>/processed/` (created on first run).

## Quickstart
```bash
uv sync                                              # .venv from pyproject + uv.lock; prefix commands with `uv run`
cp paths.example.yaml paths.yaml                     # set the one data root
# the headline contrast — the same decoder, two regimes:
uv run python -m neuroscan.experiments.run --method csp_lda --regime within --test-session 1test
uv run python -m neuroscan.experiments.run --method csp_lda --regime cross_subject   # the OOD gap
# strong decoder, GPU:
uv run python -m neuroscan.experiments.run --method atcnet --regime within --resample 250 --fmin 4 --fmax 40
# the neuroviz demo:
uv run python -m neuroviz.export --subject 1 && python -m http.server 8000 -d neuroviz/web
uv run pytest -q                                     # the eval logic is unit + integration tested
```
Runs log to a local MLflow (`uv run mlflow ui --backend-store-uri sqlite:///mlflow.db`) and write
`runs/<name>/` with an aggregate, a model card, and the run id.

## Motor imagery & the ERD signature
The decodable signal in motor imagery is **event-related desynchronization (ERD)**: imagining a movement
*suppresses* mu (8–12 Hz) and beta (13–30 Hz) rhythms over the **contralateral** sensorimotor cortex —
left-hand imagery desynchronizes the right hemisphere (C4), right-hand the left (C3). CSP learns spatial
filters that maximize this variance contrast (its patterns localize over C3/C4 — visible in neuroviz),
and deep nets learn it end-to-end. The honest-eval problem is that this signature is **subject-specific**:
the spatial pattern, the responsive band, and the SNR all vary per person, which is exactly why
cross-subject transfer collapses.

## Tests
```bash
uv run pytest -q          # unit (equivalence-class) + integration (module chains)
```
A pyramid — a wide unit base testing each module's behaviour by equivalence class (metrics, the
split-as-criteria logic, transforms, calibration), and an integration layer testing the chains that
units can't (data cloud → splits → harness end-to-end; decoder → ONNX export → parity).

## How it's built
Agent-driven build, human-owned judgment — coding agents scaffold the plumbing; the modeling decisions,
the measurement correctness, and the evaluation are mine. The architecture (two-layer engine + science,
split-as-criteria, dataset-adapter registry, calibration-under-shift) is carried from a mature prior ML
project of mine; see [`docs/STRUCTURE.md`](docs/STRUCTURE.md). The neuroscience and decoding specifics I
learn as I go.

## References
- **BCI Competition IV-2a** — Tangermann et al., *Review of the BCI Competition IV*, Front. Neurosci. 2012.
- **CSP / FBCSP** — Ang et al., *Filter Bank Common Spatial Pattern (FBCSP) in BCI*, IJCNN 2008 / 2012.
- **EEGNet** — Lawhern et al., *EEGNet: a compact CNN for EEG-based BCIs*, J. Neural Eng. 2018.
- **ShallowConvNet / Deep4Net** — Schirrmeister et al., *Deep learning with CNNs for EEG decoding*, HBM 2017.
- **ATCNet** — Altaheri et al., *Physics-informed attention temporal CNN for EEG-based MI classification*, IEEE TII 2023.
- **MOABB** — Jayaram & Barachant, *MOABB: trustworthy algorithm benchmarking for BCIs*, J. Neural Eng. 2018.
- **Braindecode** — the PyTorch EEG-decoding library the deep models are built on.

## License
MIT — see [LICENSE](LICENSE).
