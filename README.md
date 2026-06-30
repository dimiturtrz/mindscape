# mindscape — honest, efficient non-invasive neural decoding (EEG)

**What this is.** mindscape decodes **motor imagery** from non-invasive EEG and asks the question most
demos skip: *a decoder that scores ~60% on a subject's own recordings — how far does it fall on a person
it never saw?* On **BCI Competition IV-2a** (4-class motor imagery, 9 subjects, the standard benchmark),
a CSP+LDA decoder hits **0.598 within-subject** but drops to **0.382 leave-one-subject-out** — a
**22-point generalization gap** (chance = 25%). That measured gap, and the calibration under it, is the
contribution — not the headline accuracy. The second through-line is **deployability**: the decoders are
tiny and export to ONNX at millisecond-scale CPU latency.

It's also how **I'm** ramping into neural decoding: built on public data, the signal-processing /
time-series / calibration / edge-inference discipline carried from prior ML work, the **neuroscience and
the decoding methods learned as I go.** Status: **Stage 0** (a reproduced decode + a verified eval
harness) is done; Stage 1 (toward communication / semantic decoding) and Stage 2 (efficient on-device)
build on it. Full plan → **[docs/PLAN.md](docs/PLAN.md)**.

## See the signal the decoder reads — [neuroviz](neuroviz/)
![neuroviz — mu/beta ERD animated over a motor-imagery trial; the contralateral motor cortex desynchronizes, the C3↔C4 hot spot flipping by imagined hand](neuroviz/docs/media/demo.gif)

Imagining one hand **desynchronizes the opposite motor cortex** (mu/beta power drop over C3↔C4) — the
motor-imagery signature, animated over the trial. The viewer (dependency-free 2D topomaps + CSP spatial
patterns + C3/Cz/C4 waveforms) shows the signal the decoder consumes, the way the field actually looks at
it. → **[neuroviz/](neuroviz/)**

## The contribution — the generalization gap, measured
The science layer is **signal → preprocess → decode → evaluate**, and the *evaluation regime* is the
point. Every decoder is one `(fit_fn, score_fn)` pair fed through a single harness; the **regime** —
within-subject, cross-subject (leave-one-subject-out), cross-session — is a **criteria filter over the
data cloud**, so each run self-documents exactly what it held out. That's what separates a real
generalization number from an inflated one.

**The headline** (CSP+LDA, honest train-session → eval-session protocol):

| regime | accuracy | kappa | ECE |
|---|---|---|---|
| within-subject | **0.598** | 0.463 | 0.139 |
| **cross-subject (leave-one-subject-out)** | **0.382** | 0.176 | 0.135 |
| **gap** | **−0.216** | −0.287 | |

The mean understates it: per subject, cross-subject accuracy spans **0.24–0.54**, and three subjects land
**at or below chance** on a person they never saw. A "working" motor-imagery BCI is near-useless on
several unseen users — the trap the field underreports and any deployment hits first.

**Calibration under shift.** Temperature scaling fit on an in-session validation split, ECE measured
before/after on the *cross-session* test (ATCNet): test ECE **0.113 → 0.084**. We report the *transfer* —
whether an in-session calibration fix survives the session shift — not a single in-distribution ECE.
([`neuroscan/evaluation/calibrate.py`](neuroscan/evaluation/calibrate.py))

**Closing the cross-subject gap — measured, identified, fixed.** The collapse is a *domain shift*: each
subject's covariance cloud sits at a different location on the SPD manifold, so a classifier trained on
others misses them — not because the ERD contrast differs, but because the cloud is *displaced*. The
field's fix is **Riemannian re-centering** (Zanini et al. 2018): congruence-transport every subject's
covariances to the identity by their own Riemannian mean (`C → M⁻¹ᐟ² C M⁻¹ᐟ²` — the manifold version of
whitening), target included and **unsupervised**. We implemented it ([`neuroscan/experiments/align.py`](neuroscan/experiments/align.py)):

| method (leave-one-subject-out) | cross-subject acc |
|---|---|
| CSP+LDA | 0.382 |
| Riemann (tangent space) | 0.357 |
| Riemann ACM (time-delay cov) | 0.351 |
| **Riemann + re-centering** | **0.496** |

**+0.139** over plain tangent space — the displacement *was* the gap. And it's the *location*, not the
features: ACM (richer time-delay covariances) scores 0.351 alone and **0.470 even with re-centering** —
below plain re-centered tangent space (0.496). Removing the per-subject location shift is what transfers;
adding features on top doesn't. (Re-centering is unsupervised on the target → deployment-real.)

## The decoders — measured
We reproduce *standard* architectures (the decoder is commodity); the contribution is the eval rigor and
the efficient deployable, not a leaderboard number. **All our numbers sit below the published ceilings —
deliberately**: the honest train→eval-session protocol is harder than the pooled within-session CV many
papers report, and we don't do full per-model tuning or run-averaging. The gap analysis, grounded in
primary sources, is in [`research/`](research/deep_dives/2026-06-30_2a_sota_recipe.md).

Params + FLOPs at the real input (22 ch × 1125 samples, batch 1; FLOPs via fvcore, latency torch CPU
single-thread — `python -m neuroscan.models.profile`):

| model | role | params | FLOPs | CPU latency | within-subj acc | kappa |
|---|---|---|---|---|---|---|
| CSP+LDA | baseline | — | — | — | 0.598 | 0.463 |
| **Riemann (tangent space + LR)** | baseline | — | — | — | **0.706** | **0.609** |
| **EEGNet** | compact CNN | **3.7K** | 13.7M | 1.5 ms | 0.606 | 0.475 |
| **ATCNet** | attention + TCN | 114K | **2.8M** | 4.2 ms | 0.619 | 0.492 |
| EEGConformer | transformer | 871K | 72M | 4.2 ms | — | — |

Three honest findings fall out:
- **Classical geometry wins within-subject here.** Riemannian tangent-space + LR ([`baselines/riemann.py`](baselines/riemann.py))
  hits **0.706** — above both deep nets — the textbook BCI-2a result that treating each trial's *covariance*
  as a point on a curved manifold beats raw-waveform DL when per-subject data is tiny (~288 trials). But its
  *cross-subject* score is **0.357**, no better than CSP (0.382): plain tangent space doesn't transfer — the
  manifold **re-centering** that closes that gap is the next step, not implemented yet.
- **Tiny doesn't cost accuracy here.** The 3.7K-parameter EEGNet is within noise of the 30×-larger
  ATCNet (0.606 vs 0.619) on the same protocol — the edge-deployable model gives up essentially nothing.
- **Already edge-sized.** These nets are ~26 KB as ONNX with sub-ms inference; the Stage-2 tail exports
  with a **parity gate** (fp32 ONNX must match torch < 1e-3) and benchmarks INT8 — which *adds* overhead
  at this scale rather than saving. The deploy story isn't "shrink it," it's "already small, measured."
  ([`core/export_onnx.py`](core/export_onnx.py), [`neuroscan/experiments/quantize.py`](neuroscan/experiments/quantize.py))

**Published ceilings** (cited, not chased): FBCSP 0.65 · EEGNet 0.70 · ShallowConvNet 0.74 · ATCNet 0.81 ·
transformer SOTA 0.88; cross-subject SOTA 0.74.

## Honest limits (measured, not assumed)
Competent on a public benchmark, **not** a finished system:
- **Reproduction is partial.** Best within-subject ~0.62 vs published 0.81; clean subjects reproduce
  (A03 ~0.79 vs published peak ~0.85), hard subjects lag ~0.15 — documented in [`research/`](research/),
  not hidden. The contribution is the measured OOD gap + calibration + efficiency, not the peak.
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
# the strongest classical baseline — covariances on a Riemannian manifold:
uv run python -m neuroscan.experiments.run --method riemann --regime within
# a deep decoder, GPU:
uv run python -m neuroscan.experiments.run --method atcnet --regime within --resample 250 --fmin 4 --fmax 40
# the neuroviz demo:
uv run python -m neuroviz.export --subject 1 && python -m http.server 8000 -d neuroviz/web
uv run pytest -q
```
Runs log to a local MLflow (`uv run mlflow ui --backend-store-uri sqlite:///mlflow.db`) and write
`runs/<name>/` with an aggregate, a model card, and the run id.

## How motor imagery decodes — the ERD signature
The decodable signal is **event-related desynchronization (ERD)**: imagining a movement *suppresses* mu
(8–12 Hz) and beta (13–30 Hz) rhythms over the **contralateral** sensorimotor cortex — left-hand imagery
desynchronizes the right hemisphere (C4), right-hand the left (C3). CSP learns spatial filters that
maximize this variance contrast (its patterns localize over C3/C4, visible in neuroviz); deep nets learn
it end-to-end. The signature is **subject-specific** — the spatial pattern, the responsive band, and the
SNR all vary per person — which is precisely why cross-subject transfer collapses.

## Tests
```bash
uv run pytest -q          # unit (equivalence-class) + integration (module chains)
```
A pyramid: a wide unit base testing each module by equivalence class (metrics, the split-as-criteria
logic, transforms, calibration, the profiler), and an integration layer for the chains units can't cover
(data cloud → splits → harness end-to-end; decoder → ONNX export → parity).

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
