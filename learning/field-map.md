# Field map & coverage — what to learn, and how much

A **yardstick**: which fields this work sits on, which *view* of each we need, and a checklist to answer
"have we covered enough?" without guessing. Neural decoding is an intersection — the trick is taking each
field only to the depth that explains *our data and our claims*.

## Where this knowledge lives (the intersection)
| field | who owns it | what it explains for us | our depth |
|---|---|---|---|
| **Neurophysiology / electrophysiology** | neuroscientists | how neurons sum to scalp voltage; why mu/beta rhythms exist | *enough to explain the signal*, not cellular biophysics |
| **Hemodynamics / tissue optics** (fNIRS) | neuroscientists / BME | neurovascular coupling, HbO/HbR, near-infrared light in tissue, the slow HRF | *enough to explain the fNIRS signal* (M5) — the second modality |
| **Systems / cognitive neuroscience** (motor) | neuroscientists | sensorimotor cortex, motor imagery vs execution, **ERD** | the motor-system slice only |
| **Biophysics / instrumentation** | BME / EE | electrodes, volume conduction, artifacts (EOG, line noise) | *enough to know the failure modes* |
| **Signal processing (DSP)** | EE / DSP | Fourier/spectral, filtering, **time-frequency** — the core tool | working fluency — this is our hands |
| **Linear algebra** | maths | covariance, eigendecomposition → **CSP**, spatial filters, ICA | working fluency (the CSP math) |
| **Machine learning** | ML | classification, CNNs/transformers, **generalization** | our home turf (carries from prior work) |
| **Statistics** | stats | evaluation, **calibration**, cross-validation, the OOD gap | working fluency — the honesty layer |

The lane is **neural engineering / BCI** = neuroscience (what's there) × DSP + linear algebra (how to
extract it) × ML + stats (how to decode + judge it honestly). Our deliverable lives in the last two; we
take the neuroscience + DSP to the depth that makes the data and its failure modes legible.

## Target competency (the bar) — three layers
**F) Foundations** (under everything): linear algebra (covariance, eigenvectors → CSP) · Fourier / spectral
analysis (power in a band) · probability + stats (cross-entropy, ECE, cross-validation variance). Working
fluency, not proofs.

**N) Neuroscience-enough**: the motor system (where/what), motor imagery vs execution, **ERD/ERS** (the
decodable signal), how scalp EEG is generated + its artifacts, the standard rhythms (mu/beta). *Enough to
explain the signal and why decoding it is hard — NOT to do wet-lab neuroscience.*

**D) Decoding stack** (the real work): preprocessing (filter/epoch/standardize) · spatial filtering (CSP) ·
decoders (CSP+LDA → EEGNet/ATCNet) · **honest evaluation** (within vs cross-subject, calibration,
per-subject diagnostics) · efficient deployment (ONNX, quantization). This is the contribution.

## Coverage checklist (✅ theory done · ⬜ pending)
- [x] the task: 4-class motor imagery, the cue paradigm, why imagery (`01`)
- [x] motor system + imagery-vs-execution + ERD mechanism (`01`)
- [x] EEG genesis (pyramidal cells, volume conduction) + bands + artifacts (`02`)
- [x] spectral power, time-frequency, ERD as a power drop (`02`)
- [x] CSP / spatial filtering — the linear-algebra core (`02`)
- [x] decoders CSP+LDA / EEGNet / ATCNet; the eval regimes; calibration (`03`)
- [x] the cross-subject gap + why it happens; efficiency/deploy (`03`)
- [x] **fNIRS fundamentals** — hemodynamics, neurovascular coupling, Beer–Lambert, the slow HRF, systemic noise (`M5`)
- [x] **fNIRS decoding** — why amplitude features (mean/slope/peak) not covariance; the method–signal match (`M5` synthesis)
- [ ] (Stage 3) EEG-vs-fNIRS same-task comparison + fusion — when we pull the EEG n-back half
- [ ] (Stage 3) semantic / speech decoding — when we get there

## Task ladder — what each EEG task needs (and what's covered)
The fields above are shared; each *task* (see [`research/.../tasks_datasets_landscape.md`](../research/deep_dives/2026-06-30_tasks_datasets_landscape.md))
adds its own signal + method knowledge. What the lessons cover vs what a task would add:

| task | the signal (extra to learn) | the method (extra) | status |
|---|---|---|---|
| **Motor imagery** (EEG, Stage 1) | **ERD** (mu/beta, contralateral) — lessons 01–02 | **CSP** + EEGNet/ATCNet — lessons 02–03 | ✅ **covered** |
| **n-back workload** (fNIRS, Stage 2) | **hemodynamic response** (HbO↑/HbR↓, slow HRF) — `M5` | **amplitude features** (mean/slope/peak) + LDA — *not* covariance — `M5` | ✅ **covered** |
| efficiency / on-device (optional) | — | ONNX, quantization, parity — lesson 03.5 | ✅ covered |
| **Visual / semantic** (Stage 1) | visual-evoked responses; CLIP-style representational decoding | contrastive / embedding regression, zero-shot retrieval | ⬜ Stage 1 |
| **Speech / language** (Stage 1+) | cortical speech tracking; MEG ≫ EEG here | seq2seq, contrastive decoding | ⬜ Stage 1+ |
| P300 / ERP | the P300 oddball ERP (time-domain, not band power) | template / Riemannian (xDAWN) | ⬜ cite-only (pruned) |
| SSVEP / cVEP | steady-state response to flicker frequency | CCA / FBCCA (frequency match) | ⬜ cite-only (pruned) |
| affective / AAD | longer-window state; auditory envelope tracking | classification / stimulus-reconstruction | ⬜ cite-only (pruned) |

**Read:** the *foundations* (F) and the *honest-eval / decoding stack* (D) transfer across **all** tasks —
that's why the cross-subject-gap + calibration + efficiency lessons are paradigm-general. The *neuroscience-
enough* (N) layer is **per-task**: ERD is MI-specific; Stage 1 swaps in visual-evoked / speech-tracking
signal knowledge. So the curriculum covers our chosen ladder (MI → efficiency → the Stage-1 target), not the
pruned paradigms — which we cite, not build.

## The interview bar (what to be able to say cold)
1. **The task** in one sentence + why imagery (BCI for people who can't move).
2. **The signal**: ERD — mu/beta power drops over contralateral motor cortex during (imagined) movement.
3. **The contribution**: the measured within→cross-subject gap (0.598 → 0.382) + calibration — not accuracy.
4. **The methods**: CSP+LDA baseline, EEGNet/ATCNet deep, all commodity; the eval is the point.
5. **The honest limits**: below published SOTA deliberately; reproduction partial; why.
6. **The efficiency angle**: tiny nets (EEGNet 3.7K params) ≈ big ones on our protocol; edge-deployable.
7. **The second modality (fNIRS)**: a *different signal and task* — hemodynamic (HbO/HbR via near-infrared
   light, neurovascular coupling), decoding **mental workload** (n-back). Passive BCI (read state) vs motor
   imagery's active BCI (send command).
8. **Method must match the signal's physics** (the cross-modality insight): EEG's class signal is in the
   **covariance** → CSP/Riemann; fNIRS's is in the **mean HbO amplitude** → mean/slope/peak features + LDA.
   Covariance methods sit at *chance* on fNIRS because centering discards the mean. Different physics →
   different tool, deliberately.
