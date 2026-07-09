# mindscape — robust, efficient non-invasive neural decoding (EEG + fNIRS)

**What is the format of our thoughts?** Ripples on the mind — a projection of the brain, the most precious
organ we have. What happens in there when we read a word, see a face, decide to move a hand? The field that
reads (and sometimes writes) those signals is **BCI — brain–computer interface** — in two flavors, invasive
and non-invasive.

**Invasive** BCI implants electrodes in the brain: far better resolution, but you can't take it back out, it
reads *and* writes, and the dystopian failure mode writes itself (the *1984* kind). **Non-invasive** BCI reads
from outside — a cap or a scanner — trading resolution for the fact that you can take it off. The two wearable,
affordable options are **EEG** and **fNIRS**:

- **EEG** reads the *cumulative electrical activity* at the scalp — summed action potentials of large neuron
  populations. Fast, spatially blurry.
- **fNIRS** reads *blood*: it shines two infrared wavelengths into cortex and measures the scatter, tracking
  oxygenated vs deoxygenated hemoglobin. A firing region burns ATP, blood rushes in with fresh oxygen — so
  fNIRS sees the *hemodynamic echo* of activity. Slower, spatially sharper.

Complementary — opposite trades on the same question. **mindscape** explores both: the data, its generation,
and the feature-extraction and decoding methods for real tasks — and asks what most demos skip: *a decoder
that scores ~60% on a subject's own recordings — how far does it fall on a person it never saw?* The
contribution is the **robust cross-subject evaluation** (and what it reveals), not a leaderboard number —
across two tasks:

- **Motor imagery** (EEG, BCI-2a). CSP+LDA hits **0.598 within-subject** but drops to **0.391**
  leave-one-subject-out — a **21-point generalization gap** (chance 25%) — which Riemannian **re-centering**
  then largely closes (→ **0.501** zero-shot, ~half the gap; **0.650** with light calibration). The gap
  measured *and* substantially closed, not eliminated.
- **Mental workload** (Shin n-back — EEG · fNIRS · fusion). Both modalities decode the *same* task, so
  differences are the modality not the task. The **same re-centering** from Task 1 lifts EEG **0.45 → 0.60**
  cross-subject — the biggest workload gain, making fusion a **strong + weak** pair. The two still fail on
  *different* blocks (oracle **0.75** vs best-single **0.58**), so the **+17-pt** complementarity is real —
  but the fusion methods we've tried capture almost none of it (product edges best-single by +1.5 pp, a wash).
  The graded 2-vs-3 contrast is at a physiological ceiling (fNIRS reads WM *engagement*, not *level*); capturing
  the rest (boundary-aware / source-space fusion) is untested, not disproven.

Two through-lines under both tasks: the **evaluation regime is the product** — a split is a *criteria filter*
over the data cloud, so every run self-documents exactly what it held out — and **deployability**: the
decoders are edge-tiny and export to ONNX at millisecond-scale CPU latency.

It's also how **I'm** ramping into neural decoding: built on public data, the signal-processing /
calibration / edge-inference discipline carried from prior ML work, the **neuroscience and decoding methods
learned as I go.** Next: a harder, richer task (semantic decoding) where the signal is present — the fNIRS
graded-*level* result is a measured physiological ceiling (the fusion of its real complementarity stays open),
not an open promise dressed as a win. Full plan →
**[docs/PLAN.md](docs/PLAN.md)**.

## See the signal the decoder reads — [neuroviz](neuroviz/)
![neuroviz — EEG+fNIRS fusion brain-camera (Shin n-back): the fused surface-video — raw EEG band-power + the fNIRS CBSI neural map, and the locality-gated joint firing pattern, with the hemodynamic lag derived per subject](neuroviz/docs/media/demo_fusion.webp)

One dependency-free viewer, organized **task → modality**. **Motor imagery** → EEG (mu/beta ERD topomaps +
CSP/Riemann patterns). **Mental workload** → **EEG** band-power, **fNIRS** hemodynamic response, and
**Fusion** — the EEG+fNIRS **brain-camera**: raw EEG band-power + the fNIRS CBSI neural map → a locality-gated
joint firing pattern (hemodynamic lag derived per subject). Single-modality views show the signal a decoder
consumes *and whether it got it right*; the fusion view shows the **physics** — a visualization, not (yet) a
decode win. → **[neuroviz/](neuroviz/)**

## Mapping the field — a working frame for where these tasks sit
This is my attempt to organize non-invasive decoding as I learn it — a working frame, not a settled taxonomy;
others carve it differently. It reads as a ladder ordered by **what** you decode off the signal, where value
seems to climb and tractability to fall as you go up — because the higher rungs lean **endogenous**
(self-generated, weak, un-timed) and the lower ones **exogenous** (stimulus-driven, strong, time-locked,
averageable). The rough principle I keep coming back to: *a decoder works to the degree the signal is driven
by something known.* Corrections welcome.

| axis | what you decode | output | example paradigms | tractability |
|---|---|---|---|---|
| **Control** | intent to *act* | commands (move, select) | motor imagery, P300, SSVEP | works, but low-bandwidth + often gaze-bound |
| **State** | cognitive / affective *state* | monitoring (load, drowsiness) | mental workload / n-back — *passive BCI* | per-subject seems doable; graded levels appear to hit physiological ceilings |
| **Perception** | the *stimulus* being perceived | reconstruct what you see / read | image viewing, reading | **measured** — within decodes, cross lifts with subject count |
| **Communication** | inner *meaning / intent* | language, imagined content | imagined speech / text | the frontier — mostly endogenous, seems hard |

**mindscape works up this ladder.** Three rungs now measured cross-subject (no leaderboard cherry-picking):
**motor imagery = control**, **n-back = state**, and now **perception** — EEG→image on THINGS-EEG2 (a
NICE-style CLIP-retrieval decoder), where within-subject decodes (concept-avg top1 **15%**, 30× chance) but
cross-subject craters and then *lifts with training-subject count* (**2%** train-1 → **6%** train-4 LOSO) —
the same subject-generalization story as motor imagery, in perception
(→ **[neuroscan/tasks/visual/](neuroscan/tasks/visual/)**). **Communication** (EEG→text) is the follow-on
frontier-probe — decode the *reading* phase where signal is real, treat imagined "telepathy" as a probe, not
a promise. The climb → **[docs/PLAN.md](docs/PLAN.md)**.

## Task · Motor imagery (BCI-2a, EEG) — the generalization gap, measured
The science layer is **signal → preprocess → decode → evaluate**, and the *evaluation regime* is the
point. Every decoder is one `(fit_fn, score_fn)` pair fed through a single harness; the **regime** —
within-subject, cross-subject (leave-one-subject-out), cross-session — is a **criteria filter over the
data cloud**, so each run self-documents exactly what it held out. That's what separates a real
generalization number from an inflated one.

**The headline** (CSP+LDA, robust train-session → eval-session protocol):

| regime | accuracy | kappa | ECE |
|---|---|---|---|
| within-subject | **<!--r:csp_lda_within_bnci2014_001.acc-->0.598<!--/r-->** | <!--r:csp_lda_within_bnci2014_001.kappa-->0.464<!--/r--> | <!--r:csp_lda_within_bnci2014_001.ece-->0.140<!--/r--> |
| **cross-subject (leave-one-subject-out)** | **<!--r:csp_lda_cross_subject_bnci2014_001.acc-->0.391<!--/r-->** | <!--r:csp_lda_cross_subject_bnci2014_001.kappa-->0.189<!--/r--> | <!--r:csp_lda_cross_subject_bnci2014_001.ece-->0.134<!--/r--> |
| **gap** | **<!--r:csp_lda_cross_subject_bnci2014_001.acc-csp_lda_within_bnci2014_001.acc-->−0.206<!--/r-->** | <!--r:csp_lda_cross_subject_bnci2014_001.kappa-csp_lda_within_bnci2014_001.kappa-->−0.275<!--/r--> | |

The mean understates it: per subject, cross-subject accuracy spans **0.24–0.55**, and one subject lands
**below chance** on a person it never saw (two more within a few points of it). A "working" motor-imagery
BCI is near-useless on several unseen users — the trap the field underreports and any deployment hits first.

**Calibration under shift.** Temperature scaling fit on an in-session validation split, ECE measured
before/after on the *cross-session* test (ATCNet): test ECE **0.113 → 0.084**. We report the *transfer* —
whether an in-session calibration fix survives the session shift — not a single in-distribution ECE.
([`neuroscan/evaluation/calibrate.py`](neuroscan/evaluation/calibrate.py))

**Closing the cross-subject gap — the RPA ladder.** The collapse is a **domain shift**: each subject's
covariance cloud sits at a different *location* on the SPD manifold, so a classifier trained on others misses
them. **Riemannian re-centering** whitens that displacement away per-subject; we report each rung by how many
*target* labels it needs — the deployability axis ([`align.py`](neuroscan/tasks/motor_imagery/align.py)):

| method (leave-one-subject-out) | target labels | cross-subject acc |
|---|---|---|
| CSP+LDA | — | <!--r:csp_lda_cross_subject_bnci2014_001.acc-->0.391<!--/r--> |
| Riemann (tangent space) | — | <!--r:riemann_cross_subject_bnci2014_001.acc-->0.360<!--/r--> |
| **+ re-centering** (RPA step 1, Zanini 2018) | **zero-shot** | **<!--r:riemann_recenter_ts_bnci2014_001.acc-->0.501<!--/r-->** |
| **+ re-scaling** (RPA step 2) | **zero-shot** | **<!--r:riemann_recenter_scale_ts_bnci2014_001.acc-->0.519<!--/r-->** |
| **full RPA** (+ re-rotate, step 3) | calib 10 % | <!--r:riemann_rpa_c10_bnci2014_001.acc-->0.555<!--/r--> |
| **full RPA** | calib 20 % | <!--r:riemann_rpa_c20_bnci2014_001.acc-->0.595<!--/r--> |
| **full RPA** | calib 50 % | **<!--r:riemann_rpa_ts_bnci2014_001.acc-->0.650<!--/r-->** |
| MDWM | calib 50 % | <!--r:riemann_mdwm_ts_bnci2014_001.acc-->0.412<!--/r--> |

**Zero-shot** (no target labels — deployment-real) re-centering closes most of the gap, **0.36 → 0.50**, and
re-scaling nudges it to 0.52 — the displacement *was* the gap. **Calibrated**, a supervised re-rotation on even
10 % of a session reaches 0.555, scaling to **0.650** at 50 %, near the within-subject ceiling. The mechanism,
the MDWM fragility negative, the decoder comparison (params / FLOPs / latency), and why our numbers sit below
published SOTA → **[motor_imagery/](neuroscan/tasks/motor_imagery/)**.

*Grounding — challenge [BCI Competition IV-2a](https://www.bbci.de/competition/iv/) · data via
[MOABB](https://moabb.neurotechx.com/) · claimed SOTA [ATCNet](https://github.com/Altaheri/EEG-ATCNet) 0.81
(FBCSP 0.65 · EEGNet 0.71 · transformer 0.88; cross-subject 0.74) vs our measured 0.60 / 0.62 within · homework
[2a SOTA-gap analysis](research/deep_dives/2026-06-30_2a_sota_recipe.md),
[MI neuroscience](research/deep_dives/2026-06-30_motor-imagery-neuroscience.md).*

## Task · Mental workload / n-back (Shin) — one task, three approaches: EEG · fNIRS · fusion
Decode **mental workload** — which n-back load (0/2/3-back) a subject holds in working memory — from the
Shin hybrid set, where EEG and fNIRS were recorded **simultaneously**. So both modalities decode *one
identical task*, and any difference below is the **modality, not the task** — the clean comparison the
motor-imagery EEG couldn't give (different task, different chance). Same harness; only the adapter + decoder
change.

**n-back workload** (Shin · 26 subjects · 3-class · **chance 0.333**):

| modality · method | cross-subject (LOSO) | within (held-out block-series) |
|---|---|---|
| fNIRS · mean+slope+peak → LDA | <!--r:fnirs_lda_cross_subject_shin2017_nback.acc-->0.454<!--/r--> (κ 0.16) | 0.415 (κ 0.12) |
| EEG · CSP + LDA | <!--r:csp_lda_cross_subject_shin2017_nback_eeg.acc-->0.432<!--/r--> (κ 0.12) | <!--r:csp_lda_within_shin2017_nback_eeg.acc-->0.568<!--/r--> (κ 0.35) |
| EEG · Riemann (tangent space) | <!--r:riemann_cross_subject_shin2017_nback_eeg.acc-->0.452<!--/r--> (κ 0.14) | <!--r:riemann_within_shin2017_nback_eeg.acc-->0.538<!--/r--> (κ 0.31) |
| **EEG · Riemann + re-centering** (the transfer fix) | **<!--r:riemann_recenter_ts_shin2017_nback_eeg.acc-->0.604<!--/r-->** (κ 0.41) | — |

**The transfer fix carries across tasks — and hits harder here.** The same **Riemannian re-centering** that
closed the motor-imagery gap lifts workload EEG cross-subject **0.452 → 0.604** (+15 pp, zero-shot, no target
labels) — the biggest single workload gain, and now the strongest single-modality decoder here (0.60 vs fNIRS
0.47). The near-floor EEG numbers were *unaligned* covariance clouds; re-centering removes the per-subject
location shift as it did for MI. That re-centered 0.60 even *exceeds* within-subject Riemann (0.54) — 25
aligned subjects beat a data-starved personal model.

**Anchored to the field's robust benchmark.** [BenchNIRS](https://doi.org/10.3389/fnrgo.2023.994969)
(Benerradi 2023) shows *proper* cross-subject fNIRS evaluation is near-chance — most published accuracies are
inflated by within-session / personalised validation. On this exact n-back it reports LDA 0.389; we reproduce
it (0.392, [`repro_benchnirs`](neuroscan/tasks/workload/repro_benchnirs.py)) and our `fnirs_lda` reaches
**<!--r:fnirs_lda_cross_subject_kfold_shin2017_nback.acc-->0.474<!--/r-->** under its matched protocol
(+8.2 pp) — from full spatial resolution + shrinkage + more data, not a new method. Still only ~14 pp above the
0.333 floor, an order of magnitude short of what leaky validation produces.

Behind these headlines, two investigations (both → **[workload/](neuroscan/tasks/workload/)**): the fNIRS
signal is **shape, not magnitude** — a feature-importance search finds the slope *trajectory* ties the full
15-family bank while `mean`/`peak` barely clear chance; and the graded **2-vs-3 contrast is at a physiological
ceiling** (at chance even *within* subject — fNIRS reads WM *engagement*, not *level*), a rigorous negative
confirmed four ways (feature bank, windowing, CBSI cleaning, GLM-β).

### Fusion — a strong + weak pair; complementarity real, barely captured
Both decoders run on the **same aligned epochs** — EEG re-centered Riemann + fNIRS mean/slope/peak — under one
**5-fold GroupKFold** (fusion needs per-epoch EEG↔fNIRS pairing, which LOSO's single-subject test sets make too
small to read):

| role (5-fold GroupKFold, matched folds) | acc | vs best single |
|---|---|---|
| chance | 0.333 | — |
| fNIRS (mean/slope/peak → LDA) | <!--r:fusion_cross_subject_kfold_shin2017_nback.fnirs-->0.474<!--/r--> | −0.106 |
| **EEG (re-centered Riemann)** | **<!--r:fusion_cross_subject_kfold_shin2017_nback.eeg-->0.580<!--/r-->** | best |
| Late fusion (avg probabilities) | <!--r:fusion_cross_subject_kfold_shin2017_nback.late-->0.587<!--/r--> | **+0.007** |
| Feature fusion (concat → LDA) | <!--r:fusion_cross_subject_kfold_shin2017_nback.feature-->0.564<!--/r--> | −0.016 |

Re-centering flips EEG to the **strong** modality (0.58 vs fNIRS 0.47), so this is a **strong + weak** pair.
Late fusion edges best-single by +0.7 pp — within fold noise. But the two **fail on different blocks**: a
per-trial oracle picking the right modality would hit **<!--r:fusion_cross_subject_kfold_shin2017_nback.oracle_either-->0.752<!--/r-->**
(**+17 pts**, near-independent errors φ ≈ 0.11), so the complementarity is **real and large**. Yet every
output-space combiner we swept (product best, +1.5 pp) and an input-level gate capture almost none of it —
lifting the weak modality is closed (fNIRS at the physiological ceiling), leaving **boundary-aware routing and
source-space fusion** the open routes. Full combiner sweep, the gate, the z-score confirmation, and the
literature caveats → **[workload/](neuroscan/tasks/workload/)**.

*Grounding — benchmark [BenchNIRS](https://doi.org/10.3389/fnrgo.2023.994969) LDA 0.389 vs our measured 0.474 ·
data [Shin 2017](https://doi.org/10.14279/depositonce-5830.2)
([MOABB](https://moabb.neurotechx.com/docs/generated/moabb.datasets.Shin2017A.html)) · claimed fusion SOTA
96–98 % (within-subject; leakage-free LOSO 62.6 %) [fNIRS-T](https://github.com/wzhlearning/fNIRS-Transformer) ·
homework: [fNIRS decoding SOTA](research/deep_dives/2026-07-01_fnirs_decoding_sota.md),
[EEG-fNIRS fusion SOTA](research/deep_dives/2026-07-01_eeg_fnirs_fusion_sota.md),
[fNIRS datasets](research/deep_dives/2026-07-06_fnirs_datasets_benchmarks.md).*

## Task · Perception (EEG→image on THINGS) — the over-reporting audit

Third rung of the ladder: decode the *seen image* from EEG. A NICE-style encoder maps each EEG epoch to the
viewed image's CLIP embedding (InfoNCE), then retrieves zero-shot among the 200 held-out THINGS test concepts
(chance **0.5%**). The contribution is **not** a leaderboard top-k — it's a bias audit of *how much the field's
usual numbers inflate*.

**The inflation grid** ([`retrieval_audit.py`](neuroscan/tasks/visual/retrieval_audit.py)) — the same encoder
scored four ways, the commonly-quoted cell vs the defensible one:

| top-1 (top-5) | single-trial | concept-averaged |
|---|---|---|
| **within-subject** | 4.0% (14.5%) | **14.8% (39.5%)** ← usually quoted |
| **cross-subject** | **1.9% (7.6%)** ← robust | 4.8% (14.3%) |

*(measured, 2-subject mean; chance 0.5%.)* Two independent leaks stack — seeing the test *person* and averaging
test *repeats* — for an **8.0× gap** (14.8% vs 1.9%, +12.9 pts) between the quoted headline and the defensible
number. That top-left cell *is* the field's usual headline (NICE-family range); the deployment-real
cross-subject single-trial number is 8× lower. Same subject-generalization story as motor imagery, now in
perception. The zero-shot disjointness
check, the confidence calibration, and the hardest test — cross-dataset EEG1→EEG2 transfer, a **measured null**
that montage-alignment doesn't rescue → **[neuroscan/tasks/visual/](neuroscan/tasks/visual/)**.

**Lifting the defensible number — a pretrained encoder.** That cross-subject single-trial floor is **not a pure
signal-to-noise ceiling**. A pretrained EEG foundation transformer ([CBraMod](https://github.com/wjq-learning/CBraMod),
4.9M params, TUEG-pretrained), fine-tuned behind the same CLIP-retrieval head, **beats the from-scratch NICE
encoder cross-subject single-trial (~1.35×, replicated on 2 held-out subjects)** — the first method here to move
that number after three from-scratch transfer levers (input-alignment, domain-adversarial, concept-aware
InfoNCE) all came back null. A *frozen* probe of the same backbone fails (near chance), so it's the pretrained
capacity **adapted** to perception — not a linear readout — that lifts it; both fine-tune runs were still
improving when stopped (a lower bound). Modest in absolute terms, but it re-frames the wall as **capacity-bound,
not a hard SNR floor** — a swap-in encoder behind one contract → [neuroscan/models/foundation.py](neuroscan/models/foundation.py).

*Grounding — data [THINGS-EEG2](https://osf.io/3jk45/) ·
[THINGS-EEG1 (ds003825)](https://openneuro.org/datasets/ds003825) · method/SOTA NICE (Song et al., ICLR 2024) —
the claimed numbers are the within-subject / concept-averaged cell (14.8 %); our measured cross-subject
single-trial is 1.9 % · homework:
[EEG→image datasets + the leakage-trap landscape](research/deep_dives/2026-07-07_eeg_visual_semantic_decoding_datasets.md).*

## Limits (measured, not assumed)
Competent on a public benchmark, **not** a finished system:
- **Reproduction is partial.** Best within-subject ~0.62 vs published 0.81; clean subjects reproduce
  (A03 ~0.79 vs published peak ~0.85), hard subjects lag ~0.15 — documented in [`research/`](research/),
  not hidden. The contribution is the measured OOD gap + calibration + efficiency, not the peak.
- **Fusion is unsolved here, not disproven.** Complementarity is real and large (oracle **+17 pts**; fNIRS
  uniquely rescues ~17% of blocks), but the fusion methods tried so far — late, feature, output-space combiners,
  an input-level gate — don't beat the best single modality. That's a measured *failure to capture* real
  headroom, not proof it can't be: **boundary-aware routing** (fNIRS on the 0-vs-load boundary it *can*
  separate) and **source-space fusion** are untested. The graded 2-vs-3 sub-contrast *is* physiologically
  capped; the fusion of the rest is open.
- **Neuroscience is a ramp.** The signal-processing / eval discipline carries from prior work; the
  decoding methods and neuroscience are learned as I go.
- **Not a device.** Public research data only; no real-time online BCI, no clinical or prospective validation.

## Data
**BCI Competition IV-2a** (motor imagery) — 9 subjects, 4-class (left/right hand, feet, tongue), 22 EEG
channels @ 250 Hz, 2 sessions × 288 trials — pulled via **[MOABB](https://moabb.neurotechx.com/)**.

**Shin 2017 hybrid EEG+fNIRS** (mental workload) — 26 subjects, **simultaneous** EEG + fNIRS (36 optode
channels) on an n-back working-memory task (0/2/3-back) — the EEG · fNIRS · fusion half. Not on MOABB; pulled
direct from **[TU Berlin DepositOnce](https://doi.org/10.14279/depositonce-5830.2)** (`EEG_01-26_MATLAB.zip`
\+ per-subject `VP<NNN>-NIRS.zip`, GPL-3.0), parsed from the `.mat` by per-modality adapters.

Both kept **outside the repo** (size + licensing). Per-dataset adapters remap each to a canonical schema and
cache epochs to a recipe-keyed store; splits are queries over that cloud. One data root, set once:
```bash
cp paths.example.yaml paths.yaml      # then: data: <abs path to a data dir outside the repo>
```
Downloads land under `<root>/raw/`; the epoch cache under `<root>/processed/` (created on first run).

## Quickstart
```bash
uv sync                                              # .venv from pyproject + uv.lock; prefix commands with `uv run`
cp paths.example.yaml paths.yaml                     # set the one data root
# runs are named configs in experiments.yaml, picked with --exp (argv stays sparse; --set for ad-hoc tweaks):
# the headline contrast — the same decoder, two regimes:
uv run python -m neuroscan.tasks.run --exp mi_csp_within
uv run python -m neuroscan.tasks.run --exp mi_csp_cross               # the OOD gap
# the strongest classical baseline — covariances on a Riemannian manifold:
uv run python -m neuroscan.tasks.run --exp mi_riemann_within
# the second modality — fNIRS mental-workload (amplitude features, not covariance):
uv run python -m neuroscan.tasks.workload.run_fnirs --exp nback_fnirs_cross
# EEG+fNIRS fusion on the same task — complementarity + the aggregation sweep (a rigorous null):
uv run python -m neuroscan.tasks.workload.run_fusion --exp nback_fusion
# a deep decoder, GPU (ad-hoc override of a base config — broadband recipe the nets prefer):
uv run python -m neuroscan.tasks.run --exp mi_csp_within \
  --set method=atcnet --set recipe.resample=250 --set recipe.fmin=4 --set recipe.fmax=40
# the neuroviz demo (EEG / fNIRS / Fusion brain-camera view):
uv run python -m neuroviz.export --subject 1 && uv run python -m neuroviz.fusion.export --subject 1 \
  && python -m http.server 8000 -d neuroviz/web
uv run pytest -q
```
Runs log to a local MLflow (`uv run mlflow ui --backend-store-uri sqlite:///mlflow.db`) and write
`runs/<name>/` with an aggregate, a model card, and the run id.

## Tests
```bash
uv run pytest -q          # unit (equivalence-class) + integration (module chains)
```
A pyramid: a wide unit base testing each module by equivalence class (metrics, the split-as-criteria
logic, transforms, calibration, the profiler) — `tests/unit/` **mirrors the source tree**, so a module's
tests live where the module does. The integration layer covers the chains units can't (data cloud → splits
→ harness end-to-end; decoder → ONNX export → parity) and stays scenario-based.

## How it's built
Agent-driven build, human-owned judgment — coding agents scaffold the plumbing; the modeling decisions,
the measurement correctness, and the evaluation are mine. The architecture (two-layer engine + science,
split-as-criteria, dataset-adapter registry, calibration-under-shift) is carried from a mature prior ML
project of mine; see [`docs/STRUCTURE.md`](docs/STRUCTURE.md). The neuroscience and decoding specifics I
learn as I go.

## References
- **BCI Competition IV-2a** — Tangermann et al., *Review of the BCI Competition IV*, Front. Neurosci. 2012.
- **Shin 2017 (hybrid EEG+fNIRS n-back)** — Shin et al., *Open Access Dataset for EEG+NIRS Single-Trial Classification*, IEEE TNSRE 2017 (data: TU Berlin DepositOnce, DOI 10.14279/depositonce-5830.2). CBSI: Cui et al., *A quantitative comparison of NIRS and fMRI*, NeuroImage 2011.
- **CSP / FBCSP** — Ang et al., *Filter Bank Common Spatial Pattern (FBCSP) in BCI*, IJCNN 2008 / 2012.
- **EEGNet** — Lawhern et al., *EEGNet: a compact CNN for EEG-based BCIs*, J. Neural Eng. 2018.
- **ShallowConvNet / Deep4Net** — Schirrmeister et al., *Deep learning with CNNs for EEG decoding*, HBM 2017.
- **ATCNet** — Altaheri et al., *Physics-informed attention temporal CNN for EEG-based MI classification*, IEEE TII 2023.
- **MOABB** — Jayaram & Barachant, *MOABB: trustworthy algorithm benchmarking for BCIs*, J. Neural Eng. 2018.
- **Braindecode** — the PyTorch EEG-decoding library the deep models are built on.
