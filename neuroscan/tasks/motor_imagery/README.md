# Motor imagery — BCI IV-2a (EEG, the control rung)

The **control** rung of the [field-map](../../../README.md): decode which movement a subject *imagines*
(left/right hand, feet, tongue) from EEG. This is where the project's through-line — the **cross-subject
generalization gap**, and closing it — is measured most fully. The root README carries the headline gap and
the RPA ladder table; this page is the mechanism behind them, the decoder comparison, and how our numbers sit
against published SOTA.

```bash
uv run python -m neuroscan.tasks.run --exp mi_csp_within           # within-subject
uv run python -m neuroscan.tasks.run --exp mi_csp_cross            # cross-subject (LOSO) — the gap
uv run python -m neuroscan.tasks.run --exp mi_riemann_within       # the strong classical baseline
```

## Closing the gap — the RPA ladder, mechanism by regime

The collapse is a **domain shift**: each subject's covariance cloud sits at a different location on the SPD
manifold, so a classifier trained on others misses them — not because the ERD contrast differs, but because
the cloud is *displaced*. **Riemannian Procrustes Analysis** (Rodrigues 2019) aligns the domains in three
steps; we report where each sits on the **deployability axis** — how many *target* labels it needs
([`align.py`](align.py)):

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

Two regimes, read them separately. **Zero-shot** (no target labels — deployment-real): re-centering to the
identity by each subject's own Riemannian mean (`C → M⁻¹ᐟ² C M⁻¹ᐟ²`, the manifold version of whitening) closes
most of the gap, **0.36 → 0.50**; adding dispersion-alignment (re-scaling) nudges it to **0.52**. The
displacement *was* the gap — and it's the *location*, not the features (ACM's richer time-delay covariances
score 0.355 alone, only 0.471 even re-centered). **Calibrated** (a short labelled calibration session): the
supervised re-rotation aligns *class* structure and lifts further — even **10 %** of a session (≈7 trials/class)
reaches **0.555**, scaling to **0.650** at 50 %, approaching the within-subject ceiling (0.60–0.66).

**MDWM is the negative we report.** Untuned it scores 0.412, below zero-shot re-centering. Its λ knob *can*
lift it — but acc swings **0.31 → 0.57** across λ and the optimum is **λ = 1 (target-only)**, i.e. the best
MDWM ignores the source entirely. A parameterless method (re-centering: no knob, no labels) is preferable when
it's competitive, so we report MDWM untuned — tuning it up would hide the fragility worth showing.

**The one non-negotiable:** calibrated labels come from a **disjoint** stratified split of the held-out
subject — fit there, scored on the *remaining* blocks. Test labels never enter the fit, or "calibrated
transfer" is just leakage. (Same discipline as the fNIRS calibration ablation.)

**Calibration under shift.** Temperature scaling fit on an in-session validation split, ECE measured
before/after on the *cross-session* test (ATCNet): test ECE **0.113 → 0.084**. We report the *transfer* —
whether an in-session calibration fix survives the session shift — not a single in-distribution ECE
([`calibrate.py`](../../../neuroscan/evaluation/calibrate.py)).

## The decoders — commodity architectures, measured

We reproduce *standard* architectures (the decoder is commodity); the contribution is the eval rigor and the
efficient deployable, not a leaderboard number. Params + FLOPs at the real input (22 ch × 1125 samples, batch
1; FLOPs via fvcore, latency torch CPU single-thread — `python -m neuroscan.models.profile`):

| model | role | params | FLOPs | CPU latency | within-subj acc | kappa |
|---|---|---|---|---|---|---|
| CSP+LDA | baseline | — | — | — | <!--r:csp_lda_within_bnci2014_001.acc-->0.598<!--/r--> | <!--r:csp_lda_within_bnci2014_001.kappa-->0.464<!--/r--> |
| **Riemann (tangent space + LR)** | baseline | — | — | — | **<!--r:riemann_within_bnci2014_001.acc-->0.655<!--/r-->** | **<!--r:riemann_within_bnci2014_001.kappa-->0.541<!--/r-->** |
| **EEGNet** | compact CNN | **3.7K** | 13.7M | 1.5 ms | 0.606 | 0.475 |
| **ATCNet** | attention + TCN | 114K | **2.8M** | 4.2 ms | 0.619 | 0.492 |
| EEGConformer | transformer | 871K | 72M | 4.2 ms | — | — |

Three findings fall out:
- **Classical geometry leads within-subject — strong-and-cheap, not a settled verdict.** Riemannian
  tangent-space + LR ([`baselines/eeg/riemann.py`](../../../baselines/eeg/riemann.py)) hits **0.655**, above both deep
  nets *as run here* (single seed, nets un-tuned — not a fair head-to-head). Consistent with the textbook
  finding that per-trial covariance is hard to beat when per-subject data is tiny (~288 trials). But its
  *cross-subject* score is 0.360, no better than CSP — plain tangent space doesn't transfer until you
  **re-center** it (above).
- **Tiny doesn't cost accuracy here.** The 3.7K-param EEGNet lands ~1 pt behind the 30×-larger ATCNet (0.606
  vs 0.619) — comparable, not distinguishable, at single seed: the edge-deployable model gives up little.
- **Already edge-sized.** ~26 KB as ONNX, sub-ms inference; the optional deploy tail exports with a **parity
  gate** (fp32 ONNX matches torch < 1e-3) and benchmarks INT8 — which *adds* overhead at this scale. The story
  isn't "shrink it," it's "already small, measured." ([`core/export_onnx.py`](../../../core/export_onnx.py))

## Why our numbers sit below published SOTA — deliberately

**All our numbers sit below the published ceilings, on purpose.** Our robust train→eval-session protocol is
harder than the pooled within-session CV many papers report, and we don't do full per-model tuning or
run-averaging. Snapshot vs the published bar (4-class, same session-1→2 hold-out):

- **ours:** CSP+LDA within 0.598, cross-subject (LOSO) 0.391; ATCNet within 0.619.
- **published:** FBCSP 0.65 · EEGNet ~0.71 · ShallowConvNet 0.74 · **ATCNet 0.81** (10-run avg, 500 epochs,
  no early stop) · transformer SOTA 0.88; cross-subject SOTA 0.74.

The ~23-pt gap to ATCNet is **pipeline + model-usage, not honesty**: ATCNet's 0.81 uses the *same* T→E
hold-out we run — the gap is exponential-moving standardization, a full 500-epoch schedule (we early-stop
~ep 90), ATCNet's own internal window-aug (our external crop fights it), and 10-run averaging (we run a single
seed, dragged by 2-3 collapse subjects). None is data or eval-honesty. The primary-source recipe diff (what
SOTA does that we don't, confirmed against repos) →
[`research/2a_sota_recipe`](../../../research/deep_dives/2026-06-30_2a_sota_recipe.md). The contribution is the
measured within→cross gap + calibration on a credible-but-untuned decoder, not the peak.

## How motor imagery decodes — the ERD signature

The decodable signal is **event-related desynchronization (ERD)**: imagining a movement *suppresses* mu
(8–12 Hz) and beta (13–30 Hz) rhythms over the **contralateral** sensorimotor cortex — left-hand imagery
desynchronizes the right hemisphere (C4), right-hand the left (C3). CSP learns spatial filters that maximize
this variance contrast (its patterns localize over C3/C4, visible in [neuroviz](../../../neuroviz/)); deep
nets learn it end-to-end. The signature is **subject-specific** — the spatial pattern, the responsive band,
and the SNR all vary per person — which is precisely why cross-subject transfer collapses (and why
re-centering, which removes the per-subject displacement, is the fix).
