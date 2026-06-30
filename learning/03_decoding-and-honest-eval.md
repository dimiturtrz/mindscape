# 03 · Decoding & honest evaluation — the contribution

Grounded in [`research/deep_dives/2026-06-30_2a_sota_recipe.md`](../research/deep_dives/2026-06-30_2a_sota_recipe.md).
From features to a prediction, and — the point of mindscape — how to judge it honestly. Connects to
`neuroscan/models/`, `neuroscan/evaluation/`, and our measured numbers.

Lesson plan:
1. The decoder zoo — CSP+LDA, EEGNet, ATCNet (what differs)
2. The eval regimes — within vs cross-subject vs cross-session
3. The generalization gap — our headline number, and what it means
4. Calibration — confidence vs correctness, and under shift
5. The efficiency angle — tiny nets, edge deploy

---

## Lesson 1 — the decoder zoo

All decoders solve the same problem (features → class) but differ in how much they learn:
- **CSP+LDA** — handcrafted: CSP spatial filters (band-power features) → Linear Discriminant Analysis.
  ~No trainable parameters beyond the eigendecomposition + a linear boundary. Encodes the neuroscience by
  hand. **Our baseline: 0.598 within-subject.**
- **EEGNet** — a *compact* CNN (~3.7K params): a temporal convolution (learns frequency filters) + a
  depthwise spatial convolution (learns CSP-like spatial filters) + a separable conv. Learns end-to-end what
  CSP does by hand. **0.606 ours.**
- **ATCNet** — attention + temporal convolutional network (~114K params): EEGNet-style conv front end +
  multi-head self-attention + a TCN + an internal sliding-window augmentation. Heavier, more expressive.
  **0.619 ours.**

The honest point: **the architectures are commodity** (off-the-shelf from braindecode). The 3.7K EEGNet
ties the 30×-bigger ATCNet on our protocol — model size isn't the bottleneck here; the *signal* is.

*In our pipeline:* `neuroscan/models/__init__.py:get_method(name)` returns any of them as one
`(fit_fn, score_fn)` pair; the harness treats them identically.

**Takeaway.** From hand-coded (CSP) to learned (EEGNet/ATCNet) — same job, and they land within a few points
of each other. The decoder is not where the contribution is.

## Lesson 2 — the eval regimes (the actual contribution)

The same decoder gives wildly different numbers depending on *what you hold out*. mindscape makes the
**regime** a first-class, explicit choice (a criteria filter over the data):
- **within-subject** — train + test the *same* person (the standard train-session → eval-session split).
  The *ceiling* — flatters the decoder.
- **cross-subject (leave-one-subject-out)** — train on 8 people, test on the 9th. The *honest* number for
  "does it work on someone new?"
- **cross-session** — same person, different day. Tests *drift*.

*In our pipeline:* `core/data/splits.py::make_split(meta, test_subjects=…, test_sessions=…)` — a split is the
cloud *filtered on criteria*, so a run self-documents what it held out. `harness.folds_for(regime)` builds
the folds.

**Takeaway.** "Accuracy" is meaningless without the regime. Naming and reporting the regime *is* the honesty.

## Lesson 3 — the generalization gap (our headline)

| regime | CSP+LDA acc | kappa |
|---|---|---|
| within-subject | 0.598 | 0.463 |
| cross-subject (LOSO) | 0.382 | 0.176 |
| **gap** | **−0.216** | |

A decoder at ~60% on its own data drops to **38%** on an unseen person (chance 25%). Per subject, the
cross-subject accuracy spans 0.24–0.54 — three subjects are **at/below chance** on someone new. The *neural*
reason (lesson 01.5 + the research): the ERD signature is **idiosyncratic** (peak frequency, scalp locus,
SNR all vary per person), so a decoder tuned to A is mistuned for B. **This is biology, not a bug** — and
it's the trap most papers hide by reporting only within-subject (or pooled-session CV).

**Kappa vs accuracy:** Cohen's κ corrects for chance. κ=0 means "no better than guessing"; our cross-subject
κ=0.176 says *barely* above chance. Report κ because raw accuracy looks deceptively okay at 4 classes.

**Takeaway.** The gap is the contribution — measured, stratified per subject, and explained.

## Lesson 4 — calibration (confidence vs correctness)

A model can be *accurate* but *overconfident*: it says "90% sure" and is right only 70% of the time. **ECE
(Expected Calibration Error)** measures that gap between confidence and accuracy. It matters for a BCI
because a comms/control decoder needs to *know when it's unsure*.

**Under shift** is the hard part: **temperature scaling** (divide logits by a learned scalar T, fit on a
held-out set) fixes calibration *in-distribution*, but does it survive a session/subject shift? We measure
the *transfer*: ATCNet test ECE **0.113 → 0.084** after temperature scaling fit on an in-session val set.
We report whether the fix carries across the shift, not a single in-distribution ECE.

*In our pipeline:* `neuroscan/evaluation/calibrate.py` (temperature scaling), `metrics.ece`.

**Takeaway.** Calibration = does the confidence mean anything. Honest BCIs report it *under shift*, not just
in-distribution.

## Lesson 5 — the efficiency angle

Measured (`python -m neuroscan.models.profile`): EEGNet **3.7K params / 13.7M FLOPs / 1.5 ms CPU**; ATCNet
114K / 2.8M FLOPs / 4.2 ms. The 3.7K EEGNet *ties* ATCNet on accuracy → the **edge-deployable model gives up
nothing**. The decoders export to ONNX (~26 KB) with a **parity gate** (fp32 ONNX must match torch < 1e-3),
and at this size INT8 quantization *adds* overhead rather than saving — the honest finding is "already
edge-sized."

*In our pipeline:* `core/export_onnx.py`, `neuroscan/experiments/quantize.py`.

**Takeaway.** Deployability is a measured result here, and a tiny model is competitive — exactly the angle a
bandwidth/efficiency-minded BCI cares about.

## Delta — what neural decoding adds vs general classification
1. **The regime is the result.** Within vs cross-subject can swing 20 points — naming it honestly is the
   whole game.
2. **Calibration under shift** matters more than peak accuracy for a usable BCI.
3. **Tiny data + per-subject variance** → small models compete; generalization, not capacity, is the wall.
4. **Don't chase SOTA** — the decoder is commodity; the eval rigor + the deployable are the contribution.

---

## Quiz — 03
1. Why can the *same* decoder report 0.60 and 0.38 on the same dataset? What's the honest number for "works
   on a new person"?
2. Our cross-subject κ is 0.176 — what does that say, and why report κ instead of accuracy?
3. What does ECE measure, and why does a BCI care about it specifically?
4. Temperature scaling fixes calibration in-distribution — what's the *honest* thing we measure about it?
5. EEGNet (3.7K params) ≈ ATCNet (114K) on accuracy. What does that tell you about where the bottleneck is —
   model capacity or signal?

<!-- quiz log appended below on demand -->
