# 02 · Signal processing & CSP — from voltage to features

Grounded in [`research/deep_dives/2026-06-30_motor-imagery-neuroscience.md`](../research/deep_dives/2026-06-30_motor-imagery-neuroscience.md)
(cited sources). How raw scalp voltage becomes something a classifier can use. Connects to `core/data/eeg/` (preprocessing),
`baselines/csp_lda.py` (CSP), and `neuroviz` (spectra/topomaps). The half of the work that's pure DSP +
linear algebra.

Lesson plan:
1. Where the voltage comes from (EEG genesis)
2. Frequency bands & spectral power — what "mu power" means
3. Time-frequency — seeing ERD as a power-over-time drop
4. The spatial problem — why one channel isn't enough
5. CSP — the spatial filter that makes motor imagery separable

---

## Lesson 1 — EEG genesis (where the voltage comes from)

An electrode doesn't see single neurons. It sees the **summed extracellular field of millions of cortical
pyramidal neurons** firing roughly together. Pyramidal cells are aligned perpendicular to the cortex; when
many receive synchronous synaptic input, their tiny dipoles add up to a field big enough to measure at the
scalp (~microvolts).

Two consequences that shape everything downstream:
- **Volume conduction** — the field spreads through brain/skull/scalp, so each electrode picks up a *blurred
  mixture* of many sources. A signal from C3's cortex leaks into neighboring electrodes. (This is *why* we
  need spatial filters — Lesson 5.)
- **Low SNR + artifacts** — the motor signal is a small fraction of what's recorded; eye blinks (EOG),
  muscle (EMG), and 50/60 Hz line noise are often *larger* than the brain signal.

**Takeaway.** EEG = a blurred, noisy, mixed surface measurement of synchronized cortical populations. The
information is there but smeared across channels and buried in noise.

## Lesson 2 — frequency bands & spectral power

Because the signal is **oscillatory**, the useful information lives in the **frequency domain**. A Fourier
transform decomposes a channel's time-series into how much power sits at each frequency. Group into bands:
delta (0.5–4), theta (4–8), **mu/alpha (8–12)**, **beta (13–30)**, gamma (>30).

"**Mu power**" = how much 8–12 Hz oscillation is present in a window. **ERD = mu/beta power goes down.** So
the feature we ultimately want is *band power per channel* (and how it changes).

*In our pipeline:* the preprocessing bandpass-filters to the motor bands (e.g. 4–40 Hz) before decoding —
throwing away delta/gamma that carry no motor-imagery signal and only add noise. `EpochCfg(fmin, fmax)`.

**Takeaway.** Move to frequency; the motor signal is *power in mu/beta*, not raw voltage.

## Lesson 3 — time-frequency (seeing ERD)

A single power number per trial hides *when* the ERD happens. **Time-frequency analysis** (short windows, or
wavelets/Hilbert) gives power as a function of *both* time and frequency → you can watch mu power drop after
the cue and rebound (ERS) after. That's the canonical motor-imagery plot.

*In our pipeline:* `neuroviz` computes band power in sliding windows and animates it — the topomap *is* a
time-frequency view projected onto the scalp.

**Takeaway.** ERD is a *time-localized* power drop; time-frequency is how you see it.

## Lesson 4 — the spatial problem

Volume conduction means **no single electrode cleanly captures one cortical source** — each is a mixture.
But the classes differ *spatially* (left-hand ERD over C4, right-hand over C3). So the right features are
**spatial combinations** of channels that isolate a source — a *spatial filter*: a weighted sum of the 22
channels that amplifies one cortical region and cancels the rest.

This is a **linear algebra** problem: find weight vectors `w` (one per virtual "source") such that `w·X`
recovers the discriminative signal. CSP is the classic solution.

**Takeaway.** The discriminative information is spatial; we need *combinations* of channels, not channels.

## Lesson 5 — CSP (Common Spatial Patterns)

**Goal:** find spatial filters that make the band power **maximally different between two classes** — large
variance for class A, small for class B (and vice-versa for other filters). Since (for band-limited signals)
**variance = band power**, maximizing the variance *ratio* maximizes the ERD contrast.

**The math (the linear-algebra core):**
1. Band-pass the signal (so variance ≈ band power).
2. Compute each class's **covariance matrix** across channels (C₁, C₂).
3. Solve the **generalized eigenvalue problem** `C₁ w = λ (C₁+C₂) w`. The eigenvectors `w` are the spatial
   filters; eigenvalues λ near 1 → that filter has high variance for class 1, low for class 2 (and the
   λ-near-0 filters do the opposite).
4. Keep the few most extreme filters (top + bottom). Project: `w·X` → log-variance → features.
5. A simple classifier (LDA) on those few features separates the classes.

The filters are interpretable: plotted as topomaps (CSP **patterns**), they localize over C3/C4 — you can
*see* the decoder using motor cortex. (neuroviz renders these.)

*In our pipeline:* `baselines/csp_lda.py` = `CSP(n_components=6)` → `LDA`. This handcrafted, ~no-parameter
pipeline gets **0.598 within-subject** — the baseline the deep nets must beat. It encodes the neuroscience
(band power + contralateral spatial pattern) by hand; the deep nets learn it.

**Takeaway.** CSP = a learned spatial filter (via generalized eigendecomposition) that turns 22 mixed
channels into a few features whose variance *is* the class-discriminating ERD. It's the bridge from "the
signal is spatial band power" to "a vector a classifier can use."

## Delta — what EEG adds vs general time-series ML
For someone fluent in time-series ML, the genuinely-different bits:
1. **Frequency-domain features dominate** — band power, not raw samples; the physics is oscillatory.
2. **Spatial mixing (volume conduction)** forces spatial filters (CSP/ICA) — not a thing in most 1-D series.
3. **Tiny, per-subject data** + huge between-subject variance → calibration and cross-subject eval matter
   more than model size.
4. **Artifacts** (EOG/EMG/line) are often bigger than signal → preprocessing is half the battle.

---

## Quiz — 02
1. Why does an EEG electrode pick up a *mixture* of sources, and what's that phenomenon called?
2. What does "mu power" mean, and what happens to it during motor imagery?
3. Why do we band-pass filter before CSP? (Hint: variance vs power.)
4. In one sentence, what does CSP optimize, and via what linear-algebra operation?
5. CSP+LDA has almost no trainable parameters yet hits 0.598 — what prior knowledge is "baked in" that lets
   it work without learning from scratch?

<!-- quiz log appended below on demand -->
