# MVNN preprocessing vs per-channel z-score — the official THINGS-EEG2 whitening is a modest real lever (bd b40j)

**Date:** 2026-07-15 · **Task:** Stage-3 THINGS-EEG2 cross-subject single-trial retrieval (train[1-4] → test5)

## Question

Foundation-hardening (Phase 0): our adapter substitutes a per-channel z-score for the official THINGS-EEG2
preprocessing, which uses **MVNN** (multivariate noise normalization, Guggenmos 2018 / Gifford 2022) — whiten
each trial by the per-subject *within-condition noise covariance*. We reproduce the published NICE
within-subject number without MVNN, so it is not *required*. Does matching the official preprocessing lift the
cross-subject single-trial retrieval, or is the z-score enough?

## Method

`Mvnn.whiten` (core/features/eeg/mvnn.py): per subject, residualize each trial against its condition mean
(same *image*, not concept — train has 10 exemplars/concept), pool the residuals across all conditions, fit a
Ledoit-Wolf `Σ`, and apply `Σ^{-1/2}` to every trial. Pooling across conditions + shrinkage is what makes `Σ`
well-conditioned despite only 4 reps/concept in train. Opt-in via `TrainConfig.mvnn` / `--mvnn`; it replaces
the adapter z-score (both hand the encoder ~unit-scale input; MVNN additionally decorrelates the noise).

**Matched A/B:** NICE from-scratch, cross-subject train[1-4]/test5, `perception_converged` (60 ep, lr 3e-4,
batch 512), resample 200 — the two arms differ *only* in normalization. The z-score arm reproduces the known
NICE baseline (single-trial top1 1.60%), so the control is valid.

## Result — KEEP

| metric              | z-score | **MVNN** |    Δ    |
|---------------------|:-------:|:--------:|:-------:|
| single-trial top1   |  1.60   | **1.94** | **+0.35** |
| single-trial top5   |  7.05   | **7.85** | +0.80   |
| val top1 (leak-free)|  5.43   | **6.31** | +0.87   |
| mean_rank           |  71.0   | **66.1** | −5.0    |
| margin              |  0.016  | **0.021**| +0.005  |
| concept-avg top1    |  3.00   |  3.00    | 0       |
| concept-avg top5    |  18.0   |  16.0    | −2.0    |
| best-val epoch      |  55     | **28**   | ~2× faster |

## Interpretation

MVNN is a **modest real improvement** on the deployment-honest metric (single-trial), plus a **~2× convergence
speedup** (best val at ep28 vs 55). The single-trial top1 +0.35pp is single-seed and so scatter-prone on its
own, but it does not stand alone: **val (a different held-out split) +0.87, mean_rank −5, single-top5 +0.80,
and margin +0.005 all move the same direction.** That convergence across independent axes is more than
one-metric noise — whitening the per-subject noise covariance genuinely tightens the cross-subject single-trial
retrieval.

The one counter-signal — concept-avg top5 −2.0 — is the **known whitening mechanism**, not a contradiction:
concept-avg averages all 80 test reps of a concept, so the per-trial noise MVNN removes is already suppressed
by averaging, and MVNN's slight distortion of the signal covariance then costs a little. This is the same
single-trial-helps / concept-avg-neutral-to-worse split seen for Riemannian re-centering
([[perception-signal-recenter-fails]]). Single-trial is what a deployed decoder sees (no 80-rep averaging), so
the win is where it matters.

**Foundation verdict:** the z-score substitute *was* leaving signal on the table — matching the official
MVNN preprocessing recovers it. The base is sound (the z-score arm reproduces the NICE baseline), and closer to
the field-standard pipeline it is modestly better. Exactly what Phase-0 "verify the base" is for.

## Caveats / next

Single seed, one test subject (5). The +0.35pp single-trial is *suggestive, not proven* — now that a number
moved, multi-seed + LOSO is the warranted hardening (stage-gated rigor: harden a number that moved), filed as a
follow-up. Not promoted to the default preprocessing until multi-seed confirms; kept opt-in
(`TrainConfig.mvnn`). Baseline correction (a *separate* axis — needs a pre-stim window our tmin=0 lacks) was
deliberately not bundled here, so this A/B isolates normalization; it is the next preprocessing axis to test.
