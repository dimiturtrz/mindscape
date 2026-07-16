# MVNN done right: per-subject calibration is leak-free but ≈ neutral (the b40j win was the leak)

**Date:** 2026-07-16 · **Task:** Stage-3 perception (EEG→image), cross-subject single-trial retrieval ·
**Bead:** per-subject MVNN calibration · **Supersedes:** the b40j "MVNN beats z-score +0.35" finding and the
premature u9sv "retraction".

## The question

bd b40j found MVNN (official THINGS-EEG2 / Guggenmos noise whitening) beat per-channel z-score by **+0.35pp**
single-trial top1 and promoted it to the perception default. The leak rule ("normalization fits only on train,
never the scored test trials") then exposed that b40j fit the whitener on the **scored test trials using their
image labels** — and image identity is exactly the retrieval target. That is a transductive leak.

The first fix was wrong too: one **global** whitener pooled over the train subjects, applied to the held-out
test subject. That *cripples* MVNN — a subject's sensor-noise geometry is not the train-pool average — and it
scored **1.44 < z-score 1.60**. Reading that as "MVNN loses" was a mis-measurement: MVNN is intrinsically
**per subject** (each person whitens by their own noise covariance; that is how THINGS-EEG2 applies it).

So the real experiment was unrun: **per-subject** MVNN, fit **leak-free**.

## The method (leak-free + deployment-real)

Each subject gets their **own** `Σ^{-1/2}`, fit on that subject's **calibration** epochs — for the held-out
test subject, its **training-image** trials, which are **disjoint from the 200 scored test images**. The
whitener never sees the scored trials, so the retrieval target does not leak. Apply = select each row's
subject whitener and multiply: a fixed `[63×63]` matrix per subject, one matmul per trial. This is exactly
**unbatched real-world deployment** — enroll a subject once (a calibration session), then whiten every
incoming single trial by the stored matrix. (`core/normalization/mvnn.py`, per-subject `_whiteners` dict;
`Normalizer.apply(X, groups)` gained an optional per-row subject id; the trainer loads the test subject's
training trials as calibration, `neuroscan/tasks/visual/train_nice.py`.)

## Result (single-seed, matched `perception_converged` 60ep, train[1-4]/test5)

| arm | single-top1 | single-top5 | mean_rank | margin | concept-top5 | best-val |
|-----|:-----------:|:-----------:|:---------:|:------:|:------------:|:--------:|
| z-score (baseline) | **1.60** | 7.05 | 71 | 0.016 | 18 | 5.43 |
| leaky MVNN (b40j) | 1.94 | 7.85 | 66 | 0.021 | 16 | 6.31 |
| global-fit (crippled) | 1.44 | — | — | — | — | — |
| **clean per-subject** | **1.66** | 7.46 | 68 | 0.019 | 14.5 | 6.31 |

Clean per-subject sits **between** z-score and leaky on every ranking axis, far closer to z-score.

## Reading

1. **The b40j +0.35 was ~85% leak.** Leak-free per-subject recovers only **+0.06** single-top1 — within the
   baseline single-seed scatter (±0.14, `perception-cross-subject-hard-floor`). On the headline metric,
   clean MVNN ≈ z-score.
2. **Per-subject is the right implementation; global-fit was wrong.** 1.44 (global) → 1.66 (per-subject)
   confirms the crippling was the pooled whitener, not MVNN. Do not cite 1.44 as "MVNN loses".
3. **A faint, consistent ranking nudge** (top5 +0.41, mean_rank −3, margin +0.003) — the legitimate slice of
   the per-subject-whitening benefit the leak inflated — but **not** on single-top1.
4. **Real concept-avg cost** (14.5 < 18): the known whitening mechanism — concept-avg already averages ~80
   reps so per-trial denoising vanishes, while the signal-covariance distortion still costs.

## Verdict

Used right and leak-free, MVNN is **neutral-to-marginal vs z-score**, not the +0.35 the leak faked. Not
enough to flip the default (+0.06 single-seed would be reading noise; noise discipline forbids it). **z-score
stays the NICE default**; MVNN stays a **physically-correct, leak-free, documented override** with a mild
ranking edge and a concept-avg cost. No multi-seed — stage-gated rigor says don't harden a neutral number.
The lever for the cross-subject single-trial wall is elsewhere (encoder capacity / adaptation, not input
normalization) — consistent with `perception-input-align-not-bottleneck`.
