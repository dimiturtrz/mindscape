# Retrieval trainer: speed parity + rank metrics + negative-quality ablation (7tl)

**Date:** 2026-07-18 · **Task:** Stage-3 perception (EEG→image), cross-subject retrieval · **Bead:** 7tl
**Frame:** 7tl is an *efficiency + instrumentation* epic, **not a top-k lever**. The cross-subject single-trial
wall is capacity/SNR — only pretrained-capacity *adaptation* ever moved it (frozen 0.63 → NICE 1.6 → LoRA 2.1 →
ft 2.38); three transfer levers (dpi/36g/lbd) and the longer schedule (o2n) are null/spent. The goal was never
to beat the floor. It was to make the trainer fast, honestly-instrumented, and to **settle the negative-quality
question by ablation** instead of leaving it as hope.

All arms: NICE encoder, train subjects 1-4 / test 5, resample 200, 40 epochs, lr 3e-4, bf16, single seed.
Chance = 0.5%. Single-seed noise band ≈ ±0.14 (established).

## Results

| arm | sampling | sec/epoch | single-top1 | single-top5 | MRR | median-rank | PR-AUC |
|-----|----------|-----------|-------------|-------------|-----|-------------|--------|
| uniform (baseline) | uniform | 13.8 | 1.82 | — | 0.066 | 55 | 0.010 |
| subsample (frac 0.2) | uniform | **2.8** | **1.82** | — | 0.067 | 52 | 0.011 |
| balanced (ewd) | balanced | 21.4 | 1.77 | — | 0.062 | 58 | 0.010 |
| clip_hard (mnr) | clip_hard | 21.6 | 1.71 | — | 0.060 | 60 | 0.009 |

(fp32 reference: ~43 s/epoch, from perf-tf32 microbench — not re-run; the bf16 arm is the matched control.)

## Deliverable 1 — speed, measured at parity: WIN

- bf16 autocast alone: 13.8 s/epoch vs ~43 s fp32 = **3.1×**.
- bf16 + per-epoch subsample (frac 0.2, 1/5 of the fit trials each epoch): **2.8 s/epoch = ~15× fp32** — and
  **single-top1 is identical to the full-data baseline (1.82 = 1.82)**, MRR/median-rank/PR-AUC all a hair
  *better* if anything (0.067/52/0.011 vs 0.066/55/0.010, noise).

The NICE encoder is over-fed: 240k trials/epoch is far more than a 0.69M-param net needs per step, so seeing
1/5 of them per epoch costs nothing measurable while running 5× cheaper. Parity is the guard and it holds —
subsample is a free speedup, not an accuracy trade.

## Deliverable 2 — rank-aware metrics wired into the primary eval

MRR / median-rank / PR-AUC / recall@k existed (`Retrieval.retrieval_metrics`) but only `cross_dataset_eval`
consumed them. The primary cross-subject `evaluate()` now routes its `[N,C]` cosine score matrix through the
same function, so every retrieval report — per-epoch val and final test — carries rank-aware numbers, not only
top-1/5. Top-1 hides whether a miss landed at rank 2 or rank 200; median-rank/MRR don't (here the true concept
sits at **median rank ~55 of 200** — the faint-tilt floor made legible). Permanent honest-reporting value,
independent of any speed/sampler outcome. Verified live, committed separately.

## Deliverable 3 — negative-quality ablation: clean KILL, as forecast

Isolated design: train_frac 1.0 + bf16 fixed; **only `sampling` varies**. balanced (ewd) = strict
concept-balanced batches (64 concepts × 8 trials); clip_hard (mnr) = a seed concept + its 63 CLIP-nearest
neighbours per batch, `hard_beta = 0` to isolate *batch construction* from loss-weighting.

Both **lose to random, within noise, monotonically across every metric**: uniform 1.82 → balanced 1.77
(−0.05) → clip_hard 1.71 (−0.11), and the same order on MRR (0.066→0.062→0.060), median-rank (55→58→60),
PR-AUC (0.010→0.010→0.009). No arm moves top-1 above the ±0.14 noise band, and both cost ~1.5× the wall-time
(21 s vs 14 s/epoch).

**Mechanism.** This is the `lbd` precedent (concept-aware soft-negative InfoNCE, 1.73 vs 1.87 — washed) on the
batch-construction axis. The cross-subject single-trial embedding points ~86–90° off its true concept: signal
is a faint *relative tilt*, not a separable margin. Harder or better-balanced negatives sharpen a decision
boundary the encoder can't populate — clip_hard, which forces the most confusable concepts into every batch,
is the *most* negative, consistent with adding gradient noise on a boundary the per-trial SNR can't exploit.
Balanced negatives don't help either: uniform batching over 1654 training concepts already samples diverse
negatives, so enforcing exact balance changes nothing the loss was missing.

## Verdict

- **Speed (subsample + bf16): keep on.** ~15× fp32 at parity — a real, free win. bf16 was already default;
  subsample frac 0.2 is safe to enable for iteration.
- **Rank metrics: keep, permanent.** Honest reporting regardless of method.
- **balanced / clip_hard samplers: kill as defaults, keep opt-in.** They wash (marginally negative) and cost
  wall-time. Not a retrieval lever — the wall is capacity/SNR, not negative selection, now confirmed on the
  batch-construction axis too. No multi-seed warranted (nothing moved above noise to harden).
- **Non-goal held:** did not claim, and did not get, a top-k gain. 7tl delivered a faster, honestly-
  instrumented trainer and turned the negative-quality guess into a documented null.
