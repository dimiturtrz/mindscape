# Geometry prior on the trainable encoder: a mild regularizer, not a floor-breaker

**Date:** 2026-07-16 · **Task:** Stage-3 perception (EEG→image), cross-subject single-trial retrieval ·
**Bead:** 1x0 (geometry-aware spatial prior, ladder step 1) · **Epic:** qoa (Stage-3 semantic decoding)

## The question

The frozen CBraMod head-search (bd nm5) found **geometry wins monotonically**: heads that preserve electrode
layout beat the unordered channel bag, and the best geometry head (topo_cnn 1.75%) beat from-scratch NICE
(1.6%) with **zero backbone training**. That says spatial structure is decodable signal the flat NICE feature
vector throws away.

But NICE *trains* its spatial conv — it could in principle re-learn channel geometry from data. So: does an
explicit montage-adjacency prior still help once the encoder can adapt, or does training already recover it?
This is the ~700-sample cross-subject regime, where "knowledge > data" predicts the prior should pay.

## The method

Graph-Laplacian spatial-smoothness penalty on the NICE **spatial conv** (the layer that mixes the 63
channels). Build a channel graph from the unit-disk electrode positions with Gaussian-RBF adjacency
`A_ij = exp(−‖p_i − p_j‖² / 2σ²)`; the combinatorial Laplacian `L = D − A` gives the smoothness quadratic

```
penalty = tr(Wᵀ L W) = ½ Σ_ij A_ij ‖w_i − w_j‖²     (W = spatial weights reshaped [C, F·F])
```

added to the InfoNCE loss with weight `geo_lambda`. It pushes neighbouring electrodes toward similar mixing
weights — the montage adjacency injected as a prior, not learned from scratch. `σ` is the neighbourhood width
in unit-disk radii (0.2 matches the nm5 topo RBF). Off by default; NICE-only (foundation backbones have no
spatial conv to smooth). Code: `EegMontage.channel_laplacian`, `NiceEncoder.geo_penalty`,
`TrainConfig.geo_lambda/geo_sigma`.

**λ is calibrated, not guessed:** the penalty magnitude at init (σ0.2) is ≈ 46 against an InfoNCE loss of ≈ 6,
so λ=0.005 makes the prior ~4% of the loss (gentle) and λ=0.02 ~15% (stronger) — a spread bracketing "barely
regularizes" to "visibly constrains".

## Arms (matched: train[1,2,3,4] → test 5, seed 0, perception_converged 60ep, σ0.2)

| λ (geo) | single top1 | single top5 | concept top1 | concept top5 | mean_rank | margin | best-val ep |
|---------|-------------|-------------|--------------|--------------|-----------|--------|-------------|
| 0 (baseline) | 1.53 | 7.18 | 4.5 | 17.0 | 70.2 | 0.0167 | 47 |
| 0.005 | 1.65 | 6.85 | 4.5 | 20.5 | 70.2 | 0.0166 | 45 |
| 0.02 | 1.67 | 7.77 | 5.0 | 18.0 | 68.3 | 0.0209 | **19** |

## Verdict — mild regularizer, headline within noise

Two things happen, and they must be read separately:

1. **The headline retrieval number does not move beyond noise.** single-top1 goes 1.53 → 1.65 → 1.67 —
   monotone in λ, but +0.14 at λ0.02 sits exactly on the single-seed scatter (±0.14, the baseline is
   1.87±0.14 at 25ep / 1.60 converged). By the **36g precedent** — a single-seed +0.32 that washed to
   +0.13±0.06 across 3 seeds — this is forecast to wash under multi-seed. So it is **not** a confirmed
   retrieval lift.

2. **The prior is nonetheless doing something real — it regularizes.** At λ0.02 the coherent secondary
   metrics all tilt the same way: **margin +25%** (0.0167 → 0.0209), **mean_rank −1.9** (70.2 → 68.3), and
   **best-val at ep19 vs 47** — i.e. with patience 12 it early-stopped near ep31, reaching a matched-or-better
   number in **~⅔ fewer epochs** and without overfitting. The smoothness constraint trades encoder
   expressiveness for faster, flatter convergence — exactly a regularizer's signature.

So the geometry prior is **physically sound and coherently direction-positive, but it is a regularizer, not a
floor-breaker**: it does not lift the single-trial retrieval number the way adapting pretrained capacity did
(ft 2.38 / LoRA 2.10 vs NICE 1.60). This matches bd 1x0's own physics caveat — a *trainable* conv on
volume-conduction-smeared EEG already re-learns channel adjacency from data, so imposing it is largely
redundant; the gain is regularization, not new information.

**Decision:** KILL as a headline perception lever (won't move the cross-subject single-trial floor); KEEP
opt-in + documented (`TrainConfig.geo_lambda/geo_sigma`, off by default) — physically valid, real
regularization value if training-efficiency / small-data overfitting becomes the constraint. Do **not** spend
multi-seed GPU confirming it on perception (forecast to wash, per 36g).

## What it means for the roadmap

- **Redirect bd 1x0 to its fNIRS-first case.** The bead's own argument: EEG is already volume-conduction-smeared
  (adjacency baked in, prior redundant — now empirically confirmed), whereas fNIRS channels are genuinely
  local/independent, so imposing the correct cortical adjacency is *real new information*, not redundant.
  Geometry-prior-on-trainable-EEG washing is the predicted outcome; the untested real case is fNIRS n-back.
- **The perception floor stays a capacity/SNR problem**, not a structure-prior problem — consistent with the
  three null transfer levers and the ft/LoRA wins. Next perception spend goes to the capacity lever
  (longer ft schedule, then ooi's richer ViT-L/14 target), per the u9sv forecast, not to harder spatial priors.
- The reusable `EegMontage.channel_laplacian` + `NiceEncoder.geo_penalty` substrate stands regardless — it is
  the same spatial-prior capability the fNIRS case and any future montage-aware method reuse.
