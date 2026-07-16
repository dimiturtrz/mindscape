# Geometry prior on the trainable encoder: does the frozen-head win survive training? [DRAFT — arms running]

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

| λ (geo) | single top1 | single top5 | concept top1 | concept top5 | best val |
|---------|-------------|-------------|--------------|--------------|----------|
| 0 (baseline) | _pending_ | | | | |
| 0.005 | _pending_ | | | | |
| 0.02 | _pending_ | | | | |

## Verdict

_pending arm completion_

## What it means for the roadmap

_pending — the mechanism question (does geometry survive training?) decides whether ladder step 1 promotes,
and whether to carry the prior to the fNIRS n-back path (bd 1x0's "fNIRS first" argument: genuinely-local
channels where adjacency is real new info, unlike volume-conduction-smeared EEG)._
