# Frozen foundation-model head-engineering for cross-subject perception (2026-07-13)

Mind-map / roadmap consolidation for the Stage-3 EEG→image line after the CBraMod foundation-model epic
(bji). Companion to bd tasks **nm5, j4l, 29z, ooi, 71n** and the bd memories
`frozen-cbramod-perception-head-search-nm5` + `cbramod-backbone-is-hard-wired-to-200hz`.

## Where we were

CBraMod fine-tune beat from-scratch NICE cross-subject single-trial (~2.2% vs ~1.6%, epic bji). But full
backbone fine-tuning is the expensive, lower-yield lever. Working hypothesis (owner): **for a pretrained
backbone, head engineering is higher-yield than backbone adaptation.** So: freeze the backbone, precompute its
features once, and iterate head architectures cheaply (seconds/epoch, no backbone forward per step —
`frozen_head.py`).

## What the frozen features actually are

Backbone I/O is `[B, C, S, P]` = channels × time-patches × points-per-patch. CBraMod's patch is **fixed at 200
points**, and its patch embedding bakes in an **rfft spectral embedding** (101 bins) — so the **sample rate is
hard-wired to 200 Hz, 1 patch = 1.0 s** (verified in `external/CBraMod`: TUAB `reshape(16,10,200)` on 10 s,
BCI-IV-2a `22×4×200` on 4 s).

For THINGS-EEG2 the stimulus epoch is 1.0 s → `S = 200//200 = 1`: **the whole trial is one patch.** The frozen
feature is therefore `[63 channels, 1 patch, 200-d]` = **63 channel-tokens on the 2D scalp, no time axis**. Fine
ERP timing is compressed *inside* the single patch by the frozen weights.

Consequences (the trap avoided):
- **Cannot** get `S>1` by resampling up — 400/800 Hz makes each 200-pt patch 0.5/0.25 s, so its spectral bins
  map to the wrong physical frequencies and corrupt the pretrained prior.
- **Cannot** use smaller patches — `proj_in` is trained for 200 pts.
- At `resample=200` we are correct / in-distribution.
- **Sub-patch temporal signal is reachable only by weight adaptation** (full fine-tune, or LoRA — bd 29z).
  Since the endpoint is image *reconstruction*, coarse temporal may be acceptable — object identity is largely
  spatial + coarse dynamics.

## The frozen head-search finding (bd nm5)

Matched sweep, `train[1,2,3,4]→test5`, S=1/200 Hz, identical cached features:

| head | what it does | single top-1 | concept top-1 |
|---|---|---|---|
| mean_lin | mean-pool → linear | 0.56% (chance) | 0.50% |
| mean_mlp | mean-pool → MLP | 0.60% | 1.00% |
| attn_mlp | attention-pool → MLP | 0.65% | 1.50% |
| **flat_mlp** | **keep all 63 tokens → MLP** | **1.18–1.21%** | **3.00%** |

**Spatial preservation is the lever, not head depth.** Pooling over channels (mean/attn) ≈ chance; keeping every
channel-token (`flat`) ~doubles it to ≈ NICE (1.6%) **with no fine-tuning**. The `mean_lin` floor is identical at
60 and 120 ep — a chance ceiling, not undertraining. `flat` converges ~1.2%.

## The geometry lever (bd j4l, in progress)

`flat` still treats the 63 tokens as an **unordered bag** — it ignores *where each electrode sits*. The remaining
structure is scalp geometry: the 200-d vectors live on a 2D plane. Fold by geometry:
- **pos_attn** — electrode-position embedding + self-attention (the plain attn failed *because it had no
  positions*).
- **topo_cnn** — RBF-interpolate features onto a scalp grid + 2D CNN (Bashivan 2016); interpolation is a fixed
  operator (one einsum/batch).
- **GCN** — electrode-adjacency message passing (strongest inductive bias, most machinery; not yet built).

Beating `flat` (1.2%) and NICE (1.6%) frozen ⇒ spatial structure was the missing signal, and we get a cheap,
principled head that beats from-scratch **without any backbone training**.

## The label we hit

`[63, 200]` frozen features → head → **512-d unit vector** = open_clip **ViT-B/32** image embedding
(L2-normalized), one per trial. Loss = InfoNCE (cosine direction only).

For **reconstruction** (the real endpoint): retrieval top-k ("right nearest neighbor") is necessary but not
sufficient. Add **MSE-to-CLIP-target** so the predicted vector is a faithful decoder-conditioning latent, and
consider a larger CLIP (ViT-L/14, 768-d) for fidelity (bd ooi). Pipeline: EEG → predicted CLIP → diffusion/unCLIP
decoder (bd 71n).

## Roads (bd)

1. **j4l** — geometry-aware frozen head (topo-CNN / pos-attn / GCN). *In progress.*
2. **29z** — LoRA / low-rank CBraMod adaptation — the cheap sub-patch-temporal unlock.
3. **ooi** — reconstruction objective (InfoNCE + MSE) + larger CLIP target.
4. **71n** — EEG→image reconstruction pipeline (predicted CLIP → generative decoder).
5. **830** — harden the eventual winner (full LOSO + multi-seed) once a number moves.
6. **gvw** — port cardioseg's architecture-gate updates back here.
