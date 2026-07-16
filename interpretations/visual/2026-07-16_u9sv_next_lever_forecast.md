# u9sv forecast: which lever next, and is it run-worth?

**Date:** 2026-07-16 · **Task:** Stage-3 perception (EEG→image), cross-subject single-trial retrieval ·
**Bead:** u9sv (prior-mining run-gate) · **Purpose:** forecast the number a lever moves BEFORE spending GPU, so
the queue is chosen by prior strength, not tried blind.

## The established priors (what we already know moves / doesn't)

- **Hard band 1.6–2.3% single-trial top1** (train4/test5, chance 0.5%). [stage-3-single-trial]
- **Transfer levers all null** — input-align (dpi), domain-adversarial (36g), concept-aware InfoNCE (lbd). The
  wall is **single-trial SNR / encoder capacity, not subject-shift**. [perception-cross-subject-hard-floor]
- **Capacity ADAPTED is the one thing that broke the floor** — CBraMod full-ft **2.38**, LoRA **2.10**, both >
  NICE 1.60; frozen probe fails (0.63). The lever is *adapting pretrained capacity*, not architecture tricks on
  a from-scratch net. [foundation-frozen-probe-loses, cbramod-rank-8-lora]
- **Frozen-regime geometry won (topo 1.75), but on the TRAINABLE encoder a geometry prior is a regularizer,
  not a floor-breaker** (this batch, bd 1x0, 3 arms λ0/0.005/0.02): single-top1 1.53→1.65→1.67 (+0.14 =
  single-seed noise, forecast to wash per 36g), while the coherent signal is regularization (λ0.02 margin +25%,
  best-val ep19 vs 47). A trainable conv re-learns channel geometry from data, so imposing it is redundant on
  volume-conduction-smeared EEG — confirms 1x0's own "fNIRS-first" caveat. [2026-07-16_geometry_prior_trainable_encoder]

## Forecast per candidate lever

| lever | what it changes | forecast on single-trial top1 | run-worth |
|-------|-----------------|-------------------------------|-----------|
| **ooi** — ViT-L/14 target + MSE term | richer 768-d target (vs 512-d) + hit-the-embedding loss | target-dim is **not** the SNR floor → **modest** retrieval lift at best; MSE term aims at recon fidelity, orthogonal to (maybe costs) top-k | **BUILD** — high strategic value: unblocks the recon pipeline (71n), and richer-target × adapted-capacity is the untested combination. Test on the **winning encoder (ft/LoRA)**, not NICE. |
| **mnr** — hard-negative mining | batch composition / online hard-neg weight | efficiency, **not a number-mover** on top1 (hard_beta already wired, washed) | **DEFER** — low prior; a training-efficiency multiplier, not a floor-breaker. |
| **1x0 step-2/3** (perception) | spatial basis / ROI pooling on trainable EEG | if step-1 smoothness washes, harder spatial constraints wash the same way (encoder learns geometry) | **REDIRECT** — take 1x0 to its **fNIRS-first** case (genuinely-local channels, adjacency = real info), not perception step-2. |
| capacity: bigger backbone / longer ft | more params / schedule | the proven axis (ft>NICE); LoRA/ft undertrained (val still climbing) → **longer schedule is free headroom** | **CHEAP FIRST** — extend the ft/LoRA schedule before new objectives; lowest-effort move on the proven lever. |

## Recommended queue (post-geo)

1. **Longest-ft / LoRA schedule extension** — cheapest, on the proven capacity lever, both current fts undertrained (lower bounds). Quick keep/kill vs 2.38.
2. **ooi ViT-L/14 target on the ft/LoRA encoder** — the richer-target × capacity combination; primarily to unblock recon (71n), retrieval delta a bonus. Keep the MSE term OFF the retrieval metric (measure both separately).
3. **1x0 → fNIRS n-back** — the geometry prior's real untested case, per its own physics argument.

**Not worth GPU now:** mnr (efficiency), perception 1x0 step-2/3 (structure priors wash on trainable EEG),
any further transfer-invariance lever (three already null).

## Status

Geo verdict in (λ0.02 landed): mild regularizer, headline within noise → confirmed. The queue above stands —
next perception spend = capacity (longer ft schedule → ooi ViT-L/14 target), 1x0 redirected to fNIRS n-back,
mnr + perception structure-priors deferred.
