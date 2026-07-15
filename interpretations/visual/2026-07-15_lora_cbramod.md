# LoRA fine-tune of CBraMod — the cheap sub-patch temporal unlock (bd 29z)

**Date:** 2026-07-15 · **Task:** Stage-3 THINGS-EEG2 cross-subject single-trial retrieval (train[1-4] → test5)

## Question

CBraMod is 200 Hz / 1 s-patch, so our 1 s stimulus is a single patch (S=1): fine ERP timing is compressed
inside the frozen patch representation, unreachable by any head or by resampling. Full fine-tune adapts all
4.9M params and works (single-seed 2.38% single-trial top1 vs NICE 1.60%), but it is the expensive lever.
Does **rank-8 LoRA** — a low-rank residual on a *subset* of the backbone weights — cash most of that gain
at a fraction of the trainable params?

## Method

`LoraLinear` wraps a frozen `nn.Linear` with `ΔW = (α/r)·B·A` (r=8, α=16; `B` zero-init so the adapted
forward equals the pretrained one at step 0). `Lora.inject` swaps the transformer feed-forward `linear1`/
`linear2` in all 12 layers (24 linears) — the module-callable surface, and ~4× wider than attention.

The fused `nn.MultiheadAttention` is **out of reach**: q/k/v are a single `in_proj_weight` Parameter and
`out_proj` is read by attribute inside torch's attention kernel (not called as a module), so neither a
wrapper nor PEFT's `nn.Linear`-targeted LoRA can adapt them. PEFT would therefore add a dependency for the
same FFN-only `nn.Linear` subset — hand-rolled instead, no vendored dep.

Trainable: 557k / 5.44M = **10.2%** (LoRA A/B 192k + head 365k). Matched to the full-ft protocol:
resample 200, cross-subject train[1-4]/test5, `perception_lora.json` (60 epochs, lr 1e-3, batch 512).

## Result — KEEP

| metric (%)         | frozen probe | NICE | **LoRA** | full-ft |
|--------------------|:------------:|:----:|:--------:|:-------:|
| single-trial top1  | 0.63 (chance)| 1.60 | **2.10** | 2.38    |
| single-trial top5  | —            | 7.05 | **8.83** | 9.58    |
| concept-avg top1   | —            | 3.0  | **4.50** | 4.50    |
| concept-avg top5   | —            | 18.0 | **23.5** | 20.0    |

LoRA lands **between NICE and full-ft** on single-trial (2.10%, ~64% of the ft-over-NICE gap), **beats NICE
on all four metrics**, **matches full-ft on concept-top1**, and **beats full-ft on concept-top5** — all at
10.2% trainable params, adapting only the FFN while attention stays frozen. Best-val epoch = 59 of 60 (val
top1 6.54%, still climbing) → an **under-trained lower bound**, the same caveat that held for the full-ft run.

## Interpretation

The sub-patch temporal signal the frozen patch buries **is** reachable by a cheap low-rank adaptation of the
FFN — full backbone fine-tuning is not required to cash most of the perception gain. That the attention
mixing stays frozen and LoRA still reaches 2.10% says the adaptation the task needs is largely in the
per-token feed-forward transform, not in re-learning the attention pattern.

## Caveats

Single seed, one test subject (5), under-trained (best = last epoch), rank/α untuned, FFN-only (the fused
attention is unadaptable by this route). Directionally clear against three matched reference points, but
the headline vs full-ft (2.10 vs 2.38) is within the scatter these single-seed runs carry — LoRA is a
confirmed cheap **alternative** to full-ft, not a proven **improvement** over it. Hardening (multi-seed,
LOSO, longer schedule, rank sweep) is a follow-up, gated on this keep.
