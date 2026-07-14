# Fine-tuning an EEG foundation transformer: discriminative LR vs LoRA

Date: 2026-07-15
Context: fine-tuning CBraMod (~4M params, criss-cross patch-embedding transformer) for downstream
EEG→image retrieval. Current recipe = ONE LR (1e-4) for backbone+head via a shared AdamW; already
beats from-scratch (2.38% vs 1.6% single-trial top1). Question: improve via (a) discriminative /
layer-wise LR, or (b) LoRA adapters.

---

## TL;DR recommendation

1. **Cheapest high-signal first test = discriminative LR with a 2-group split** (backbone low, head high),
   matching CBraMod/LaBraM's own published recipe. Concretely: **layer-wise LR decay (LLRD) with decay
   0.65–0.75 across the transformer blocks + a head LR 10× the top-block LR**, AdamW, 3-epoch linear
   warmup → cosine to ~1e-6, weight decay 0.05. This is a param-group change to your existing optimizer,
   no new modules, ~30 min to run.
2. **LoRA IS worth building at this scale — the "LoRA only helps big models" folklore is false for EEG
   foundation models.** The single most relevant primary source (Panagos et al. 2025, on LaBraM which is
   5.8M params ≈ CBraMod's 4M) found **LoRA rank=2 BEAT full fine-tuning: 77.7% vs 74.2% mean accuracy,
   with 67.7k trainable params vs 5.85M.** But do it *second* — discriminative-LR full-FT is the simpler
   change and is the published CBraMod recipe, so it's the correct first move; LoRA is the follow-up that
   may push further AND regularize (its win over full-FT is plausibly a regularization effect on small
   downstream data).

---

## 1. Discriminative / layer-wise LR

### ULMFiT discriminative fine-tuning (the origin)
- Howard & Ruder 2018. Each layer gets its own LR via exponential decay: choose the LR for the **last
  (top) layer** η_L empirically, then η_{l-1} = η_l / **2.6** for successive lower layers. (2.6 ≈ decay
  factor 1/2.6 ≈ 0.385 per layer — aggressive.)
- Rationale: task-specific functionality lives in later layers; early layers hold stable low-level
  features → freeze them harder (lower LR) to prevent catastrophic forgetting.
- Paired in ULMFiT with **slanted triangular LR** (warmup then linear decay) and **gradual unfreezing**
  (unfreeze top layer first, then progressively lower layers each epoch).
- Sources: ULMFiT paper summaries; ritvik19 "Papers Explained 447"; mbrenndoerfer LLRD writeup.

### Layer-wise LR decay (LLRD) in BERT/ViT — the modern standard
- Same idea, gentler decay. Large LR on the **top** transformer block, multiplicative decay downward:
  `lr_layer_i = base_lr * decay^(N - i)` where i indexes from bottom.
- **Typical decay factor 0.65–0.95.** BERT fine-tuning defaults land around **0.65–0.9**; ViT/MAE
  fine-tuning commonly uses **0.65–0.75**. Optimal is task/dataset dependent (one cited study found 0.775
  best; another found best beat the 0.65 default by 1.3% under heavy masking). Head/classifier gets a
  slightly higher LR than the top block.
- The **simple 2-group split** (all backbone at LR_b, head at LR_h with LR_h ≈ 10× LR_b) is the pragmatic
  reduction of LLRD and is what most transfer-learning recipes actually use when per-layer tuning isn't
  worth it. fast.ai popularized both.
- Sources: mbrenndoerfer "Fine-tuning Learning Rates: LLRD, Warmup & Decay"; Towards Data Science
  "Advanced Techniques for Fine-tuning Transformers"; arxiv 2509.00027 (LLRD default 0.65).

### PyTorch implementation (param groups)
```python
# Coarse 2-group (do this first):
optim.AdamW([
    {"params": backbone.parameters(), "lr": 1e-5},   # backbone: low
    {"params": head.parameters(),     "lr": 1e-4},   # head: 10x
], weight_decay=0.05, betas=(0.9, 0.999))

# Per-block LLRD (do this if 2-group shows signal):
groups, N = [], len(blocks)
for i, blk in enumerate(blocks):          # i=0 bottom ... N-1 top
    lr = base_lr * (decay ** (N - 1 - i)) # decay=0.65..0.75
    groups.append({"params": blk.parameters(), "lr": lr})
groups.append({"params": head.parameters(), "lr": base_lr * head_mult})  # head_mult ~ 1..2 over top
optim.AdamW(groups, weight_decay=0.05)
```
Embedding/patch-embed layer = lowest LR (bottom of the stack). One `param_group` per block; the scheduler
scales all groups by the same factor so **each group keeps its own base LR through warmup+cosine**.

### Gradual unfreezing — does it help?
Helps most when downstream data is tiny and forgetting is severe; adds scheduling complexity. For a
foundation model that already transfers (you beat from-scratch), LLRD alone usually captures most of the
benefit. Treat gradual unfreezing as an optional third lever, not the first move.

---

## 2. LoRA for EEG / small transformers

### General folklore (partly wrong here)
- Common guidance: full-FT is manageable below ~1M params and LoRA "shines for large models." Typical LoRA
  knobs: **rank r ∈ {4,8,16}**, **alpha ≈ 2r**, target `q_proj`/`v_proj` first, or all-linear for more
  capacity. r=8→~2.7M trainable on a 3B model; r=16 often no better than r=8. (Raschka; mbrenndoerfer LoRA
  hyperparams; TrueFoundry.)

### The directly-relevant EEG evidence — this overrides the folklore
**Panagos, Barmpas et al., "Are Large Brainwave Foundation Models Capable Yet? Insights from Fine-tuning"
(arXiv 2507.01196, 2025)** — tested on **LaBraM (5.8M params, ≈ CBraMod scale)** and NeuroGPT:
- Ranks swept r ∈ {1,2,4,8,16} on attention + fully-connected layers; conv layers fixed rc=4 (LaBraM).
- **Full LaBraM fine-tune: 74.2% mean acc. LaBraM + LoRA (r=2): 77.7% mean acc — LoRA BEAT full-FT** — with
  **67,749 trainable params vs 5,854,288** (≈1.2%).
- **Frozen backbone (linear probe): ~63.5–66.3%** — far below LoRA's ~74–77%. Authors: "demonstrates the
  necessity of full-model fine-tuning, and in turn makes PEFT like LoRA extremely valuable."
- **Best module choice = combine multiple layer types.** "LoRA on only a specific layer (attention or
  convolution) usually yields lower performance than a combination of two or three." Best avg =
  **conv layers + (fully-connected OR attention)**. Single-module (attention-only) underperforms.
- Takeaway for us: at 4–6M EEG-foundation scale, **low rank (r=1–4) is enough, and adapting across
  attention+MLP(+patch-embed conv) matters more than the rank value.** LoRA's edge over full-FT is
  consistent with a regularization benefit on small downstream sets.

Other EEG PEFT work exists (Graph Adapter arXiv 2411.16155; TaKF+; EEG-FM-Bench benchmarks
frozen/full/LoRA) but Panagos is the cleanest same-scale head-to-head.

---

## 3. Published EEG-foundation fine-tuning recipes (quoted)

### CBraMod (ICLR 2025, Wang et al., arXiv 2412.07236) — YOUR backbone
Default downstream fine-tune:
- LR **1e-4** (reduced to **5e-5** where a task didn't converge), **AdamW β=(0.9, 0.999)**
- **50 epochs**, batch **64**, dropout **0.1**, **weight decay 5e-2**
- **CosineAnnealingLR**, 50-epoch cosine cycle, **min LR 1e-6**
- **gradient clip norm 1**, **label smoothing 0.1** (multi-class)
- Downstream variants seen: LR 1e-4 or 5e-4 with cosine + **warmup over first 20% of steps**.
- Note: the *released* CBraMod recipe uses a **single LR** (like your current setup) — it does NOT ship
  LLRD. So discriminative LR is a genuine, untested improvement axis for CBraMod specifically, not a
  reproduction of their recipe.

### LaBraM (ICLR 2024, Jiang et al., arXiv 2405.18765)
- Weight decay **0.05**, batch **32**, LR **1e-5**, **50 epochs**, **3 warmup epochs**
- **Layer decay = 0.65** (LLRD, exponential bottom→top) — i.e. LaBraM DOES use discriminative LR.
- A downstream LaBraM-encoder application used **layer decay 0.8**.

### EEGPT
- Uses **linear probes on a frozen backbone** for multi-task eval; per-task heads. Frozen-backbone is
  weaker than full/LoRA per the Panagos results above — not the strategy to copy for max accuracy.

Pattern across the field: **AdamW, weight decay 0.05, ~50 epochs, 3-epoch (or 20%-step) warmup, cosine to
~1e-6, and — for LaBraM — LLRD 0.65.** CBraMod is the outlier that ships a single LR, which is exactly the
gap you're targeting.

---

## 4. Warmup + cosine + discriminative LR interaction
- Warmup and LLRD are **orthogonal and composable**: warmup ramps a global multiplier 0→1 over the first
  few hundred steps / 3 epochs / 20% of steps; LLRD sets the **per-group base LR**; the scheduler multiplies
  every group by the same warmup/cosine factor. Each group keeps its relative scale throughout.
- Standard stack (LaBraM, MAE, BERT all do this): **linear warmup (3 epochs or ~10–20% steps) → cosine
  decay to min LR (~1e-6), with LLRD-set per-block base LRs.** This is the field-standard combination and
  what you should adopt wholesale.
- Warmup matters MORE with discriminative LR because the head (high LR) can destabilize early; warmup
  protects the pretrained backbone during the first steps when head gradients are largest.

---

## Concrete recipe to implement FIRST (cheapest high-signal)

Keep everything in your current CBraMod fine-tune (AdamW, 50 ep, bs 64, wd 0.05, dropout 0.1, grad-clip 1,
cosine→1e-6) and change only the optimizer to param groups + add warmup:

- **Backbone base LR = 1e-5**, **head LR = 1e-4** (10:1) — the 2-group split. (This alone tests the core
  hypothesis: is the head starved / backbone forgetting at a shared 1e-4?)
- Add **3-epoch linear warmup** before the cosine.
- If 2-group moves the number: upgrade to **per-block LLRD, decay 0.7** (start point between BERT-0.65 and
  gentle-0.9), head at 1× the top-block LR (LaBraM-style, layer decay 0.65–0.8), same warmup+cosine.
- Cost: one param-group refactor, one short run vs your existing 2.38% baseline. No new modules.

**Then (second test): LoRA r=2, alpha=4, on attention + MLP (+ patch-embed conv) of the transformer,
backbone otherwise frozen, head trained fully.** Panagos shows this can beat full-FT at exactly your scale
with ~1% of params. Worth building — but AFTER discriminative-LR full-FT, because full-FT with LLRD is the
smaller change and the published-recipe-aligned baseline that LoRA must then beat.

Verdict on "does LoRA dominate discriminative full-FT at 4M?": **not a priori — they're competitive and the
one same-scale EEG study puts LoRA slightly ahead.** Do the discriminative-LR full-FT first (cheap, aligned
with CBraMod's own recipe gap), then LoRA r≈2 as the follow-up; keep whichever wins on your retrieval
metric, and don't multi-seed until one of them moves the number.

---

## Sources
- CBraMod — arXiv 2412.07236; github.com/wjq-learning/CBraMod; alphaxiv overview 2412.07236v6
- LaBraM — arXiv 2405.18765 (ICLR 2024)
- Panagos et al., "Are Large Brainwave Foundation Models Capable Yet? Insights from Fine-tuning" —
  arXiv 2507.01196 (LoRA r=2 77.7% vs full-FT 74.2% on LaBraM 5.8M)
- EEG-FM-Bench — arXiv 2508.17742 (frozen/full/LoRA strategies)
- Graph Adapter for PEFT of EEG FMs — arXiv 2411.16155
- ULMFiT (Howard & Ruder 2018) — decay 2.6; ritvik19 "Papers Explained 447"; mbrenndoerfer LLRD writeup
- LLRD in BERT/ViT — mbrenndoerfer "Fine-tuning Learning Rates: LLRD, Warmup & Decay"; Towards Data Science
  "Advanced Techniques for Fine-tuning Transformers"; arXiv 2509.00027 (LLRD default 0.65)
- LoRA hyperparams — Raschka "Practical Tips for Finetuning LLMs Using LoRA"; mbrenndoerfer LoRA
  hyperparameters; TrueFoundry LoRA guide
- PyTorch param groups — pytorch.org/docs optim; PyTorch Forums "differential learning rate by parameter
  groups"
