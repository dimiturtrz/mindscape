# Visual — EEG→image decoding (Stage 3, perception axis)

The perception rung of the [field-map](../../../README.md): decode *which image a subject is viewing* from
EEG, by retrieval in CLIP space. EEG epoch → NICE encoder → 512-d embedding, trained contrastively (InfoNCE)
against the viewed image's CLIP embedding; evaluated by zero-shot retrieval over the 200 held-out THINGS
concepts. Dataset: **THINGS-EEG2** (Gifford 2022), our own preprocessing off the raw (see
`core/data/eeg/things_eeg2.py`). Method after **NICE** (Song et al., ICLR 2024).

```bash
uv sync --extra data --extra visual
python -c "from core.data.eeg.things_eeg2 import download; download()"      # raw EEG (~110 GB) + images
python -m neuroscan.tasks.visual.train_nice --train 1 --test 1              # within-subject
python -m neuroscan.tasks.visual.train_nice --train 1 2 3 --test 4          # cross-subject (LOSO)
```

## Results — honest, leak-free

Model selection (which epoch to keep) is on a **validation set of 165 held-out *training* concepts** — the
test subject/concepts never touch checkpoint selection (best-val checkpoint). Chance = 1/200 = 0.5% (top1),
2.5% (top5). All 10 subjects available; cross-subject is a LOSO (train N, hold one out).

| concept-avg | within (s01→s01) | cross · train-1 (s01→s02) | cross · LOSO train-4 (s01-04→s05) |
|---|---|---|---|
| **top1** | **15.0%** | 2.0% | **6.0%** |
| top5 | 35.5% | 10.5% | 25.0% |
| single-trial top1 | 4.06% | 1.45% | 1.94% |

**The point is the gap — and that it moves with subject count.** Within-subject concept-avg top1 (15%, 30×
chance) craters cross-subject, but the crater *shrinks as training subjects are added*: 2% (train-1) → **6%**
(train-4 LOSO), top5 10.5% → 25%. Single-trial stays near the floor (~2%) — that's the genuinely hard part.
Same shape as motor imagery (0.71→0.36): a *subject-specific* EEG→semantic map that transfers better the more
subjects it sees. Within-subject is comparable to the NICE family; the honest cross-subject number **and its
subject-count scaling** is the contribution the field under-reports. (A full train-9 LOSO is one run away —
the DataLoader is numpy-backed + index-viewed so 9 subjects' epochs fit in RAM without the doubled copy.)

## Critique / limitations (what's weak, on purpose visible)

- **Subject count** — the train-4 LOSO (6%) already shows cross lifts off the train-1 floor (2%); the full
  train-9 LOSO (all 10 downloaded) is the remaining stronger number, one memory-fixed run away.
- **InfoNCE false negatives** — same-concept-different-image pairs in a batch are treated as negatives though
  their CLIP targets are ~0.7 similar; a concept-aware batch sampler or soft targets would help. Not yet done.
- **Compact encoder, short training** — 693k params, early-stops ~ep 9; a bigger encoder / more epochs / LR
  schedule would raise the within number (the gap is the story, so not chased here).
- **Concept-averaged retrieval** averages 80 test reps — reported alongside single-trial (the deployment-real
  one), never instead of it. Single-trial is the honest headline metric.

## Fixes already applied (found while building)

- **Per-channel z-score** in the adapter — EEG is in volts (~1e-5), which left BatchNorm's running variance
  ill-conditioned and collapsed eval-mode embeddings to chance. Standardizing to O(1) fixed it (train/eval
  retrieval matched 98% on an overfit-a-batch check).
- **Leak-free early stopping** — validation on held-out *training* concepts, not the test set (picking the
  best *test* epoch would be leakage — the field's sin).
- **Seed** for reproducibility; **eval batch ≤1024** (batch ≥2048 trips a cuDNN illegal-access on this conv
  shape, Blackwell / cu130).
