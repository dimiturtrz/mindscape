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

## Results — honest, leak-free (preliminary, 2 subjects)

Model selection (which epoch to keep) is on a **validation set of 165 held-out *training* concepts** — the
test subject/concepts never touch checkpoint selection. Numbers are the best-val checkpoint (ep 9). Chance =
1/200 = 0.5% (top1), 2.5% (top5).

| | within (s01→s01) | cross (s01→s02) |
|---|---|---|
| single-trial top1 | 4.06% | 1.45% |
| single-trial top5 | 14.0% | 6.35% |
| concept-avg top1 | **15.0%** | **2.0%** |
| concept-avg top5 | 35.5% | 10.5% |

**The point is the gap, not the peak.** Within-subject concept-avg top1 (15%, 30× chance) craters to 2%
(4× chance) cross-subject — a 7.5× drop. Same finding as motor imagery (0.71→0.36): the model learns a
*subject-specific* EEG→semantic map that barely transfers. Within-subject is comparable to the NICE family;
the honest cross-subject number is the contribution the field under-reports.

## Critique / limitations (what's weak, on purpose visible)

- **Single training subject** — cross here is train-on-*one*-person → test-on-another, the hardest, lowest
  case. Real LOSO (train on many, hold one out) will lift cross off the floor; it widens automatically as the
  download completes (currently 2–4 of 10 subjects).
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
