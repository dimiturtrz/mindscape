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

**The point is the gap.** Within-subject concept-avg top1 (15%, 30× chance) craters cross-subject. Concept-avg
lifts with training subjects (2% train-1 → 6% train-4 LOSO), but **single-trial — the deployment-real metric —
stays pinned near the floor (~1.9%, ≈4× chance)** no matter what. Within-subject is comparable to the NICE
family; the honest, leak-free cross-subject single-trial number is what the field under-reports.

## Closing the cross-subject gap — levers tried (honest negative)

Stages 1-2 closed their cross-subject gaps by removing per-subject *distribution displacement* (MI: Riemannian
re-centering; workload: per-subject z-score). We transplanted that logic here and tested it under **3-seed**
rigor (train-4 → test-5, matched config). It does **not** transfer to the single-trial headline:

| lever (single-trial top1) | axis · mechanism | result |
|---|---|---|
| baseline | — | **1.87 ± 0.14 %** |
| per-subject signal re-centering | *input space* · whiten each subject's epochs `M⁻¹ᐟ² X` ([`covariance.py`](../../../core/features/eeg/covariance.py)) | ~1.71 (flat; concept-avg *hurt*) |
| domain-adversarial | *embedding space* · GRL + subject discriminator ([`nice.py`](../../../neuroscan/models/nice.py)) | 2.00 ± 0.06 (**Δ +0.13, within noise**) |
| concept-aware soft InfoNCE | *loss* · CLIP-similarity soft targets ([`nice.py`](../../../neuroscan/models/nice.py)) | 1.73 (washes) |

Three levers on **three different axes** — input alignment, embedding invariance, loss quality — and none moves
the single-trial-top1 number beyond noise. That breadth is the point: it's not one idea failing. *Why they
fail, diagnosed:* re-centering whitens an **ill-conditioned** per-subject covariance (cond ≈ 1500), amplifying
noise channels ~38× and *hurting* the averaged metric while leaving single-trial flat (a shrinkage-regularised
variant recovers the loss but not a gain); the adversary *does* act (consistent val-drop, a weak **top5** +0.43
pp ≈ 2.5 σ) but subject-invariance isn't the missing piece, and a λ-ramp doesn't rescue it; soft-negatives fix a
real false-negative but don't lift transfer. **Conclusion: the single-trial cross-subject gap is a genuine hard
floor for this compact encoder at this scale — bounded by single-trial SNR / capacity, not by a fixable
subject-shift or loss bug.** The remaining untested axis is *capacity* (a bigger encoder / longer schedule) — a
different question (representation power, not transfer). All three operators are kept opt-in + documented
(`TrainConfig.recenter`, `.adversarial`, `.soft_tau`); the negative is the finding.

## Critique / limitations (what's weak, on purpose visible)

- **The single-trial floor is the honest ceiling** — not closed by the transfer-alignment levers above; the
  remaining untested angles are *capacity* (bigger encoder / longer schedule — 693k params, early-stops ~ep 9-15)
  and a loss-quality fix (below), not another subject-invariance trick.
- **InfoNCE false negatives** — same-concept-different-image pairs in a batch are treated as negatives though
  their CLIP targets are ~0.7 similar; a concept-aware batch sampler or soft targets would help. Not yet tested.
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

## The over-reporting audit — how much the field's usual numbers inflate

The commonly-quoted perception number is within-subject, concept-averaged. The defensible one is
cross-subject, single-trial. The same encoder scored four ways
([`retrieval_audit.py`](retrieval_audit.py)):

| top-1 (top-5) | single-trial | concept-averaged |
|---|---|---|
| **within-subject** | 4.0% (14.5%) | **14.8% (39.5%)** ← usually quoted |
| **cross-subject** | **1.9% (7.6%)** ← robust | 4.8% (14.3%) |

*(measured, 2-subject mean; chance 0.5%.)* Two independent leaks stack: seeing the test *person*
(within→cross, 4.0 → 1.9%) and averaging test *repeats* (single→avg, 1.9 → 4.8%) — together an **8.0× gap**
(14.8% vs 1.9%, +12.9 pts) between the commonly-quoted headline and the defensible number.

**Zero-shot is verified, not assumed** — the train/test concept sets are checked disjoint on concept *names*
(1,654 train / 200 test / **0 overlap**; comparing split-local indices would have falsely flagged all 200).
**Confidence calibration** ([`../../evaluation/retrieval.py`](../../evaluation/retrieval.py)) asks the
deployable question the top-k can't: when the retrieval is confident, is it right? — ECE + a hit-vs-miss
confidence gap.

## Continuous metrics — the retrieval is a tiny margin, not a semantic hit

Top-k accuracy is one bit per trial (did #1 hit); it hides *how* the misses miss. `Nice.retrieval_continuous`
adds the geometry as **extras** (never replacing accuracy): cos-to-true, a discrimination **margin**
(cos-to-true − mean cos-to-the-other-199), **mean-rank** of the true concept, and a **z** against the concept
bank's own random-pair cosine — CLIP concepts cluster (two *different* THINGS concepts already sit at cos
0.334 ± 0.09), so absolute cosine isn't self-interpretable. Backfilled across the frozen CBraMod head zoo
(120 ep, [writeup](../../../learning/2026-07-14_perception_continuous_metrics.md)):

| frozen head | single-top1 | cos-to-true (≈ angle) | margin | mean-rank |
|---|---|---|---|---|
| mean_lin | 0.60% | 0.001 (~90°) | 0.001 | 97 |
| gcn | 1.07% | 0.026 (~89°) | 0.009 | 82 |
| flat_mlp | 1.21% | 0.054 (~87°) | 0.009 | 76 |
| pos_attn | 1.59% | 0.012 (~89°) | 0.017 | 74 |
| topo_cnn | 1.75% | 0.058 (~87°) | 0.015 | 70 |

**Every arm points ~90° from the true concept — even the best.** The encoder never lands *near* the right
concept absolutely; retrieval lives entirely in a **tiny positive margin** (topo's 1.75% = 3.5× chance from a
0.015 margin). Raw cos-to-true is geometry-confounded — EEG embeddings sit *outside* the tight CLIP concept
cluster, so every arm reads ~orthogonal — while **margin and mean-rank** are what track accuracy, exactly what
InfoNCE optimizes (relative closeness, not absolute alignment). The single-trial cross-subject "signal" is a
faint relative tilt, not a semantic hit. *(single-trial embeddings; the trial-averaged cut may align better —
untested.)*

## Cross-dataset zero-shot — the hardest test, a measured null

[`cross_dataset_eval.py`](cross_dataset_eval.py) trains on **THINGS-EEG1** (Grootswagers ds003825 — 50 subj,
63-ch, [adapter](../../../core/data/eeg/things_eeg1.py) real-data validated) and retrieves on **THINGS-EEG2** —
different people, different rig, same 1,854 concepts; EEG2's test concepts are held out of EEG1
([bridge](../../evaluation/cross_dataset.py)) so it's cross-dataset *and* concept-zero-shot *and* cross-subject
at once. **Result: a measured null.** The EEG1 encoder learns (within-EEG1 val 2.4%, ~5× chance) but transfers
at chance (0.5% top-1), and **montage-aligning the 62 shared electrodes doesn't rescue it**
(`common_channel_order` / `align_channels` — the rigs share all but Fz/Cz, in scrambled order). So it's not a
channel-order artifact; the datasets are just too far apart (reference, 10 vs 5 Hz RSVP, EEG1's weaker
single-shot SNR). *Caveat: reference/filtering aren't harmonized either, so this is "naive transfer fails," not
"impossible" — a common-reference re-projection is the untested next step.*
