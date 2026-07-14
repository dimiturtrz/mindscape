# Continuous perception eval metrics — what the angular error reveals (bd 2y7k)

Retrieval top-k accuracy collapses the ranked 200-way candidate list to one bit (did #1 hit), throwing away
geometry: a rank-2 miss 5° off and a rank-180 miss 120° off score identically. We added continuous extras
(`Nice.retrieval_continuous`) alongside accuracy — never replacing it (accuracy stays the field-comparable
number): `cos_to_true` (mean/std), a discrimination `margin` (cos-to-true − mean cos-to-others), `mean_rank`
of the true concept, and — because CLIP concept vectors cluster — a `cos_to_true_z` measured against the
candidate bank's own off-diagonal cosines (the "random concept-pair" baseline).

## The concept space is clustered, not uniform

Two *different* THINGS test concepts sit at cosine **0.334 ± 0.092** in CLIP space (200 concepts, pairs up to
0.80), not ~0. So an absolute `cos_to_true` is not self-interpretable — a bare 0.35 is the random-pair floor,
not "35 % aligned." Every downstream reading is relative to this.

## Frozen CBraMod head zoo — the readout (train[1-4]→test5, 120 ep, off the disk feature-cache)

| arm      | single-top1 | cos-to-true | z     | margin | mean-rank |
|----------|-------------|-------------|-------|--------|-----------|
| mean_lin | 0.60 %      | 0.001       | −3.63 | 0.001  | 97.2      |
| gcn      | 1.07 %      | 0.026       | −3.36 | 0.009  | 82.0      |
| flat_mlp | 1.21 %      | 0.054       | −3.05 | 0.009  | 75.5      |
| pos_attn | 1.59 %      | 0.012       | −3.51 | 0.017  | 73.9      |
| topo_cnn | 1.75 %      | 0.058       | −3.01 | 0.015  | 69.6      |

(random-pair cos 0.334 ± 0.092; chance top1 0.5 %; mean-rank chance ≈ 100)

## Three findings

**1. Every arm points ~orthogonal to the true concept.** cos-to-true 0.001–0.058 → arccos ≈ **86–90°**, for
all arms *including the best*. The encoder never lands *near* the right concept in absolute terms — not 30°,
not 120°, ~90° everywhere.

**2. Raw cos / z are geometry-confounded; margin + mean-rank carry the signal.** All z are −3.0 to −3.6: the
EEG embedding is *farther* from its own true concept than two random concepts are from each other. The cause
is a fixed geometry offset — EEG embeddings sit *outside* the tight CLIP concept cluster, so absolute
cos-to-any-concept is ~0 regardless of arm. What actually tracks accuracy is **mean-rank** (97→70, monotone
with top1 up to the pos_attn/flat swap) and **margin** (0.001→0.017). InfoNCE optimizes *relative* closeness,
so the meaningful continuous metric is relative — exactly what the loss predicts (raw cos does *not* track:
pos_attn's 1.59 % comes with a low cos 0.012 but the highest margin 0.017; topo's high cos 0.058 is
geometry, not quality).

**3. Retrieval lives entirely in a tiny positive margin.** topo's 1.75 % (3.5× chance) is produced by a
margin of 0.015 — the true concept is a *hair* closer than the other 199 while absolutely ~90° away. The
single-trial cross-subject "signal" is a faint relative tilt, not a semantic hit.

**Does higher accuracy carry tighter angular error?** No. cos_std is ~0.02–0.03, flat across arms; the
accuracy gap is a mean margin/rank shift, not a variance tightening.

## Consequence for how we read perception numbers

- Report **mean-rank** and **margin** as the continuous companions; treat raw `cos_to_true`/`z` as a geometry
  diagnostic (how far EEG embeddings sit from the CLIP concept cloud), not an arm-quality signal.
- A method that "moves the number" should move margin/mean-rank, and — per the eval-side mirror — its
  `val_cos_to_true` should track `val_top1`/loss during training (flag it if not).
- Caveat: these are *single-trial* embeddings. The trial-averaged prediction (`concept_avg`, higher accuracy)
  may align better; the ~90° per-trial figure is the noisy single-trial geometry, not the averaged one. A
  trial-averaged continuous readout is the natural next cut.
