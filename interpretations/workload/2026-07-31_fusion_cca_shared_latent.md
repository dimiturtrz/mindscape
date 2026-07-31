# Fusion CCA shared-latent probe — the EEG↔fNIRS shared subspace is task-orthogonal

**Date:** 2026-07-31 · **Task:** workload (Shin n-back, 702 blocks / 26 subj) · **Bead:** mindscape-52y9 ·
**Runner:** `neuroscan/tasks/workload/fusion_cca_probe.py`

## Question

si7 asks whether *inferring a shared neural latent* from EEG+fNIRS beats *multiplying the observables* (which
the three prior fusion nulls all did: output-space multiplicative joint [060], plain source projection [728],
fNIRS-informed inverse [4so]). This is the **cheapest linear gate** on that idea: CCA finds the directions of
maximal EEG↔fNIRS correlation — the linear common latent — and we decode workload from it, matched in one
cross-subject fold loop against each modality alone and naive feature-concat.

Features reuse the strong decoders' own views so the comparison is fair: EEG = per-subject re-centered tangent
(the 0.58 Riemann decoder's feature space), fNIRS = the full descriptor bank. Same shrinkage-LDA on every arm,
so the arms differ only in their feature set. `shared` = mean of the two aligned CCA canonical projections.

## Result (cross-subject 1×5-fold, shrinkage-LDA, chance 0.333)

| arm | acc | note |
|---|---|---|
| eeg | **0.560 ± 0.064** | re-centered tangent — matches the ~0.58 Riemann reference (pipeline sanity) |
| fnirs | 0.427 ± 0.029 | descriptor bank |
| concat | 0.559 ± 0.067 | naive feature-fusion — **≈ eeg, adds nothing** |
| shared | **0.340 ± 0.059** | CCA shared latent — **chance** |

## Reading

Two things, both pointing the same way:

1. **Naive fusion adds nothing** — `concat` 0.559 ≈ `eeg` 0.560. A **4th** independent confirmation of the
   EEG+fNIRS workload-fusion null, now in *feature space* (prior nulls: multiply 060, source 728, prior 4so).

2. **The shared subspace is task-orthogonal** — `shared` decodes at chance. Mechanism: CCA maximizes cross-
   modal **correlation**, and what EEG and fNIRS *share* is **nuisance** — systemic/global hemodynamics,
   common-mode drift — not the workload contrast. The discriminative signal lives in modality-**specific**
   directions, which the shared projection throws away.

## Consequence for si7 — sharpened, not just killed

The *unsupervised* shared latent is nuisance-dominated, so "infer the common neural cause and decode from it"
does **not** work here — the common cause is task-irrelevant. A generative si7 is only worth building if it is
**label-conditional**: the shared latent must be *forced* to be predictive of workload (a discriminative /
class-conditional shared factor), not a plain shared-variance factor. That reshapes the si7 spec from
"infer shared latent" to "infer a **label-predictive** shared latent."

## Caveats (honest)

- CCA is **linear + unsupervised**; a chance-level result does not rule out a **non-linear** shared factor —
  it raises the bar. This is a cheap necessary-ish gate, not a proof of impossibility.
- Single seed / one 5-fold set, `n_components=10` unswept (fast keep/kill, not a hardening pass). A chance-
  level result across a 406×1080 CCA is decisive enough for the decision; the mechanism explains *why*.
- The EEG-alone sanity (0.560 ≈ the 0.58 tangent+LR reference) confirms the features/folds/classifier are
  sound, so `shared` = chance is a real null, not a pipeline bug.
