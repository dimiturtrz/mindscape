# Neural-signal synthesis (EEG / fNIRS) — the 2024-2026 landscape + our-use verdict

**Date:** 2026-07-07 · **Why:** the fusion decode hit a small-data + physiological ceiling; "generation" is the
next stage. Prior notes were **GAN-only and dated** — this corrects that (diffusion is now SOTA) and gives an
honest verdict on what synthesis can/can't do *for us*.

---

## 1. Headline — diffusion has overtaken GANs (the correction)
The field moved **GAN → diffusion → flow-based**. For neural signals specifically (2024-2026):

- Diffusion models **match or surpass GANs in fidelity + coverage and are more stable to train**; GANs suffer
  training instability that limits reliable brain-signal generation.
- EEG diffusion work: MI-EEG synthesis ([arXiv 2510.17832](https://arxiv.org/html/2510.17832v1)), improved-DDPM
  augmentation ([PubMed 39693767](https://pubmed.ncbi.nlm.nih.gov/39693767/)), ERP conditional diffusion
  ([arXiv 2403.18486](https://arxiv.org/pdf/2403.18486)), a WGAN-GP-vs-diffusion benchmark for EEG channels
  ([OpenReview r0216Av8tP](https://openreview.net/pdf?id=r0216Av8tP)), and a diffusion-for-brain-imaging survey
  ([Brain Informatics 2026](https://link.springer.com/article/10.1186/s40708-026-00301-5)).
- **Honest caveat:** diffusion for **time-series** (vs images) is still under-explored/maturing — the fidelity
  wins are strongest on image/audio; signal-domain results are newer and thinner.

So: **GAN (Ye 2021, Neurophotonics — the only ref in our old notes) is the dated baseline, not the recommendation.**

## 2. The cross-modal find that cuts against our fusion — SCDM
**[SCDM: Unified Representation Learning for EEG-to-fNIRS Cross-Modal Generation in MI-BCIs](https://arxiv.org/abs/2407.04736)**
— a **spatio-temporal controlled diffusion model** (SCG maps EEG→fNIRS reps; MTR learns a unified temporal/spatial
latent) that **generates fNIRS *from* EEG** (unidirectional). Motivation: co-locating EEG+fNIRS sensors is hard, so
record EEG only and *synthesize* the fNIRS.

**Their headline result:** *"joint classification of EEG + **synthetic** fNIRS is comparable to or even better than
EEG + **real** fNIRS."* (Datasets/subjects + code URL not in the abstract; not verified whether Shin n-back.)

**Our honest reading — this CONFIRMS our redundancy finding, it doesn't refute it.** You can only regenerate fNIRS
from EEG and have it work as well *if fNIRS carries ~no information EEG lacks* — i.e. `fNIRS ≈ f(EEG)` for the task.
That is the **identity trap made explicit**. SCDM's win is **convenience** (skip the fNIRS hardware), not **added
information**. It is independent evidence for the same thing our fusion nulls showed: on these tasks fNIRS is
largely redundant with EEG. (If real fNIRS carried independent signal, an EEG→fNIRS generator could not match it.)

## 3. Physics-forward simulators (theory-driven — the interpretable path)
- **2026 High-Fidelity 3D fNIRS simulator** ([arXiv 2605.30552](https://arxiv.org/abs/2605.30552)) — mesh Monte-Carlo
  photon transport, full-head, parameterized double-gamma HRF (amplitude κ / peak τ / width ω per region) +
  systemic physiology (cardiac 0.6-2.5, respiratory 0.2-0.6, Mayer 0.06-0.14 Hz) + motion artifacts
  (spikes/baseline-shift/dropout), **flexible montage** (ICP-registered), **fNIRS-only**, open-source (CC-BY, code
  in their Appendix C). Their own limit: **canonical HRF shapes "insufficient for longer-duration / quasi-continuous
  stimulation"** — i.e. edge-of-scope for n-back's ~40 s blocks.
- **2024 precursor** ([arXiv 2405.11242](https://arxiv.org/abs/2405.11242)) — Monte-Carlo + parametric head models,
  Docker/Xarray, cloud-scale generation.
- **SEREEGA** (EEG analog, not re-verified here) — lead-field forward simulation for EEG; the piece a *paired*
  physics generator would need alongside the fNIRS sim.

## 4. Theory-driven vs data-driven — which for us (n ≤ 30 subjects)
| | physics-forward (sim) | diffusion (learned) | GAN |
|---|---|---|---|
| interpretable / mechanistic params | **yes** | no | no |
| ground-truth labels | **yes** (you set the sources) | no | no |
| needs large training data | **no** | yes | yes |
| fidelity to real recordings | good, model-bounded | **highest** (image/audio; signals maturing) | lower, unstable |
| cross-modal (EEG↔fNIRS) | needs both forward models | **yes** (SCDM) | possible |
| memorization / identity-trap risk | low (physics prior) | **higher** | higher |

For **small-N + our honesty bar**, physics-forward wins for *validation*; diffusion is the learned cross-modal path
but data-hungry and its own results confirm redundancy here.

## 5. The identity trap — now with evidence
**Synthesis adds variety, not discriminative signal.** No generator (GAN, diffusion, or physics) manufactures the
2-vs-3-back signal physiology doesn't produce. SCDM's *"synthetic ≈ real fNIRS"* is the cleanest demonstration: it
only works because fNIRS ⊂ EEG's information on that task.

## Our-use verdict (what to actually do)
1. **Physics sim (2605.30552) → validate our fNIRS extraction.** Plant a known HRF lag/shape under realistic
   systemic + artifacts; check `estimate_coupling` recovers it and `cbsi_neural` rejects the common-mode. First
   **independent** (non-circular) check of the extraction. Also domain-randomization augmentation for robustness.
   **NOT a decode-ceiling-breaker.** (bd `uqw`, `jdh`)
2. **Diffusion (SCDM-style) → note, don't chase for a fusion win here.** Its success = evidence of redundancy. Worth
   revisiting only on a task where fNIRS has *independent* signal (motor execution, 0-vs-load).
3. **Paired physics generator (bd `si7`) → the interpretable analog of SCDM.** EEG forward (SEREEGA/MNE lead field)
   + fNIRS forward (HRF), one shared latent; used to generate (synthesis) *and* infer (fusion). Links epic `vka`.
4. **GAN → skip.** Dated.

## Sources
- SCDM (EEG→fNIRS cross-modal diffusion): https://arxiv.org/abs/2407.04736
- 2026 High-Fidelity 3D fNIRS simulator: https://arxiv.org/abs/2605.30552
- 2024 fNIRS synthetic-data precursor: https://arxiv.org/abs/2405.11242
- EEG diffusion (MI): https://arxiv.org/html/2510.17832v1 · improved-DDPM: https://pubmed.ncbi.nlm.nih.gov/39693767/
- ERP conditional diffusion: https://arxiv.org/pdf/2403.18486 · WGAN-GP vs diffusion: https://openreview.net/pdf?id=r0216Av8tP
- Diffusion-for-brain-imaging survey (2026): https://link.springer.com/article/10.1186/s40708-026-00301-5
- GAN baseline (dated): Ye et al. 2021, Neurophotonics 8(2):025002 (PMC8362663)
