# EEG→image / EEG→text decoding — the Stage-3 task decision + dataset landscape

**Date:** 2026-07-07 · **Why:** picking the harder task after fusion. Decided **EEG→image (perception) next**,
**EEG→text (Chisco) as the follow-on frontier-probe**. This records the datasets, the load-bearing principle,
the repository/API method, and why **no synthetic data is needed**.

---

## 1. The load-bearing principle — exogenous vs endogenous
Decoding difficulty scales with how much the signal is **driven by a known stimulus** vs **self-generated**:
```
perceive (read / view)   → exogenous, strong, TIME-LOCKED, stereotyped, averageable   ← SOLVABLE
overt produce (speak)    → real motor output to lock onto                              ← works, EMG-confounded
imagined / inner ("telepathy") → endogenous, weak, un-timed, idiosyncratic, no feedback ← FRONTIER / mostly null non-invasively
```
- **Perception decodes** because an external, known stimulus drives a strong, repeatable, time-locked response you can average over trials.
- **Imagery mostly nulls non-invasively** — faint, no onset to lock/average, idiosyncratic (aphantasia exists), no sensory-feedback stabilization. The impressive imagined-speech/thought results are **invasive (ECoG, high SNR on cortex)** or the **EEG-to-text LLM-leakage illusion** (the language model reconstructs from priors; the EEG barely contributes). Evidence: 175-h single-subject EEG speech gets top-1 48%, but **2.5% at the typical ~10 h** ([2407.07595](https://arxiv.org/abs/2407.07595)).
- **Perceptual reinstatement is a REAL partial lever (not a solve).** Imagery reactivates *part* of the perceptual circuit (Hebbian recall) — so training on perceived/overt and transferring to imagined works *above chance* (combining overt+imagined training +3-5%, [PMC12245923](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12245923/)). But the overlap is **attenuated + higher-order-only** (missing the strong bottom-up sensory drive), and the weak/un-timed/idiosyncratic signal problems remain. It **lifts imagined off the floor, doesn't reach perception's level.**

**Consequence:** decode where the signal is *driven* (image viewing, text reading); treat imagined/telepathy as an honest frontier-probe, not a deliverable. This is also *why* **EEG→image works** (perceiving images) and **EEG→imagined-text doesn't**.

## 2. EEG→image datasets (the next task — perception, WORKS)
| dataset | source | GB | subj | notes |
|---|---|---|---|---|
| **THINGS-EEG1** | OpenNeuro **`ds003825`** | 44 | **50** | Grootswagers — 22,248 images. Big **cross-subject** set. |
| **THINGS-EEG2** | OSF **`osf.io/3jk45`** (raw) / `3eayd` (preproc) | ~10 | 10 | Gifford (NeuroImage 2022) — 16,740 images. **The reconstruction benchmark.** Deep per-subject. |
| **Alljoined-1.6M** | HuggingFace / [arXiv 2404.05553](https://arxiv.org/html/2404.05553v1) | big | ~8 | affordable-BCI, **1.6M** EEG-image trials. |
| **EIT-1M** | [arXiv 2407.01884](https://arxiv.org/pdf/2407.01884) | — | — | **1M EEG-image-TEXT** pairs (doubles for the text stage). |

Secondary OpenNeuro object-recognition EEG (more subjects for cross-subject robustness): `ds007162` (34s), `ds003885`/`ds003887` (24s, object representation), `ds005363` (43s, object/aging), `ds004252` (rotation-tolerant), `ds005648` (object-space), `ds005087` (hemifield-object), `ds004018` (200 objects RSVP), `ds005106` (200 objects, infants). Category variants: food (`ds007012` 117s, `ds008092` 77s), faces (`ds003645` MEEG, `ds007096` N170).

**The method that works** (2023-2026): EEG → CLIP-style embedding → **diffusion** → reconstruct the seen image (recognizable). Here **generation earns its place honestly** — diffusion is the *reconstruction head of the decoder*, NOT laundered training data (avoids the identity trap). See EEG→image reconstruction via diffusion, e.g. [2403.07721](https://arxiv.org/html/2403.07721v7).

## 3. EEG→text datasets (the follow-on — reading solvable, imagined = trap-watch)
- **Chisco** — OpenNeuro **`ds005170`** (= the same "Chisco" from GitHub; grab via OpenNeuro). Chinese Imagined Speech Corpus, 5 subj, ~900 min/subj. **Paired design per trial: READ the sentence (5 s EEG, perceived) → IMAGINE speaking it (3.3 s EEG).** Goal = **sentence semantic reconstruction** (first attempt), i.e. EEG→text. Follow-on: [Assembling the Mind's Mosaic (2601.20447)](https://arxiv.org/pdf/2601.20447).
- **Chisco 2.0** — upcoming (sub-06/07, new paradigm); details not public yet.
- **ZuCo** (EEG-while-reading + eye-tracking + text, OSF) — the classic, but the "EEG-to-text with LLM" line is **largely debunked (teacher-forcing/leakage)**.
- **Honest plan:** decode the **reading (perceived) phase** where signal is real; use the **read↔imagine pairing to *bootstrap* the imagined phase** via reinstatement; treat any "sentence reconstruction" claim with **ZuCo-level skepticism** (is it EEG, or the LLM?).

## 4. Repositories + the reusable search method
- **OpenNeuro** = primary (BIDS, versioned, API-searchable). **OSF** = THINGS-EEG2, ZuCo. **HuggingFace** = Alljoined. **MOABB = NOT useful here** — it's BCI *paradigms* (MI / P300 / SSVEP), not natural-image viewing.
- **Download by source:** OpenNeuro → `openneuro-py download --dataset dsXXXXXX` (or `aws s3 sync --no-sign-request s3://openneuro.org/dsXXXXXX`); OSF → `osf -p <id> clone`; Zenodo → `zenodo_get`; GitHub → `git clone`. (`openneuro-py` is a *downloader*, not a search tool.)
- **OpenNeuro GraphQL search (record this — it's the reusable "CLI search"):** endpoint `https://openneuro.org/crn/graphql`. `datasets(modality: "EEG", first: 100, after: <cursor>)` paginated via `pageInfo{endCursor hasNextPage}`; node → `latestSnapshot { size description{Name} summary{subjects} }`. Filter names client-side. (`modality` is a **String** — quote it. `search(q:)` returned null; use `datasets(modality:)` or `advancedSearch(query:)`.) One paginated sweep = 442 EEG datasets, structured, sized — far better than web-snippet fishing.

## 5. No synthetic data needed
Real EEG→image data is **abundant** (THINGS-EEG1 50 subj + THINGS-EEG2 10 subj + Alljoined 1.6M + a dozen object sets). The reconstruction field trains on THINGS-EEG2 directly, no sim. **So the physics-sim / diffusion-generation thread stays parked as the *fNIRS-extraction validator* (`bd 7jn`) — a different problem.** Image decoding uses real data; identity-trap avoided by construction.

## 6. The pick + the honest-eval contribution
**Primary: THINGS-EEG2 (OSF) + THINGS-EEG1 (`ds003825`).** They **share the THINGS image set** → a natural **cross-dataset** test (train on one, eval on the other's subjects). The EEG→image field **over-reports** (within-subject / leaky retrieval metrics); our contribution is the **honest cross-subject generalization gap** — same playbook as MI (0.706→0.357). Reconstruction is the demo; the measured cross-subject drop is the point.

## Sources
- THINGS-EEG1 (ds003825): https://openneuro.org/datasets/ds003825 · THINGS-EEG2 (OSF): https://osf.io/3jk45/
- Alljoined: https://arxiv.org/html/2404.05553v1 · EIT-1M: https://arxiv.org/pdf/2407.01884
- EEG→image diffusion reconstruction: https://arxiv.org/html/2403.07721v7
- Chisco (ds005170): https://www.nature.com/articles/s41597-024-04114-1 · semantic-intent follow-on: https://arxiv.org/pdf/2601.20447
- 175-h EEG speech scaling (imagined nulls at scale): https://arxiv.org/abs/2407.07595 · overt→imagined transfer: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12245923/
- honest cross-subject vowel benchmark (ethos match): https://arxiv.org/pdf/2605.00865
- OpenNeuro GraphQL: https://openneuro.org/crn/graphql
