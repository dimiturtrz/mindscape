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

---

## 2026-07-07 (extension) — dataset expansion + shared-stimulus map

**Why:** find what the first pass MISSED, verify THINGS-MEG, defuse the Spampinato trap, and — load-bearing for our cross-dataset contribution — nail down exactly which datasets share the SAME stimulus images. Method: fresh OpenNeuro GraphQL sweep (442 EEG datasets, 125 visual-keyword hits) + HF/Zenodo/arXiv. Specs below are cited; where a spec came from a Haiku scout and I could not reach the primary file, it's flagged `[unverified]`.

### A. THINGS-MEG — verified, and it IS the same THINGS image set
- **OpenNeuro `ds004212`** (confirmed). Hebart et al. 2023, *eLife* 12:e82580 (THINGS-data). **4 subjects × 12 sessions**, **272-channel CTF MEG** + eye-tracking, 500 ms stim + ~1000 ms fixation (SOA 1500±200 ms), passive viewing. **22,248 training images (1,854 concepts × 12 exemplars) + 200-image test set repeated per session.**
- **Same stimulus database — CONFIRMED.** THINGS-MEG, THINGS-EEG1, THINGS-EEG2 all draw from the **THINGS database: 1,854 object concepts, 26,107 naturalistic images** (elifesciences.org/articles/82580; things-initiative.org). MEG and EEG1 both cover **all 1,854 concepts × 12 exemplars**; EEG2 covers a **1,654-concept train subset + 200-concept test**. So THINGS-MEG shares the concept set (and, for the 200-concept test, the exemplar images) with both EEG sets.
- **Correction to a scout artifact:** the "1,654 concepts × 10 img × 4 reps + 200 × 1 × 80 reps" split is **THINGS-EEG2's** design (Gifford 2022, 63-ch, 100 ms SOA), NOT THINGS-EEG1's. **THINGS-EEG1** (Grootswagers 2022, *Sci Data* 9:3, ds003825, 64-ch, 1000 Hz, 5–10 Hz RSVP) shows all **22,248** images (1,854 × 12), broad-and-shallow across 50 subjects. EEG2 is deep-and-narrow (10 subjects, heavy repetition) — the reconstruction benchmark.

### B. NET-NEW datasets (not in the original list)

**Tier-1 — naturalistic object/scene EEG/MEG viewing (directly usable):**
| dataset | id / URL | subj | images | ch | rate | license | paradigm |
|---|---|---|---|---|---|---|---|
| **NOD-EEG** | OpenNeuro `ds005811` | 19 | **~57,000 ImageNet** (1000 cat × ~57) | 64/66 (mixed) | 500/1000 Hz | CC-BY-4.0 | viewing + animacy judgment |
| **NOD-MEG** | OpenNeuro `ds005810` | 31 | same 57k ImageNet | 275 (CTF) | 1200→250 Hz | CC-BY-4.0 | viewing + animacy judgment |
| **NOD-fMRI** | OpenNeuro `ds004496` | 30 | same 57k ImageNet | — | — | CC-BY-4.0 | (fMRI arm of NOD) |
| **EEG-ImageNet (2024)** | GitHub `Promise-Z5Q2SQ/EEG-ImageNet-Dataset` | 16 | 4,000 (ImageNet-21k, 80 cat) | 62 | 1000 Hz | open | RSVP 500 ms; multi-granularity labels |
| **MSS natural-image** | Nature SD `s41597-025-04843-x` | 32 | 10,000 (PASCAL+ImageNet) | 122 | 1000 Hz | CC-BY | dual RSVP (5 Hz stream + 1 s/img) |
| **ds002814** | OpenNeuro `ds002814` | 21 | 125 (5 cat: animals/chairs/faces/fruits/vehicles) | `[unverified]` | `[unverified]` | `[unverified]` | 1-back, joint fMRI+EEG |
| **ds004995 (food time-course)** | OpenNeuro `ds004995` | 20 | 314 (154 food/160 non-food) | 128 | 1000 Hz | CC0 | passive viewing |
| **ds005586 (occluded scenes)** | OpenNeuro `ds005586` | 23 | game-board scenes, 0–32 objects | 60 | `[unverified]` | `[unverified]` | passive viewing of object-count/occlusion |

Key find: **NOD** (Zhang et al. 2025, *Sci Data* 12:857, PMC12102372) = a 3-modality suite (EEG+MEG+fMRI) over ~57k ImageNet images — the largest naturalistic-object EEG set after THINGS, and the cleanest same-image cross-modality set (see map below). The **2024 "EEG-ImageNet"** (Promise) is a *distinct, new* dataset — NOT Spampinato's 2017 set (§C).

**Tier-2 — video / imagery / adjacent (context, weaker fit to static EEG→image):**
- **EEG2Video / SEED-DV** (NeurIPS 2024, GitHub `XuanhaoLiu/EEG2Video`): 20 subj, 1,400 naturalistic video clips (40 concepts), ~62-ch. Video, not static images — train/test domain shift.
- **CineBrain** (arXiv 2503.06940): 6 subj, simultaneous **EEG+fMRI** on ~9 h audiovisual narrative (Big Bang Theory). Fusion, small-N.
- **ds004306** (OpenNeuro, 12 subj, 124-ch, 1024 Hz, CC-BY-4.0): semantic **perception vs imagination** of 3 concepts × 3 modalities — imagery-transfer probe.
- **ds005274 = "UV_EEG" / Nature SD visual-imagery BCI** (`s41597-025-06512-5`, 22 subj, 32-ch, 1000 Hz, CC-BY-NC-ND): 10 imagined objects (3 categories). **One dataset, two identifiers** — dedup.
- **Dream2Image** (arXiv 2510.06252, HF `opsecsystems/Dream2Image`, 38 subj): dream EEG → generated images. Endogenous — frontier-probe only.
- **EgoBrain** (ICLR 2026, HF `ut-vision/EgoBrain`, 40 subj, EEG+IMU+egocentric video, CC-BY-NC): action understanding, not image reconstruction.
- **ds004357 "Features-EEG"** (16 subj, 128-ch, CC-BY): 256 gabor gratings — low-level features, NOT naturalistic objects. Noted for completeness, low relevance.

**Dedup / already-known (do not double-count):** the "Infant 200-objects" Nature SD paper (`s41597-025-04744-z`, Grootswagers, 42 infants) **= `ds005106`** already in the list. `ds007964` (rsvp-flagged) is actually an *emotional word-association* task, not visual. **No Alljoined-2/3 exists.** BraVL and NSD are fMRI. Reading-EEG sets surfaced (`ds004952` ChineseEEG, `ds005383` TMNRED, `ds007753` BCCWJ-EEG) belong to the EEG→text stage, not here.

### C. The Spampinato/Kavasidis 2017 "brain2image" trap — leakage, not a target
**The dataset** (Spampinato, Palazzo, Kavasidis et al., CVPR 2017, arXiv 1609.00344; GitHub `perceivelab/eeg_visual_classification`): 6–7 subjects, 32-channel EEG @ 1 kHz, **40 ImageNet classes × 50 images = 2,000 stimuli**, reported **~40% 40-way accuracy**. **The fatal design:** images were shown **blocked by class** — "all 50 stimuli in a block being images of the same class," 0.5 s each, blocks separated by 10 s blanking. So each class = one contiguous ~25 s recording window.

**The rebuttal** (Li, Johansen, … Siskind, *IEEE TPAMI* 2020, arXiv 1812.07697, orig. titled "Training on the test set? An analysis of Spampinato et al."): because trials are blocked, slow non-stimulus EEG drift (DC/low-frequency cortical state) is **constant within a block and correlated between that block's train and test trials** — "classification of arbitrary brain states based on block-level temporal correlations that are known to exist in all EEG data, rather than stimulus-related activity." When they **interleave classes (rapid-event design)**, accuracy **collapses to ~chance** (2.5% for 40-way); a **random-codebook classifier matches or beats** the EEG one. They name Spampinato et al. explicitly and prescribe **randomized/interleaved trials**. Palazzo & Spampinato's 2020 reply concedes the slow-drift inflation exists but argues it's overstated.

**Verdict — do NOT use as a decoding target or benchmark.** The block design makes it a **temporal-leakage trap**: any classifier can score ~40% by learning *which 25-second window* a trial came from, with zero visual-stimulus information. Reported accuracy is an artifact of train/test sharing a block, not evidence of image decoding. It is the canonical cautionary tale, not a dataset. **Design rule we inherit:** only trust EEG→image sets with **interleaved/randomized trials** (THINGS-EEG1/2, THINGS-MEG, NOD all use randomized RSVP/viewing — clean) and always verify our own splits never let train and test share a temporal block.

### D. Shared-stimulus map (LOAD-BEARING for cross-dataset train-on-one/test-on-other)
Two clean shared-stimulus groups enable cross-dataset / cross-modality transfer:

**GROUP A — THINGS family** (shared *concept set*; 200-concept test = shared *exemplar images*):
- `ds003825` **THINGS-EEG1** — all 1,854 concepts × 12 exemplars (22,248 imgs), 50 subj, EEG.
- `osf.io/3jk45` **THINGS-EEG2** — 1,654 train concepts × 10 + 200 test concepts × 1 (16,740 imgs), 10 subj, EEG.
- `ds004212` **THINGS-MEG** — all 1,854 concepts × 12 + 200 test, 4 subj, MEG.
- **Overlap:** all three draw from the SAME THINGS DB (1,854 concepts / 26,107 images). EEG1 ∩ MEG = **full 1,854-concept overlap** (both use 12 exemplars/concept — likely the same exemplar files, same pool; exact file-identity `[unverified]`). EEG2's 1,654+200 concepts ⊂ 1,854. The **200-concept, single-exemplar test set is the cleanest bridge**: documented as *identical images* across EEG2 and MEG; EEG1 also holds a 200-concept test (same-200 identity `[unverified]`). → supports **cross-dataset (EEG1↔EEG2)** and **cross-modality (EEG↔MEG)** zero-shot retrieval on shared THINGS stimuli. This is the natural home for our honest cross-subject/cross-dataset generalization-gap contribution.

**GROUP B — NOD family** (shared *exact image files* — strongest form of sharing):
- `ds005811` **NOD-EEG** (19 subj) · `ds005810` **NOD-MEG** (31 subj) · `ds004496` **NOD-fMRI** (30 subj).
- **Overlap:** the paper states the ~57,000 ImageNet stimuli "used for MEG and EEG are **identical**" (same `synsetID_imageID.JPEG` files). → the cleanest same-image cross-modality set available, exact-file-matched (better than THINGS for image-level pairing).

**NOT a clean shared group (ImageNet-derived but different subsets — no direct image-level transfer):** Spampinato-2017 (40 cls/2k imgs, and a leakage trap anyway), EEG-ImageNet-2024 (4k imgs, ImageNet-21k), NOD (ILSVRC2012 57k) — all ImageNet but disjoint subsets. **Possible minor group:** `ds004018` and `ds005106` are both "200-object" RSVP sets from the Carlson/Grootswagers line — may share the 200-object stimulus set `[unverified]`; worth a file-level check if a second small object set is wanted. Food sets (`ds007012`, `ds008092`, `ds004995`) use different image pools — not confirmed shared.

**Bottom line:** our cross-dataset plan stands and now has a second, stronger option. **THINGS group** = concept-level shared, huge, the field's benchmark (EEG1↔EEG2↔MEG). **NOD group** = exact-image shared across EEG/MEG/fMRI — use it for a clean cross-*modality* generalization test if we want image-level (not just concept-level) pairing.

### Extension sources
- THINGS-MEG (ds004212) / THINGS-data: https://elifesciences.org/articles/82580 · https://openneuro.org/datasets/ds004212
- THINGS-EEG1 (ds003825, Grootswagers 2022): https://www.nature.com/articles/s41597-021-01102-7 · THINGS-EEG2 (Gifford 2022): https://doi.org/10.1016/j.neuroimage.2022.119754
- NOD (Zhang 2025, Sci Data 12:857): https://pmc.ncbi.nlm.nih.gov/articles/PMC12102372/ · NOD-EEG https://openneuro.org/datasets/ds005811 · NOD-MEG https://openneuro.org/datasets/ds005810
- EEG-ImageNet 2024: https://arxiv.org/html/2406.07151v1 · https://github.com/Promise-Z5Q2SQ/EEG-ImageNet-Dataset · MSS 32-subj: https://www.nature.com/articles/s41597-025-04843-x
- ds004995 food: https://openneuro.org/datasets/ds004995 · ds002814: https://www.biorxiv.org/content/10.1101/2022.05.12.491595v1 · ds004306: https://pmc.ncbi.nlm.nih.gov/articles/PMC10272218/ · ds005274 imagery: https://www.nature.com/articles/s41597-025-06512-5
- EEG2Video/SEED-DV: https://github.com/XuanhaoLiu/EEG2Video · CineBrain: https://arxiv.org/abs/2503.06940 · Dream2Image: https://arxiv.org/abs/2510.06252 · EgoBrain: https://huggingface.co/datasets/ut-vision/EgoBrain
- Spampinato 2017: https://arxiv.org/abs/1609.00344 · Li et al. 2020 block-design rebuttal: https://arxiv.org/abs/1812.07697
