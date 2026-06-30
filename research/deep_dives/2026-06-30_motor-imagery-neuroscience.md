# Motor imagery, ERD, and CSP — the neuroscience + signal-processing behind mindscape

> Deep-dive · 2026-06-30 · the cited raw material behind the `learning/` lessons (01, 02). Grounds every
> domain claim the project makes about *why* motor-imagery EEG is decodable and why it's hard. Web-researched,
> primary/review sources cited inline. Read-only.

---

## 1. Motor imagery vs execution — the functional-equivalence hypothesis
**Claim:** imagining a movement engages largely the same motor network as performing it.

- Motor imagery (MI) and motor execution (ME) share overlapping regions: **primary motor cortex (M1),
  premotor cortex (PMC), supplementary motor area (SMA), cerebellum, basal ganglia, posterior parietal
  cortex**. This is the **functional-equivalence hypothesis** — MI recruits motor mechanisms similar to real
  action. [MI/ME narrative review (2024)](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12550537/),
  [Functional equivalence EEG/ERP study](https://www.frontiersin.org/journals/human-neuroscience/articles/10.3389/fnhum.2016.00467/full)
- **The key difference — output inhibition:** during MI, **the SMA acts to *suppress* M1** so the imagined
  movement does not become an actual muscle contraction. So M1 is engaged but its corticospinal output is
  gated. [SMA suppresses M1 (fMRI + dynamic causal modeling)](https://www.researchgate.net/publication/5616508_The_suppressive_influence_of_SMA_on_M1_in_motor_imagery_revealed_by_fMRI_and_dynamic_causal_modeling)
- **Mental chronometry** (imagined and real movements take ~equal time) supports shared planning; parietal
  lesions disrupt accurate imagined timing. [musculoskeletalkey summary](https://musculoskeletalkey.com/brain-activity-during-motor-imagery/)

**Consequence for us:** MI produces a motor-cortical signature *similar in topography to execution but
weaker* (M1 engaged at lower intensity, output inhibited, no sensory feedback) — which is exactly why MI
decoding accuracy is lower than decoding overt movement, and why MI is usable as a BCI for people who cannot
move.

## 2. ERD/ERS — the decodable EEG signature
**Claim:** the feature a motor-imagery decoder reads is a localized drop in mu/beta power.

- **Sensorimotor rhythms (SMR):** oscillations over sensorimotor cortex in **mu (8–12 Hz)**, **beta (≈13–30 Hz)**,
  and gamma. The rolandic mu ("central alpha") is recorded over central electrodes. [Sensorimotor Rhythm overview, ScienceDirect](https://www.sciencedirect.com/topics/medicine-and-dentistry/sensorimotor-rhythm)
- **ERD = a reduction in oscillatory power** related to a sensorimotor event; **ERS = a power increase**
  (rebound). Foundational framework: **Pfurtscheller & Lopes da Silva (1999)**, *Event-related EEG/MEG
  synchronization and desynchronization: basic principles*, Clin. Neurophysiol.
- **Contralateral + topographically like execution:** "Imagination of right and left hand movements results
  in desynchronization of mu and beta rhythms over the **contralateral** hand area, **very similar in
  topography to planning and execution** of real movements." [ERD/ERS review (Frontiers, 2022)](https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2022.1045715/full)
- **Mechanism:** ERD (localized power attenuation) reflects **increased excitability** of the underlying
  cortical neurons during processing — i.e. the population stops idling-in-synchrony and engages. Beta ERD
  spans planning + execution; beta ERS rebounds above baseline after movement ends. [Neupsy Key](https://neupsykey.com/eeg-event-related-desynchronization-erd-and-event-related-synchronization-ers/)

**Consequence for us:** the class-discriminating information is **band power (mu/beta) localized over
contralateral motor cortex (C3 right-hand, C4 left-hand, Cz feet)**, time-locked to the imagery — exactly
what `neuroviz` visualizes and what the decoders quantify.

## 3. Why scalp EEG needs spatial filtering
- An electrode sums the extracellular field of **many synchronized cortical pyramidal neurons**; the field
  spreads through tissue (**volume conduction**), so each electrode records a **blurred mixture** of sources,
  and motor signal is low-SNR amid EOG/EMG/line-noise artifacts. (Standard EEG biophysics.)
- Therefore the discriminative features are **spatial combinations** of channels that unmix one cortical
  source — a spatial filter.

## 4. CSP — Common Spatial Patterns
**Claim:** CSP finds spatial filters that maximize the band-power difference between two classes.

- CSP "derives optimal spatial filters by solving a **generalized eigenvalue decomposition** of the class
  covariance matrices, **maximizing variance for one class while minimizing it for the other**." For
  band-limited signals **variance = band power**, so this maximizes the ERD contrast. [Variance-preserving
  CSP (Frontiers 2023)](https://www.frontiersin.org/journals/human-neuroscience/articles/10.3389/fnhum.2023.1243750/full),
  [CSP overview](https://www.emergentmind.com/topics/common-spatial-patterns-csp)
- The problem: `Σ⁺ wᵢ = λᵢ Σ⁻ wᵢ` (Σ± = class covariances; λ = the between-class variance ratio). Top/bottom
  eigenvectors are the discriminative spatial filters; project → log-variance → features → a linear classifier.
- CSP is "**the most popular technique** for extracting EEG features in motor-imagery BCI." [emergentmind](https://www.emergentmind.com/topics/common-spatial-patterns-csp)

**Consequence for us:** `baselines/csp_lda.py` (CSP→LDA) is a near-parameter-free pipeline that *encodes the
neuroscience by hand* (band power + contralateral spatial pattern) and sets our **0.598 within-subject**
baseline; the deep nets (EEGNet/ATCNet) learn the same kind of spatial-spectral features end-to-end.

## 5. Why cross-subject transfer collapses (the biology, not the code)
The ERD signature is **idiosyncratic**: peak frequency, exact scalp locus (cortical folding), and SNR vary
per person; skull/electrode differences add more between-subject variance. A decoder tuned to one subject's
spatial-spectral pattern is mis-tuned for another — the *neural* reason our cross-subject accuracy drops
**0.598 → 0.382**, and why real BCIs calibrate per user. (Synthesis of the ERD reviews above + the
cross-subject DL benchmarks in `2026-06-30_2a_sota_recipe.md`.)

## Sources
- Pfurtscheller & Lopes da Silva 1999, *Event-related EEG/MEG synchronization and desynchronization*, Clin. Neurophysiol. (foundational).
- MI/ME EEG signatures narrative review — https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12550537/
- Functional equivalence (EEG/ERP) — https://www.frontiersin.org/journals/human-neuroscience/articles/10.3389/fnhum.2016.00467/full
- SMA suppresses M1 (fMRI/DCM) — https://www.researchgate.net/publication/5616508
- ERD/ERS beta review (2022) — https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2022.1045715/full
- Sensorimotor rhythm overview — https://www.sciencedirect.com/topics/medicine-and-dentistry/sensorimotor-rhythm
- ERD/ERS principles — https://neupsykey.com/eeg-event-related-desynchronization-erd-and-event-related-synchronization-ers/
- CSP (variance-preserving) — https://www.frontiersin.org/journals/human-neuroscience/articles/10.3389/fnhum.2023.1243750/full
- CSP overview — https://www.emergentmind.com/topics/common-spatial-patterns-csp
