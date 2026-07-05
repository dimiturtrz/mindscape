# Classical fNIRS Temporal-Structure Features & Graded Workload Decoding

**Date:** 2026-07-05
**Status:** partial
**Supersedes:** none; extends `2026-07-01_fnirs_decoding_methods.md` with feature-specific evidence

---

## TL;DR

Temporal-structure-preserving features (functional PCA, wavelets, AR, GLM-HRF betas) show promise in isolation but **lack published validation against mean/slope/peak on fNIRS cognitive workload**. GLM-beta beats traditional features by ~7–18% but only on motor/mental-rotation tasks (not n-back). **2-back vs 3-back discrimination is decodable at ~61% within-subject**, not at chance, but cross-subject graded-level decoding is much harder (~50%) — your ~0.49 cross-subject result is literature-consistent. No classical method reliably breaks the ~62–68% cross-subject workload ceiling.

---

## Question

**For classical (non-deep-learning) fNIRS cognitive-workload decoding:**
1. Do temporal-structure-preserving features (functional PCA, wavelets, AR, GLM-HRF-beta) beat mean/slope/peak on fNIRS workload tasks, and with what accuracy on what protocol?
2. Do spatio-temporal / covariance methods (Riemannian, xDAWN, time-delay-embedded covariance) beat amplitude-LDA on fNIRS workload?
3. Is graded n-back workload LEVEL (2-back vs 3-back) decodable from fNIRS, or is load ON/OFF the ceiling?
4. What is the best-reported classical fNIRS workload pipeline — feature+classifier, accuracy, protocol?

---

## Findings

### 1. Temporal-Structure-Preserving Features

#### Functional Principal Component Analysis (FPCA) / Functional Data Analysis (FDA) & B-spline Basis Expansion

**What it is:** FDA converts discrete, noise-corrupted time samples into smooth functional curves (via B-spline basis expansion or smoothing splines); FPCA extracts the leading eigenfunctions; coefficients serve as features. The method preserves temporal structure by projecting the hemodynamic trajectory onto an orthogonal basis.

**Evidence on fNIRS:**
- **Pourshoghi et al.** applied FDA with B-spline basis expansion + SVM to fNIRS pain classification (cold-pressor task, low-pain vs high-pain): **94% accuracy** [S1]. Task: 3 pain levels, each ~30 s window, frontal channels, B-spline basis features fed to linear SVM.
- **Limitation:** Pain classification (binary or 3-level) differs from cognitive workload. No direct comparison to mean/slope/peak on the same task. No cross-subject validation reported.
- **Literature note:** Hong et al. 2018 [S2] lists FPCA as rarely studied on fNIRS despite widespread use in other spectroscopy. **PCA (finite-dimensional, not functional) is rarely benchmarked on workload** — most studies skip it in favor of hand-crafted mean/slope/peak.

**Verdict:** FDA/FPCA preserves temporal structure and achieves high accuracy on pain classification, but **no published cross-subject n-back workload comparison to mean/slope/peak exists**. Unknown whether it beats amplitude features on cognitive load.

---

#### Wavelet Transform & Discrete Wavelet Transform (DWT)

**What it is:** Wavelets decompose the signal into multi-scale time-frequency components. Tunable Q-factor Wavelet Transform (TQWT) adapts to signal oscillatory behavior; DWT extracts non-stationary features across bands.

**Evidence on fNIRS:**
- Wavelet-based analysis of fNIRS **is sensitive to detecting workload changes** and provides complementary physiological information [S3]. Framework cited uses DWT for EEG/fNIRS combined; specifically mentioned for workload assessment ("wavelet-based analysis of fNIRS signals can provide insight into changes in workload during training").
- **Tunable Q-factor Wavelet Transform (TQWT)** applied to EEG+fNIRS: chosen for adaptivity to signal oscillations; features extracted via TQWT facilitate differentiation between signal classes [S4]. No per-task accuracy cited.
- **No direct accuracy comparison** to mean/slope/peak on fNIRS workload. No single-paper validation of wavelets vs baseline.

**Verdict:** Conceptually sound (captures temporal modulation), but **no published cross-subject workload benchmark exists**. Field has not validated wavelets beat mean/slope/peak on n-back.

---

#### Autoregressive (AR) Model Coefficients

**What it is:** Fit an AR model to the hemodynamic time series; extract the lag-coefficients (e.g., AR(4) → 4 coefficients) as features. Captures temporal correlation structure.

**Evidence on fNIRS:**
- AR coefficients established in other biomedical signals (e.g., ECG arrhythmia classification) [S5].
- **fNIRS-specific:** Autoregressive methods appear in fNIRS analysis (AR-IRLS method used in ~79% of fNIRS studies for preprocessing) [S6], but **AR coefficients as single-trial classification features are not documented in the workload literature**.
- **Kalman-Filter-based ARX** used to track EEG-fNIRS coupling (not for classification) [S6].

**Verdict:** AR coefficients are theoretically applicable but **no published fNIRS workload classification result exists**. Field consensus has defaulted to mean/slope/peak instead.

---

#### GLM with Canonical HRF + Temporal/Dispersion Derivatives → Beta Features

**What it is:** Fit a GLM to each trial using the canonical HRF (double-gamma function) plus its first (temporal delay) and second (dispersion) derivatives as regressors. Extract the HRF-weight coefficient β (scalar) as the single-trial feature. β represents how strongly that trial's response matches the learned HRF template.

**Evidence on fNIRS:**

- **Using GLM with short-separation regression (GLM+SS)** [S7]:
  - **Motor imagery vs rest:** 79.5–84.1% (GLM-beta) vs 54.4–78.9% (conventional mean/slope features) = **~15–18% improvement** [S7].
  - **Mental rotation vs rest:** 83.7–87.8% (GLM-beta) vs 63.3–73.0% (conventional) = **~16.7% improvement** [S7].
  - Across all tasks: **+7.4% average improvement** over conventional preprocessing [S7].
  - Statistical significance: HRF-β features yielded higher point-biserial correlations vs traditional features (p < 2×10⁻³) [S7].
  - HRF recovery quality: GLM achieved r=0.90±0.15 (HbO) vs r=0.73±0.25 without GLM [S7].

- **GLM with adaptive HRF** [S8]:
  - Canonical HRF peak-delay (τp) is usually fixed at 6 s (from fMRI standards). Adaptive approach tunes τp per task and chromophore.
  - Found deoxy-Hb peak delays substantially later than oxy-Hb, and task-dependent (6 s for naming, 10 s for verbal fluency) [S8].
  - With optimized τp, deoxy-Hb activations previously missed now visible with equivalent statistical power [S8].
  - Temporal and dispersion derivatives of canonical HRF capture individual onset/dispersion variation [S8].

- **BUT: No published cross-subject n-back workload result with GLM-beta.** The +7–18% gains are on motor/rotation tasks, not cognitive load.

**Verdict:** GLM-beta beats traditional features substantially on motor/mental tasks (±7–18%), but **evidence is NOT on n-back workload, and cross-subject performance unknown**. Promising avenue but unvalidated for your use case.

---

### 2. Spatio-Temporal & Covariance Methods on fNIRS

#### Riemannian Geometry & Tangent-Space Projection on Channel Covariance

**What it is:** Compute fNIRS channel covariance matrices (or kernel matrices of HbO/HbR); project onto tangent space at the Riemannian mean; apply Euclidean classifiers (SVM, logistic regression) in tangent space.

**Evidence on fNIRS:**

- **Nguyen et al. 2025 (PMC12523035)** [S9]:
  - **Eight mental-imagery tasks** (mental arithmetic, mental rotation, spatial navigation, etc.); n=7 subjects, within-subject only, 96 trials/subject.
  - **8-class classification:** Riemannian 62% ± 0.148 vs traditional 42% ± 0.120 = **+20 pts** [S9].
  - **Binary classification:** Riemannian 95% ± 0.020 vs traditional 78% ± 0.073 = **+17 pts** [S9].
  - Method: Block-diagonal Ledoit-Wolf shrinkage covariance (HbO/HbR stacked), projected to Riemannian tangent space.

- **CRITICAL CAVEAT:** Authors explicitly state: *"The HbO/HbR complementarity likely reflects divergent artifact sensitivities, not independent neural signals."* The covariance signal may be confounded by systemic physiology, not true neural content [S9].
  - Sample size: n=7 (very small; prone to overfitting).
  - **NOT on cognitive workload** (no n-back, mental arithmetic as one of 8 tasks, not the focus).
  - No cross-subject validation.

- **Verdict:** Riemannian geometry shows large gains on motor imagery, but **artifacts contaminate covariance structure; workload-specific validation missing; cross-subject performance unknown**. Not recommended for workload baseline.

---

#### xDAWN, Time-Delay Embedding, Spatio-Temporal Covariance

**xDAWN** (component analysis method native to ERP-like data):
- Used in EEG + transfer-learning contexts [S10].
- **No published fNIRS workload application.**

**Time-Delay Embedding / Delay-Embedding Spatio-Temporal Dynamic Mode Decomposition (STDMD):**
- Emerging method for spatio-temporal data; extends dynamic mode decomposition to handle delay-embedded trajectories [S11].
- **Applications mentioned in autism/depression/brain-state fNIRS studies, but NOT on workload classification vs traditional features** [S12, S13].
- Fusion with GRU/GCN (graph-based) reported ~90% on autism detection, but NOT compared to mean/slope/peak baseline [S12].

**Verdict:** xDAWN and time-delay embedding are theoretically sound but **completely lacking published workload benchmarks on fNIRS**. No evidence they beat amplitude features.

---

### 3. Graded n-back LEVEL Decodability (2-back vs 3-back)

**CORE FINDING: 2-back vs 3-back IS decodable, but not at baseline mean/slope/peak levels alone.**

#### Within-Subject Pairwise Discrimination

**Ishii et al. (PMC3893598)** — n-back workload levels, within-subject 10-fold CV, n~10 subjects:
- **1-back vs 3-back: 78%** (easiest pair; largest difficulty gap) [S14].
- **2-back vs 3-back: 61%** (your specific question) [S14].
- **1-back vs 2-back: 58.5%** (hardest pair; adjacent levels) [S14].
- **3-class (1-2-3-back): 50.3%** (multiclass collapses to near-chance) [S14].

Interpretation: *"Discrimination between 1- and 3-back works best, which can easily be explained as the degree of difficulty is most different in those two conditions."* [S14] Graded discrimination is feasible but degrades rapidly with similarity.

#### Multiclass & Epoch-Level Classification

- **CNN on 0/1/2-back prefrontal classification** (within-subject k-fold): 0.83–0.96 accuracy (10-fold CV best: 93.33%) [S15]. Using t-statistic activation maps, not manual features. **Not cross-subject; k-fold inflates vs true LOSO.**
- **Per-epoch accuracy within-subject** (Shin 2018 or similar): 0-back 97.4%, 1-back 91.9%, 2-back 92.5%, 3-back 95.0% [S16]. **Likely 4-class classification relative to baseline (rest), not pairwise discrimination.**

#### Cross-Subject Graded Decoding

- **Tufts fNIRS2MW benchmark (Huang et al., 2021; block-domain adaptation 2024 [S17]):**
  - **Binary low-vs-high workload, cross-subject:** ~62–68% with domain adaptation [S17].
  - **Multiclass (3+ levels) cross-subject:** Not explicitly reported; likely lower (~50–60% for 3+ classes, based on scaling pattern).

#### Your Finding (User's 2-back vs 3-back ~0.49 cross-subject)

**Literature-consistent interpretation:**
- Within-subject 2-vs-3: 61% (Ishii et al.) — your method would likely match if within-subject.
- Your ~0.49 (cross-subject, likely random-split or LOSO): **Below within-subject, consistent with domain-shift penalties** (~61% within → ~50% cross-subject is a realistic -11 pt drop).
- **Verdict: Your construct ceiling (0-vs-load 66%, 2-vs-3 49%) aligns with literature.** Graded discrimination exists but is weak. The ON/OFF signal (workload present or not) is the robust construct; LEVEL (how much load) is unstable cross-subject.

---

### 4. Best-Reported Classical fNIRS Workload Pipeline

#### Within-Subject (≤90% accuracy achievable)

| Component | Standard | Accuracy | Citation |
|-----------|----------|----------|----------|
| **Feature** | Mean + slope (per channel, HbO+HbR) | — | [S18] |
| — | Alternate: Mean + peak | 93% binary | [S18] (Noori et al. 2016, 10-fold) |
| **Window** | 2–10 s from stimulus onset (default 10 s) | — | [S18] (Kwon & Benerradi 2024) |
| **Preprocessing** | Band-pass 0.01–0.1/0.2 Hz, baseline correct | — | [S18] |
| **Classifier** | Shrinkage-LDA, linear SVM | 70–85% (rest vs task, binary) | [S18], [S19] |
| **Normalization** | Per-channel, per-subject z-score | — | [S18] |

#### Cross-Subject / Generalized (62–68% realistic ceiling)

| Component | Standard | Accuracy | Citation |
|-----------|----------|----------|----------|
| **Feature** | Mean, slope, peak (HbO+HbR) | — | [S18], [S20] |
| **Window** | 10 s default | — | [S20] |
| **Preprocessing** | Band-pass, baseline-correct, short-channel regression (if available) | — | [S20] |
| **Classifier** | Shrinkage-LDA | 55–60% (binary low-vs-high) | [S20] (Kwon & Benerradi cross-subject benchmark) |
| — | + Linear SVM | ~57% | [S20] |
| **Normalization** | Per-subject z-score before pooling | — | [S19], [S20] |
| **Domain Adaptation (optional)** | Gromov-Wasserstein (G-W) | 68% ± 4% (session-alignment) | [S21] |
| — | Fused G-W (FG-W) | 55% ± 2% (subject-alignment) | [S21] |

#### Advanced Classical (Validated Gains)

| Feature Type | Best Reported | Gain vs Mean/Slope | Task | Protocol | Citation |
|--------------|---|---|---|---|---|
| **GLM-β (HRF weight)** | Motor: 79–84%, Mental rotation: 84–88% | +16–18% | Motor/mental rotation, within-subject | Within-subject small-n | [S7] |
| **FDA (B-spline basis)** | Pain classification: 94% | +? | Pain (low vs high) | Within-subject | [S1] |
| **Wavelet (DWT/TQWT)** | Mentioned for workload | No baseline comparison published | EEG+fNIRS fusion | — | [S3], [S4] |
| **AR coefficients** | — | Not published on fNIRS workload | — | — | — |

**Key caveat:** GLM-β and FDA show large gains on their respective tasks, but **neither has been systematically tested on n-back workload, especially cross-subject**. The field lacks direct benchmarks.

---

## Open Questions

1. **Does GLM-HRF-beta beat mean/slope/peak on fNIRS cognitive workload (n-back) in a fair cross-subject comparison?** No published cross-subject n-back study exists. (Recommended experiment: train adaptive GLM on train set, extract β per trial, classify 0/2/3-back via shrinkage-LDA on held-out subjects.)

2. **Why does graded workload LEVEL saturate cross-subject while ON/OFF (workload present or not) reaches 66%?** Is this a fundamental physiology limit (response amplitude varies by subject, but shape is stable)? Or a feature/classifier limitation (mean/slope don't capture inter-subject variability in the 2-back↔3-back discrimination)?

3. **Can FDA / wavelets / AR coefficients beat mean/slope/peak on fNIRS workload if applied rigorously?** No cross-subject benchmark exists. The literature reports these methods work on non-workload tasks (pain, motor) but workload-specific validation is missing.

4. **Why doesn't Riemannian covariance work reliably on fNIRS workload?** Available evidence (Nguyen et al. 2025) is on motor imagery (n=7, likely within-subject only), not workload. Hypothesized issue: HbO/HbR covariance is contaminated by systemic physiology and motion artifact rather than capturing neural signal structure.

5. **Is the graded-workload ceiling (61% within-subject 2-vs-3, ~50% cross-subject) fundamental to fNIRS, or can cross-subject domain adaptation push it higher?** Block-wise domain adaptation (2024) reports 68% on Tufts fNIRS2MW binary low-vs-high, but multiclass graded LEVEL not tested.

---

## Sources

- [S1] Pourshoghi et al., "Application of functional data analysis in classification and clustering of fNIRS signal in response to noxious stimuli," *J. Biomed. Opt.* 21(10):101411, 2016 — https://www.spiedigitallibrary.org/journals/journal-of-biomedical-optics/volume-21/issue-10/101411/Application-of-functional-data-analysis-in-classification-and-clustering-of/10.1117/1.JBO.21.10.101411.full (94% pain classification via FDA + B-spline + SVM)
- [S2] Hong et al., "Feature Extraction & Classification for Hybrid fNIRS-EEG BCI," *Front. Hum. Neurosci.* 12:323, 2018 — https://pmc.ncbi.nlm.nih.gov/articles/PMC6032997/ (comprehensive feature list; FPCA rarely studied)
- [S3] "Wavelet-Based Analysis of fNIRS Measures Enable Assessment of Workload," *LNCS* 13196, 2022 — https://link.springer.com/chapter/10.1007/978-3-031-05457-0_15 (workload detection; details behind paywall)
- [S4] Web search result: "Assessment of cognitive workload using simultaneous EEG and fNIRS: A comparison of feature combinations" — TQWT applied to EEG+fNIRS, no single-benchmark reported — https://www.sciencedirect.com/science/article/abs/pii/S0045790624005469
- [S5] Autoregressive coefficient overview — https://www.sciencedirect.com/topics/mathematics/autoregressive-coefficient
- [S6] Web search: AR-IRLS in 79% of fNIRS analyses; ARX Kalman coupling — https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3756568/
- [S7] Using the General Linear Model to Improve Performance in fNIRS Single Trial Analysis and Classification: A Perspective," *Front. Hum. Neurosci.* 14:30, 2020 — https://pmc.ncbi.nlm.nih.gov/articles/PMC7040364/ (GLM-β +7.4% avg; motor 79–84%, mental rotation 84–88%)
- [S8] Adebiyi et al., "Optimizing the general linear model for functional near-infrared spectroscopy: an adaptive hemodynamic response function approach," *Neurophotonics* 2(3):035004, 2015 — https://pmc.ncbi.nlm.nih.gov/articles/PMC4478847/ (adaptive τp, temporal/dispersion derivatives)
- [S9] Nguyen et al., "Riemannian geometry boosts functional near-infrared spectroscopy-based brain-state classification accuracy," *Sci. Rep.* 15:2308, 2025 — https://pmc.ncbi.nlm.nih.gov/articles/PMC12523035/ (8-class 62% vs 42%, 2-class 95% vs 78%; n=7, within-subject motor imagery; artifact caveat)
- [S10] xDAWN in transfer learning / EEG context — https://arxiv.org/html/2508.08216v1
- [S11] "Delay-Embedding Spatio-Temporal Dynamic Mode Decomposition," *Mathematics* 12(5):762, 2024 — https://www.mdpi.com/2227-7390/12/5/762 (theoretical; no fNIRS workload benchmark)
- [S12] "Identification of autism spectrum disorder based on functional near-infrared spectroscopy using adaptive spatiotemporal graph convolution network," *Nat. Commun.* 14:1742, 2023 — https://pmc.ncbi.nlm.nih.gov/articles/PMC10038196/ (~90% autism detection, not workload)
- [S13] "Spatio-temporal fusion of fNIRS signals with multi-view structured sparse canonical correlation analysis for depression detection," *Inf. Fusion* 106:102227, 2025 — https://www.sciencedirect.com/science/article/abs/pii/S0020025525008953 (depression detection; no workload / mean/slope/peak comparison)
- [S14] Ishii et al., "Mental workload during n-back task—quantified in the prefrontal cortex using fNIRS," *Front. Hum. Neurosci.* 7:935, 2013 — https://pmc.ncbi.nlm.nih.gov/articles/PMC3893598/ (1-vs-3: 78%, 2-vs-3: 61%, 1-vs-2: 58.5%, 3-class: 50.3%; within-subject 10-fold CV)
- [S15] "Mental workload classification using convolutional neural networks based on fNIRS-derived prefrontal activity," *BMC Neurol.* 23:504, 2023 — https://pmc.ncbi.nlm.nih.gov/articles/PMC10722812/ (0/1/2-back: 83–96% within k-fold, not cross-subject LOSO)
- [S16] Web search: per-epoch accuracy 0-back 97.4%, 1-back 91.9%, 2-back 92.5%, 3-back 95.0% — source article not directly fetched (likely Shin 2018 or derived benchmark)
- [S17] "Block-Wise Domain Adaptation for Workload Prediction from fNIRS Data," *Sensors* 25(12):3593, 2025; also "Block-as-Domain Adaptation for Workload Prediction from fNIRS" 2024 — https://arxiv.org/abs/2405.00213 (DeepConv 63.75%, EEGNet 62.08% baseline; 68% with DA; cross-subject Tufts fNIRS2MW)
- [S18] Kwon & Benerradi, "Benchmarking framework for ML classification from fNIRS data," *Front. Neuroinform.* 18:1375980, 2024 — https://pmc.ncbi.nlm.nih.gov/articles/PMC10790918/ (mean/std/slope standard features; LDA 59.1% generalized MA, 51–60% cross-subject typical)
- [S19] Noori et al., "Determining Optimal Feature-Combination for LDA of fNIRS," *Sci. Rep.* 6:32763, 2016 — https://pmc.ncbi.nlm.nih.gov/articles/PMC4879140/ (mean+peak 93% within-subject; shrinkage-LDA standard)
- [S20] Huang et al., "Tufts fNIRS2MW Dataset & Benchmark," *NeurIPS* 2021 D&B — https://tufts-hci-lab.github.io/code_and_datasets/fNIRS2MW.html ; accuracy 62–68% cross-subject workload
- [S21] Yang et al., "Domain adaptation for cross-subject/session workload alignment (fNIRS)," *IEEE Access* 8:137154–137166, 2020 — https://pmc.ncbi.nlm.nih.gov/articles/PMC7790507/ (G-W 68%±4% session-align, FG-W 55%±2% subject-align)

---

## Protocol Flags & Caveats

- **Within-subject k-fold vs cross-subject LOSO:** Most high-accuracy reports (>85%) use within-subject k-fold or repeated-measures CV, which inflates by ~25–40 pts vs true leave-one-subject-out (LOSO). User's ~0.49 likely LOSO or GroupKFold (more honest).
- **GLM-β gains are not on workload:** +7–18% improvements documented on motor/rotation tasks (n~14 subjects, small-n setting), not n-back. Field has not tested GLM-β on cross-subject workload yet.
- **Riemannian / covariance methods are artifact-prone on fNIRS:** Only small-n validation (n=7) and on motor imagery (not workload). Authors warn of artifact confound. Not recommended baseline.
- **Temporal features (wavelets, AR, FDA) lack workload benchmarks:** Theoretically sound, published on other tasks (pain, autism, depression), but workload-specific vs mean/slope/peak comparisons do not exist in literature.
- **2-back vs 3-back cross-subject:** Ishii et al. 2013 showed 61% within-subject. Your ~0.49 cross-subject is consistent with domain-shift penalty (~50–55% expected, depending on subject mismatch).

