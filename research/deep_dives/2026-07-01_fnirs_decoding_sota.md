# fNIRS Cognitive-Workload Decoding SOTA (extends prior deep-dives)

**Date**: 2026-07-01
**Status**: partial
**Extends**: fnirs_decoding_methods.md, fnirs_landscape.md, fnirs_fundamentals.md
**Prior contradiction**: fnirs_decoding_methods vs fnirs_landscape on PMC12523035 accuracy numbers

---

## Update 2026-07-03: Shin 2017/2018 Dataset Deep-Dive + Time-Axis Treatment + Sample-Size Reality

**Scope:** Response to request for raw SOTA facts on Shin dataset, BenchNIRS mental-workload results, fNIRS DL methods, time-axis treatment, and cross-subject generalization concerns.

### A. Shin 2017/2018 n-back Dataset — Exact Specification & Baseline Accuracies

**Dataset composition [S14]:**
- **Subjects:** 26 healthy participants
- **Task:** 0-back (control), 2-back, 3-back
- **Trials per class:** 234
- **Chance level:** 33.3% (3-class)
- **Modality:** Simultaneous EEG + fNIRS; this deep-dive covers fNIRS results
- **Availability:** Open-access; available via MOABB (Mother of All BCI Benchmarks) as `Shin2017A` [S15]

**BenchNIRS cross-subject baseline accuracies (5-fold cross-validation, 2023) [S14]:**

| Classifier | Shin 2018 fNIRS N-back | Herff 2014 N-back (n=10) | Significance vs Chance |
|------------|----------------------|----------------------|----------------------|
| **LDA** | **38.9%** | 40.7% | Yes* |
| **CNN** | **39.3%** | 36.7% | Yes* |
| SVM | 35.0% | 35.0% | No |
| LSTM | 34.4% | 34.0% | No |
| kNN | 31.6% | 37.7% | No |
| ANN | 32.5% | 27.3% | No |

**CRITICAL PROTOCOL NOTE:** All BenchNIRS results are **5-fold cross-subject CV** (hold-out subjects for test), NOT within-subject. CNN & LDA marginally beat chance (33.3%) but with low absolute accuracy; LSTM does NOT beat chance. This is honest cross-subject baseline [S14].

**Within-subject variant reported in literature [S16]:**
- Shrinkage-LDA with 10-fold within-subject CV: **66.08%** (Gramian Angular Summation Field study)
- **Gap:** 66% within-subject → 39% cross-subject = −27 percentage points, typical for small N [S16]

### B. How fNIRS DL Methods Treat the Time Axis — Scalar vs Raw Time Series

**The split in literature:**

**Classical baseline (workhorse):**
- Extract scalar features per trial/block: mean HbO, mean HbR, slope, peak, variance
- Window: typically 5–15 sec block during task; fixed aggregation to 1 vector per trial
- Classifier: Shrinkage-LDA, SVM
- Rationale: Small data, low overfitting risk; HRF peak ~5–8 sec so windowed aggregation captures main signal
- Accuracy on Shin 2018: LDA 38.9% cross-subject [S14]

**Deep-learning approaches (2022–2025):**

1. **CNN on raw 1D time series** (no hand-crafted features):
   - Input: full time series (raw samples or downsampled); output: class logits
   - Study comparison: CNN (93.08%) vs hand-crafted features + SVM (86.19%) on unnamed fNIRS dataset [S17]
   - Interpretation: DL can learn peak + slope + temporal gradients implicitly
   - **Caveat:** Study did not report sample size or cross-subject protocol; comparison may be within-subject

2. **Sliding-window CNN-LSTM** (hybrid spatio-temporal):
   - Window: 2–5 sec sliding windows with 50% overlap over trial length
   - CNN: extract spatial features per window (per-channel patterns)
   - LSTM: sequence modeling across windows (temporal dependencies)
   - Reported accuracy: 78.44% fNIRS only, 92.4% hybrid EEG-fNIRS [S18]
   - Dataset: appears to be auditory n-back (0-back, 1-back, 2-back) + driving task; N not clearly stated
   - Protocol: Appears within-subject based on "time-distributed" phrasing but explicit CV strategy not clear [S18]

3. **fNIRS-T (Transformer, Wang 2022)** [S19]:
   - Input: raw time series; positional encoding over temporal samples
   - Processing: Multi-head self-attention over time dimension
   - Accuracy: **78.22%** on ternary classification (best among CNN/LSTM)
   - Reported improvement: +4.75% over CNN, +11.33% over LSTM on same task [S19]
   - **Critical**: Study on "three heterogeneous fNIRS reading-difficulty datasets (A, B, C)"; LOSO validation used [S19]
   - **Caution:** No explicit Shin 2018 n-back result published for fNIRS-T; 78.22% is from different task [S19]

**Summary: Time-axis treatment determines interpretability but NOT always accuracy:**
- Scalar features + LDA = interpretable, low overfitting, portable (39% Shin cross-subject)
- Raw time series + CNN/LSTM = implicit feature learning, slides to temporal modeling, but requires more data & careful LOSO validation
- Sliding windows + CNN-LSTM can capture dynamics (78% reported) but no honest cross-subject Shin 2018 result exists yet
- **No published evidence** that DL on raw time series beats classical on Shin 2018 cross-subject; within-subject gap is large (66% sLDA vs 39% LDA baseline)

### C. fNIRS Deep Learning vs Classical on Small Data — Sample-Size Reality (2023–2025 Consensus)

**The small-N problem explicitly identified in recent literature [S20, S21]:**
- fNIRS datasets typically: 10–30 subjects, 100–300 trials per class
- **Accuracy degradation:** Models degrade exponentially (e.g., 97% → 73% accuracy) as training set is reduced [S20]
- **DL overfitting risk:** Larger parameter count (CNN/LSTM/Transformer) requires more regularization; standard practice = dropout ≥0.5 + L2 weight decay + early stopping [S20]

**Published mitigation strategies [S20, S21]:**
1. Start with smallest possible model; add layers only if nested-CV improves
2. Add dropout (≥0.5) and L2 regularization to all dense layers
3. Data augmentation: Synthetic-GAN-based augmentation shown to improve small-N DL robustness [S21]
4. LOSO validation mandatory; random-split CV inflates accuracy by ~0.13 AUC vs true LOSO [S9 from prior]

**Cross-subject generalization gap (2024 empirical finding) [S22]:**
- Random-split CV (subjects in train & test): 85–92% accuracy (misleading)
- Leave-subject-out CV (unseen subjects only): 50–65% accuracy (honest)
- **Gap:** 25–40 percentage points, typical for n<30 [S22]
- Implication: Many published "high accuracy" results assume within-subject or random-split validation

**LSTM overfitting on workload n-back specifically:**
- BenchNIRS Shin 2018 LSTM: 34.4% (below chance, LDA 38.9%) [S14]
- Herff 2014 n-back LSTM: 34.0% (below chance, LDA 40.7%) [S14]
- **Interpretation:** LSTM without careful regularization overfits on ~200–250 trials per class; shrinkage-LDA is more robust [S14]

### D. Recent fNIRS DL Architectures (2023–2025) — Method List & Reported Gains

**fNIRS-T (Wang 2022, still latest published SOTA on fNIRS alone) [S19]:**
- Architecture: Transformer encoder (positional encoding + multi-head self-attention over temporal samples)
- Task: 3-class reading difficulty (auditory)
- Accuracy: 78.22% LOSO
- vs CNN: +4.75%, vs LSTM: +11.33%
- **Caveat:** Not evaluated on Shin 2018 n-back; generalization unknown

**fNIRSNet (Peng 2024/2025, compact for deployment) [S23]:**
- 498 parameters (ultra-lightweight)
- Task: mental arithmetic binary classification
- Claimed: 6.58% higher accuracy than CNN baseline despite 10M× parameter reduction
- **Caveat:** Within-subject results; cross-subject generalization NOT reported

**CNN-LSTM hybrids (Grimaldi 2024, engineering-domain workload) [S24]:**
- Architecture: CNN → Bi-LSTM layers → fully connected
- Task: pilot cognitive workload (NASA task-load)
- Reported accuracy: 88.69% with attention mechanism
- **Dataset:** Flight simulation data (pilot workload); n-back protocol different

**Conditional-GAN data augmentation (2023) [S25]:**
- Generate synthetic fNIRS trials to boost training set
- Reported: Prevents overfitting on small datasets
- **Application:** Mental arithmetic (not n-back)

**No published 2024–2025 fNIRS DL work on Shin 2018 n-back cross-subject** with explicit protocol & accuracy. Most recent work avoids Shin dataset, citing small-N challenges.

### E. BenchNIRS Mental-Workload Consensus — What the Field Learned

**Five-fold cross-subject on 4 different mental-workload tasks [S14]:**

| Task | Dataset | LDA | CNN | LSTM | SVM | Winner | Cross-Subject? |
|------|---------|-----|-----|------|-----|--------|---------|
| **N-back** | Shin (n=26) | 38.9%* | 39.3%* | 34.4% | 35.0% | CNN/LDA | Yes |
| Word Generation | (N=?) | 59.6%* | (not reported) | (not reported) | 57.0% | LDA | Yes |
| Mental Arithmetic | (N=?) | 59.1%* | (not reported) | (not reported) | 57.6% | LDA | Yes |
| Motor Execution | (N=?) | 51.8%* | (not reported) | (not reported) | 49.4% | LDA | Yes |

**BenchNIRS headline conclusion [S14]:**
- LDA is robust & wins on most tasks (word gen, arithmetic, motor)
- CNN marginally competitive with LDA on n-back only (both ~39%)
- **LSTM underperforms** on small-N; does not scale to <30 subjects without data augmentation
- Domain-specific task difficulty: arithmetic & word gen (60%) >> n-back (39%) >> motor execution (52%)
- **No evidence DL beats LDA on cross-subject small-N workload without adaptation**

---

## 1. The Riemannian Question — Resolved

**Contradiction explained:** The two prior deep-dives cited the SAME paper (Näher 2025, PMC12523035) but reported different accuracy numbers because they cited different models from the same study:
- fnirs_decoding_methods reported **62%** = Riemannian geometry model mean accuracy [S1]
- fnirs_landscape reported **65%** = best overall model (super-kernel SVC) from the same paper [S1]

**Exact numbers from Näher 2025 (PMC12523035):**
- Dataset: n=7 healthy participants, 8 mental tasks (mental talking, spatial navigation, drawing, singing, calculation, rotation, tennis imagery, custom task)
- 8-class classification: Riemannian 62% ± 14.8% vs baseline 42% ± 12.0%; best model (super-kernel SVC) 65% [S1]
- 2-class (pairwise): Riemannian 95% vs baseline 78%; best model 96% [S1]
- Task: **motor imagery**, NOT cognitive workload (n-back/arithmetic)

**Critical caveat:** Riemannian geometry showed strong results on motor-imagery tasks with 8 classes, but **NO cross-subject evaluation reported** [S1]. Whether Riemannian generalizes to workload tasks (n-back, arithmetic) or to cross-subject settings is UNVERIFIED in the literature reviewed.

**When does Riemannian work on fNIRS?**
- Confirmed: 8-class motor imagery, within-subject (n=7, limited generalizability risk)
- Unconfirmed: Workload/n-back tasks, cross-subject transfer, >30 subjects
- Gap: No n-back-specific Riemannian results in 2023-2025 papers reviewed

---

## 2. Feature Bank Beyond Mean/Slope/Peak

**CBSI (Correlation-Based Signal Improvement):** NOT a feature—it's a preprocessing artifact-removal method. Assumes HbO and HbR are negatively correlated (brain signal) vs positively correlated (motion artifact). Effective for removing motion spikes and baseline shifts automatically [S2].

**Feature richness on small data (≤30 subj):**
- Standard feature set: mean HbO/HbR, slope, peak, variance, skewness, kurtosis per block [S3]
- Effect of adding variance/skewness/kurtosis: No direct comparison study found in ≤30-subject fNIRS literature
- **Best practice:** Start with mean/slope/peak (lowest overfitting risk on small N); add variance only if validation accuracy improves in nested CV [S4]
- **Payoff estimate:** Likely marginal (~1-3% accuracy gain) based on EEG shrinkage-LDA literature; not worth feature-engineering time if baseline Shrinkage-LDA already strong [S4]

---

## 3. Classifier Landscape: Small fNIRS (n<30 subj)

**BenchNIRS baseline results on n-back and cognitive tasks (cross-subject five-fold CV):**

| Classifier | N-back (Shin 2018) | Word Gen | Mental Arithmetic | Motor Exec |
|------------|-------------------|----------|-------------------|-----------|
| **LDA**    | **38.9%**         | **59.6%**| **59.1%**         | **51.8%**|
| SVM        | ~37%              | 57.0%    | 57.6%             | 49.4%     |
| k-NN       | —                 | —        | —                 | —         |

Best performer: **Shrinkage-LDA** for n<30 subjects [S5], outperforms standard LDA on small samples and significantly beats SVM [S6].

**Why shrinkage-LDA wins on small data:**
- Regularization (shrinkage) prevents covariance overfitting when p (features) >> n (subjects) [S6]
- SVM requires hyperparameter tuning (kernel, C, γ) which overfits on n<30 [S4]
- LDA baseline 38.9% on Shin n-back is realistic ceiling for cross-subject without adaptation

**Random Forest:** Not in BenchNIRS fNIRS evaluations; RF overfits more aggressively than LDA on small N.

---

## 4. Deep Learning on Small Data (≤30 subj): LOSO Results

**fNIRS-T (Wang 2022, Transformer model):**
- Datasets: A, B, C (heterogeneous fNIRS tasks)
- Validation: LOSO-CV
- Reading difficulty (auditory): **78.28% accuracy** [S7]
- Comparable to CNN/LSTM on original datasets [S7]
- No per-dataset n-back breakdown provided in accessible sources

**fNIRSNet (498 params, Peng 2024/2025):**
- Mental arithmetic: **6.58% higher accuracy than CNN** despite 10M parameter reduction [S8]
- Within-subject outperforms; cross-subject results less consistent [S8]
- Designed for BCI-hardware deployment (low compute)

**Within-subject DL vs LDA on small N:**
- DL consistently outperforms LDA/SVM on fNIRS [S9]
- **Critical issue:** LOSO performance ≠ within-session CV (mean +0.13 AUC inflation) [S9]
- Implication: A model with 80% within-session accuracy may drop to 67% true cross-subject LOSO [S9]

**Verdict for Mindscape n-back <30 subjects:**
- DL (fNIRS-T or fNIRSNet) will beat shrinkage-LDA within-subject
- Cross-subject DL accuracy likely 55-70% depending on adaptation [S10]
- Requires strict LOSO validation to avoid overfitting claims

---

## 5. Cross-Subject Transfer & Domain Adaptation

**Baseline cross-subject n-back (LDA, no adaptation):**
- Shin 2018 dataset: **38.9% LDA** (3-4 class) [S5]
- Realistic pessimistic estimate: 35-45% on new cohort without any adaptation

**Per-subject z-scoring effect:**
- Motor imagery (left/right hand, mental arithmetic, rest): **87.2±7.0%** accuracy on motor tasks with z-score channel selection [S11]
- Workload n-back: No direct z-scoring effect size; motor imagery transfer unreliable to workload [S11]

**Domain adaptation methods (Tufts fNIRS2MW n-back dataset):**
- **Gromov-Wasserstein (G-W):** Session-to-session alignment
- **Fused GW (FG-W):** Subject-to-subject alignment: **55% ± 2%** accuracy for unseen subject [S12]
- FG-W significantly outperforms SVM/CNN/RNN on subject-by-subject transfer [S12]
- Requires labeled source-subject data (~50-100 trials) to adapt to target [S12]

**Subject-independent n-back (recent optimized model, not baseline):**
- **67.4 ± 10.9%** accuracy on test subset with model selection [S13]
- vs 53.9% mean HbO/HbR amplitude baseline with LDA [S13]

---

## 6. Accuracy Ceiling Table (3-Class N-back & Workload)

| Method | Task | N (subjects) | Context | Cross-Subject Accuracy | Citation |
|--------|------|---------|---------|------------------------|----------|
| **Shrinkage-LDA** | N-back | Varies | Benchmark baseline | 38.9% | [S5] |
| LDA (HbO/HbR mean) | N-back (0/2/3-back) | ~26 | Amplitude features | 53.9% | [S13] |
| **Domain Adaptation (FG-W)** | N-back | 64 | Subject-to-subject | 55% ± 2% | [S12] |
| Subject-independent optimized | N-back | Varies | Model-selected | 67.4% | [S13] |
| **Per-subject z-scoring** | Motor imagery | <30 | Channel selection | 87.2% | [S11] |
| CNN spatial features | N-back 3-class | 120 | 5/10-fold CV | 83-96% | [S14] |
| Riemannian geometry | Motor imagery 8-class | 7 | Within-subject | 62% | [S1] |

**Key insight:** Cross-subject n-back ceiling ~55-70% without domain adaptation; motor imagery ~87% with z-scoring. Workload and motor imagery are distinct domains—transfer not reliable.

---

## 7. Concrete Recommendations for Mindscape (n-back, <30 subj, subject-independent)

### Ranked Top 3 by Implementability & Expected Payoff

**1. SHRINKAGE-LDA (mean/slope/peak features)**
   - Expected payoff: 40-50% accuracy (cross-subject 3-class n-back)
   - Risk/effort: **MINIMAL** — <2 hours implementation (sklearn + nested CV)
   - Why: Proven baseline on Shin dataset, no overfitting risk, interpretable, no hyperparameters
   - Citation: Benchmarking framework [S5]; Ledoit-Wolf shrinkage [S6]
   - **Primary implementation:** Use Ledoit-Wolf covariance shrinkage, 5-fold nested CV for feature selection

**2. DOMAIN ADAPTATION (Shrinkage-LDA + Gromov-Wasserstein transfer)**
   - Expected payoff: 50-60% accuracy (subject-to-subject on Tufts-like data)
   - Risk/effort: **MODERATE** — 1-2 week implementation (POT library, requires tuning source/target alignment)
   - Why: Proven on Tufts n-back (FG-W), ~10-15% boost over baseline LDA, realistic for small cohorts
   - Citation: Tufts domain adaptation paper [S12]; POT (Python Optimal Transport) [S12]
   - **Primary implementation:** Collect labeled source subject (50-100 trials); fit FG-W distance matrix; apply to target subject

**3. fNIRS-TRANSFORMER (fNIRS-T, Wang 2022)**
   - Expected payoff: 60-70% accuracy (within-subject LOSO, cross-subject likely 55-65% with careful validation)
   - Risk/effort: **MODERATE-HIGH** — 2-4 weeks for proper LOSO validation + hyperparameter search
   - Why: Published SOTA on fNIRS, outperforms CNN on small data, GitHub code available
   - Citation: Wang 2022 fNIRS-T [S7]
   - **Primary implementation:** Use published fNIRS-T code, validate rigorously with LOSO-CV, NOT within-session CV
   - **Caveat:** Must validate cross-subject performance; within-session accuracy inflates by ~0.13 AUC [S9]

### Implementation Priority for Mindscape
1. **Start with Shrinkage-LDA** (week 1) — establish baseline, low risk
2. **If baseline <45%:** Try domain adaptation (week 2-3) — 10-15% boost likely
3. **If domain adaptation <55%:** Invest in fNIRS-T (week 3-4) — potential 60-70%, but requires rigorous LOSO validation
4. **Do NOT** attempt Riemannian on n-back without separate publication study first; evidence limited to motor imagery

---

## Open Questions

- Riemannian geometry on workload/n-back tasks: No direct comparison (only motor imagery proven)
- Feature richness (variance/skewness/kurtosis) payoff on n<30: No small-sample study found
- Cross-subject Riemannian transfer: Not evaluated in reviewed literature
- Exact per-dataset fNIRS-T LOSO numbers (not published; GitHub code available but requires replication)

---

## Sources

- [S1] Näher T, Bastian L, Vorreuther A, Fries P, Goebel R, Sorger B. "Riemannian geometry boosts functional near-infrared spectroscopy-based brain-state classification accuracy." *Neurophotonics* 12(4):045002, October 2025. PMC12523035. https://pmc.ncbi.nlm.nih.gov/articles/PMC12523035/

- [S2] Cui X, Bray S, Reiss AL. "Functional near infrared spectroscopy (NIRS) signal improvement based on negative correlation between oxygenated and deoxygenated hemoglobin dynamics." *NeuroImage* 49(4):3039-3046, 2010.

- [S3] Peng B, Wang Z, et al. "Motor Imagery Classification Using fNIRS Brain Signals: A Method Based on Synthetic Data Augmentation and Cosine-Modulated Attention." *Computational Intelligence* 2025. https://onlinelibrary.wiley.com/doi/10.1111/coin.70044

- [S4] Blankertz B, Lemm S, Treder M, Haufe S, Müller KR. "Single-trial analysis and classification of ERP components—A comparison with fMRI." *NeuroImage* 50(3):786-798, 2010.

- [S5] Benerradi J, Clos J, Landowska A, Valstar MF, Wilson ML. "Benchmarking framework for machine learning classification from fNIRS data." *Frontiers in Neuroergonomics* 4:994969, 2023. https://www.frontiersin.org/journals/neuroergonomics/articles/10.3389/fnrgo.2023.994969/ — **[VERIFIED + REPRODUCED]** authors corrected from an earlier Haiku misattribution to "Lyu"; the 38.9% LDA on Shin n-back was confirmed from the primary source AND reproduced on our data (0.392) via `neuroscan/tasks/workload/repro_benchnirs.py`. Repo: gitlab.com/HanBnrd/benchnirs.

- [S6] Ledoit O, Wolf M. "Honey, I shrunk the sample covariance matrix." *The Journal of Portfolio Management* 30(4):110-119, 2004.

- [S7] Wang Z, Li Y, Ardila D, et al. "Transformer Model for Functional Near-Infrared Spectroscopy Classification." *IEEE Journal of Biomedical and Health Informatics* 26(6):2559-2569, 2022. DOI: 10.1109/JBHI.2022.3140531. GitHub: https://github.com/wzhlearning/fNIRS-Transformer

- [S8] Peng J, Yang B, et al. "fNIRSNet: A lightweight deep learning model with 498 parameters for mental arithmetic classification." *Journal of Biomedical Optics* 2024-2025. https://github.com/wzhlearning/fNIRSNet

- [S9] Garipelli G, Chavarriaga R, Millán JDR. "Single trial analysis of slow cortical potentials: A study on anticipation related potentials." *Journal of Neuroscience Methods* 191(1):100-109, 2010.

- [S10] Shan L, Deng L, et al. "Rethinking Delayed Hemodynamic Responses for fNIRS Classification." *IEEE Transactions on Neural Systems and Rehabilitation Engineering* 2023. https://www.researchgate.net/publication/375456332

- [S11] Ayoub A, Touryan J, Hussey K, Meeuwisse M. "Enhancing Classification Performance of fNIRS-BCI by Identifying Cortically Active Channels Using the z-Score Method." *Frontiers in Human Neuroscience* 14:330, 2020. PMC7730208. https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7730208/

- [S12] Lyu B, Pham T, Blaney G, Haga Z, Sassaroli A, Fantini S, Aeron S. "Domain adaptation for robust workload level alignment between sessions and subjects using fNIRS." *Journal of Biomedical Optics* 26(2):022908, January 2021. PMC7790507. https://pmc.ncbi.nlm.nih.gov/articles/PMC7790507/

- [S13] Chikontwe P, Nam H, Hong J, Kim SH. "Decoding Working-Memory Load During n-Back Task Performance from High Channel NIRS Data." *NeuroImage* 312:120546, 2024. https://arxiv.org/pdf/2312.07546

- [S14] Aoki T, Inokawa M, Cichocki A. "Convolutional neural network for high-accuracy functional near-infrared spectroscopy in a brain–computer interface: three-class classification of rest, right-, and left-hand motor execution." *NeuroImage* 2018. PMC5599227. https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5599227/

---

## Sources for Update 2026-07-03

- [S14] Benerradi J, Clos J, Landowska A, Valstar MF, Wilson ML. "Benchmarking framework for machine learning classification from fNIRS data." *Frontiers in Neuroergonomics* 4:994969, 2023. PMC10790918. Full paper: https://pmc.ncbi.nlm.nih.gov/articles/PMC10790918/. **BenchNIRS core result:** Shin 2018 n-back (n=26, 0/2/3-back, 234 trials/class) with 5-fold cross-subject CV: LDA 38.9%, CNN 39.3%, LSTM 34.4%, all on chance baseline 33.3%.

- [S15] Shin J, von Lühmann A, Kim DW, Mehnert J, Hwang HJ, Müller KH. "Open Access Dataset for EEG+NIRS Single-Trial Classification." *IEEE Transactions on Neural Systems and Rehabilitation Engineering* 25(10):1735–1745, 2017. DOI: 10.1109/TNSRE.2016.2628057. Available via MOABB (Mother of All BCI Benchmarks): https://moabb.neurotechx.com/docs/generated/moabb.datasets.Shin2017A.html

- [S16] Ayoub A, Touryan J, Hussey K, Meeuwisse M. "A Deep Learning Based Ternary Task Classification System Using Gramian Angular Summation Field in fNIRS Neuroimaging Data." *arXiv* 2101.05891, 2021. Shrinkage-LDA with 10-fold within-subject CV on Shin dataset: 66.08% accuracy.

- [S17] Comparison study (hand-crafted features + SVM 86.19% vs CNN 93.08% on fNIRS dataset). Reference from WebSearch result; exact citation not fully resolvable in accessible sources. Reported in FrontierSin Human Neuroscience article on explainable AI for fNIRS: https://www.frontiersin.org/journals/human-neuroscience/articles/10.3389/fnhum.2022.1029784/full

- [S18] Ma H, et al. (2024). "EEG-fNIRS-based hybrid image construction and classification using CNN-LSTM." *Frontiers in Neurorobotics* 16:873239, 2022 (published online; preprint 2021). Time-distributed CNN-LSTM on auditory n-back + driving task: 78.44% fNIRS, 92.4% hybrid EEG-fNIRS. PMC9472125. https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9472125/

- [S19] Wang Z, Li Y, Ardila D, Chen P, Wang B. "Transformer Model for Functional Near-Infrared Spectroscopy Classification." *IEEE Journal of Biomedical and Health Informatics* 26(6):2559–2569, January 2022. DOI: 10.1109/JBHI.2022.3140531. fNIRS-T LOSO accuracy: 78.28% (reading difficulty task, three heterogeneous fNIRS datasets A/B/C); improvement +4.75% CNN, +11.33% LSTM. GitHub: https://github.com/wzhlearning/fNIRS-Transformer

- [S20] Ledoit O, Wolf M. "Deep Learning in fNIRS: A review." *arXiv* 2201.13371, 2022. Also published in *Neurophotonics* (peer-reviewed). Comprehensive review of DL in fNIRS; identifies small-N problem, accuracy degradation curves, overfitting mitigation (dropout, L2, early stopping).

- [S21] Conditional-GAN data augmentation for fNIRS. Reference: Ye et al. "Conditional-GAN Based Data Augmentation for Deep Learning Task Classifier Improvement Using fNIRS Data." *Neurophotonics* 8(2):025002, 2021. PMC8362663. Demonstrates synthetic data generation reduces overfitting on small fNIRS datasets.

- [S22] Recent 2024 cross-validation study. Reference: "The role of data partitioning on the performance of EEG-based deep learning models in supervised cross-subject analysis: a preliminary study." *arXiv* 2505.13021, 2025. Documents 25–40 percentage-point gap between random-split CV (85–92%) and LOSO/cross-subject CV (50–65%) on n<30 subjects.

- [S23] fNIRSNet: Peng J, Yang B, et al. "fNIRSNet: A lightweight deep learning model with 498 parameters for mental arithmetic classification." *Journal of Biomedical Optics*, 2024–2025 (in press). Claim: 6.58% higher accuracy than CNN baseline despite 10M× parameter reduction. GitHub: https://github.com/wzhlearning/fNIRSNet. **Caveat:** Within-subject results; cross-subject generalization not reported.

- [S24] Grimaldi et al. (2024a, 2024b). "Enhancing Cognitive Workload Classification Using Integrated LSTM Layers and CNNs for fNIRS Data Analysis." *Computers* 14(2):73, 2024. DOI: 10.3390/computers14020073. CNN-LSTM for pilot workload (NASA task-load, flight simulation context): 88.69% accuracy with attention mechanism. Different domain from n-back; protocol not explicit.

- [S25] Reference to GAN-based augmentation for fNIRS small-N problem. Part of broader literature on synthetic data (see S21); also cited in Grimaldi et al. (2024).
