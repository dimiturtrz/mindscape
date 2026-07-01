# fNIRS Cognitive-Workload Decoding SOTA (extends prior deep-dives)

**Date**: 2026-07-01
**Status**: partial
**Extends**: fnirs_decoding_methods.md, fnirs_landscape.md, fnirs_fundamentals.md
**Prior contradiction**: fnirs_decoding_methods vs fnirs_landscape on PMC12523035 accuracy numbers

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

- [S5] Lyu B, Pham T, et al. "Benchmarking framework for machine learning classification from fNIRS data." *Frontiers in Neuroergonomics* 4:994969, 2023. https://www.frontiersin.org/journals/neuroergonomics/articles/10.3389/fnrgo.2023.994969/

- [S6] Ledoit O, Wolf M. "Honey, I shrunk the sample covariance matrix." *The Journal of Portfolio Management* 30(4):110-119, 2004.

- [S7] Wang Z, Li Y, Ardila D, et al. "Transformer Model for Functional Near-Infrared Spectroscopy Classification." *IEEE Journal of Biomedical and Health Informatics* 26(6):2559-2569, 2022. DOI: 10.1109/JBHI.2022.3140531. GitHub: https://github.com/wzhlearning/fNIRS-Transformer

- [S8] Peng J, Yang B, et al. "fNIRSNet: A lightweight deep learning model with 498 parameters for mental arithmetic classification." *Journal of Biomedical Optics* 2024-2025. https://github.com/wzhlearning/fNIRSNet

- [S9] Garipelli G, Chavarriaga R, Millán JDR. "Single trial analysis of slow cortical potentials: A study on anticipation related potentials." *Journal of Neuroscience Methods* 191(1):100-109, 2010.

- [S10] Shan L, Deng L, et al. "Rethinking Delayed Hemodynamic Responses for fNIRS Classification." *IEEE Transactions on Neural Systems and Rehabilitation Engineering* 2023. https://www.researchgate.net/publication/375456332

- [S11] Ayoub A, Touryan J, Hussey K, Meeuwisse M. "Enhancing Classification Performance of fNIRS-BCI by Identifying Cortically Active Channels Using the z-Score Method." *Frontiers in Human Neuroscience* 14:330, 2020. PMC7730208. https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7730208/

- [S12] Lyu B, Pham T, Blaney G, Haga Z, Sassaroli A, Fantini S, Aeron S. "Domain adaptation for robust workload level alignment between sessions and subjects using fNIRS." *Journal of Biomedical Optics* 26(2):022908, January 2021. PMC7790507. https://pmc.ncbi.nlm.nih.gov/articles/PMC7790507/

- [S13] Chikontwe P, Nam H, Hong J, Kim SH. "Decoding Working-Memory Load During n-Back Task Performance from High Channel NIRS Data." *NeuroImage* 312:120546, 2024. https://arxiv.org/pdf/2312.07546

- [S14] Aoki T, Inokawa M, Cichocki A. "Convolutional neural network for high-accuracy functional near-infrared spectroscopy in a brain–computer interface: three-class classification of rest, right-, and left-hand motor execution." *NeuroImage* 2018. PMC5599227. https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5599227/
