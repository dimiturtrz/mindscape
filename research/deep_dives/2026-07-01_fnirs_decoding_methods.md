# fNIRS Decoding Methods — What the Field ACTUALLY Uses (mental-workload / n-back / cognitive-state BCI)

**Date:** 2026-07-01
**Purpose:** Drive a real baseline implementation for mindscape. Focus: field-standard decoding for HbO/HbR amplitude signals — NOT EEG's CSP/covariance/Riemannian methods (those mismatch fNIRS; see caveat box in §3/§5).
**Confidence:** Feature set, classifier choice, and timing conventions are strongly corroborated across multiple sources. Exact accuracy numbers are task/dataset-specific — quoted with source. Flagged where unverified.

---

## ★ WHAT TO IMPLEMENT FOR MINDSCAPE (ranked, concrete)

**Baseline #1 (implement first) — Feature+shrinkage-LDA per channel.**
- Preprocess: convert to ΔHbO/ΔHbR (modified Beer-Lambert), band-pass ~0.01–0.1/0.2 Hz to kill Mayer waves/cardiac/respiration, baseline-correct to pre-task window.
- Epoch the task block (n-back trial/block). **Feature window: the task block; if using a sliding window, 0–10 s from onset is the benchmark default (Kwon/Benerradi framework used 2–10 s windows, 10 s default).** Peak HbO tends to occur ~5–8 s post-onset, so a 2–7 s or whole-block window both work.
- **Features per channel (the workhorse set): MEAN and SLOPE of ΔHbO over the window. Add PEAK/MAX for a strong 3-feature set.** Signal mean + signal peak was the single best 2-feature combo for LDA (Noori et al. 2016: 93.0% HbO, 89.9% HbR). Mean/slope/peak are consistently the most discriminative; variance, skewness, kurtosis help only in synergy.
- Use **BOTH HbO and HbR** channels stacked (HbO carries most of the signal — ~6% better than HbR alone, MNE-NIRS — but HbR is decorrelated from systemic noise so it adds robustness).
- Classifier: **shrinkage-regularized LDA** (`sklearn.discriminant_analysis.LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')`) — the field workhorse, robust with few trials / many channels. Linear SVM is the standard alternative.
- Per-subject z-score normalization of each channel feature is standard for cross-subject (see §5).

**Baseline #2 — "flatten the epoch" logistic regression (MNE-NIRS canonical recipe).**
- Skip hand features. Take `epochs.get_data()` (channels × time), pipe `Scaler → Vectorizer → LogisticRegression(solver='liblinear')`, 5-fold CV, ROC-AUC. This is literally the MNE-NIRS decoding tutorial. ~89% ROC-AUC HbO on finger-tapping. Good sanity baseline.

**Baseline #3 (only if #1/#2 saturate) — a small 1D-CNN or fNIRSNet-style net on raw ΔHbO/ΔHbR time series.** Deep learning does NOT reliably beat feature+LDA within-subject on small fNIRS datasets and overfits easily; its main payoff is subject-independent generalization, and even there gains are modest (Tufts: ~62–68% cross-subject regardless of model). Do not lead with this.

**Do NOT** default to CSP/covariance/tangent-space/Riemannian pipelines. They are EEG-native (oscillatory power / spatial covariance). fNIRS discriminability lives in per-channel HbO amplitude & slope. One 2025 paper (Nguyen et al.) got covariance-Riemannian to work on fNIRS by stacking HbO/HbR block-diagonal covariance — but n=7 and the authors caveat the "signal" is partly artifact (§3). Treat as research curiosity, not baseline.

---

## 1. CANONICAL fNIRS FEATURES

**The workhorse temporal features** (computed per channel, per chromophore, over the task window):
- **Signal MEAN** — average ΔHbO/ΔHbR over window. *The* dominant feature.
- **SLOPE** — linear-regression slope of the signal over the window (captures the rising hemodynamic ramp). Co-dominant with mean.
- **PEAK / MAX value** (and MIN). Strong; pairs with mean for the best LDA 2-feature set.
- Secondary: **variance / standard deviation, skewness, kurtosis, time-to-peak (latency), area-under-curve (AUC/integral)**. These are the "most widely used" list per the hybrid fNIRS-EEG review (Hong et al., Front. Hum. Neurosci. 2018) but individually weaker; they contribute in synergy.
- **Initial dip**: an early transient HbO decrease; discussed physiologically but rarely a standard classifier feature (unreliable to detect).

**Benchmark evidence on which features win:**
- **Noori, Naseer et al. 2016** (PMC4879140), "Determining Optimal Feature-Combination for LDA": candidate set = {mean, slope, variance, peak, skewness, kurtosis}; 10-fold CV; mental-arithmetic vs rest; whole 44 s task window; 0.1–0.3 Hz band. **Best 2-feature combo = MEAN + PEAK: 93.0% (HbO), 89.9% (HbR).** Best 3-feature combos all contained mean+peak plus one of {slope, kurtosis, variance, skewness}, 92–94%. Verified.
- **Feature-ablation study** (via lit review): removing max/min/mean dropped accuracy to 80.74% (critical features); removing kurtosis+skewness only dropped to 95.9% (marginal). Confirms mean/peak/slope are load-bearing.
- **Kwon & Benerradi "Benchmarking framework for ML classification from fNIRS data"** (PMC10790918, 2024): uses exactly **mean, standard deviation, slope** per region-of-interest per chromophore. Window sweep 2–10 s, 10 s default. This is the cleanest citable "standard feature set" statement.
- **Aghajani, Omurtag et al.**: SVM + moving-window on n-back cognitive load, 74.8% binary — used mean-type window features.

**Window timing conventions:** whole task block is common; for sliding/online, 0–10 s from stimulus onset is the benchmark default. Some report best discrimination in later windows (Noori: 11–20 s window ≥91% with GA-SVM optimal features) because the hemodynamic response is delayed 5–8 s. **HbO vs HbR vs both:** HbO has the higher SNR and is the primary feature; HbR adds a decorrelated view. Standard practice = use both.

---

## 2. STANDARD CLASSIFIERS

**Field workhorse ranking for fNIRS mental workload:**
1. **LDA / shrinkage-LDA** — most-used, robust in the small-n/high-dim regime typical of fNIRS. Reported: 62–71% on mental arithmetic / word generation / mixed mental tasks; Liu et al. 68.1% on n-back (EEG+fNIRS combined).
2. **SVM (linear, then RBF)** — the other dominant choice. Shin et al. ~75–77% mental-arithmetic vs baseline; Aghajani 74.8% binary n-back cognitive load.
3. **Regularized logistic regression** — MNE-NIRS default; ~89% ROC-AUC HbO (finger-tapping tutorial).
4. **kNN, Random Forest** — used, generally weaker than LDA/SVM on the standard feature set. In the Kwon/Benerradi generalized benchmark kNN was worst.

**Representative accuracy numbers (quote + cite):**
- Kwon & Benerradi generalized (subject-independent) 10 s epochs, **mean/std/slope features**: Mental Arithmetic — LDA 59.1%, SVM 57.6%, kNN 54.5%, ANN 57.9%, CNN 60.2%, LSTM 59.2%. Word Generation — LDA 59.6% (best classical), CNN 58.7%. Motor Execution — LDA 51.8%. **Key takeaway: cross-subject/generalized accuracy is LOW (~55–60%) and LDA ≈ CNN ≈ LSTM — deep learning gives no advantage in the generalized setting.**
- Within-subject with optimized features, binary rest-vs-task: **91.31%** (mean+slope combo, cited in workload lit); Noori 93% (mean+peak). Within-subject is where the high numbers come from.
- Shin 2018 dataset word-generation baseline reported up to **95.2%** (WG vs baseline is an easy contrast; n-back load discrimination is much harder).

**Rule of thumb for mindscape:** within-subject binary rest-vs-task 85–93% is achievable with feature+LDA; cross-subject and multi-level load (0/2/3-back) collapse toward 55–70%.

---

## 3. DEEP LEARNING FOR fNIRS

**Architectures actually used:**
- **1D-CNN** on raw ΔHbO/ΔHbR time series (channels as input dims) — most common.
- **2D-CNN** treating channel×time as an image, or transforming to **GAF/GASF images** (Wickramaratne & Mahmud CNN on Gramian Angular Summation Fields: **87.14%** mental arithmetic vs fixation).
- **LSTM / BiLSTM** and **CNN-LSTM hybrids** (e.g. MDPI Computers 14(2):73 integrated LSTM+CNN for workload).
- **fNIRS-specific nets:**
  - **fNIRS-T (Transformer)** — Wang, Zhang et al., IEEE JBHI 2022 (github wzhlearning/fNIRS-Transformer). Spatial + channel-level attention; 1D-avgpool+layernorm preprocessing replaces filtering (fNIRS-PreT end-to-end variant). Reported best-in-class vs traditional ML/CNN/LSTM on three open datasets; LOSO ~90% quoted on open data. [accuracy figure partially verified — exact per-dataset table not fetched]
  - **fNIRSNet** — Wang 2024, injects delayed-hemodynamic-response domain knowledge (kernel/receptive-field sized to HRF delay). **Only 498 parameters, yet 6.58% higher than a million-param CNN on mental arithmetic.** Strong efficiency story. Calibration study (arXiv 2402.15266) notes fNIRSNet is weak at OOD rejection vs transformers.
  - Newer: TopoTempNet, FCS-TPNet, dilation-CapsuleNet — motor-imagery/MA, niche.

**Do they beat feature+LDA?**
- **Within-subject small data: NO reliable win, and overfitting is the norm.** The Kwon/Benerradi generalized benchmark shows CNN/LSTM within ~1% of LDA (sometimes below). "Deep Learning in fNIRS: A review" (arXiv 2201.13371) repeatedly flags small-dataset overfitting.
- **Where DL helps:** large multi-subject datasets and subject-independent transfer, where end-to-end nets + domain adaptation edge out static features — but gains are modest (Tufts §4: ~62→68% with domain adaptation).
- Best single-dataset DL numbers (recent hybrid, MA/UFFT): **90.04 ± 3.53% (MA), 81.66 ± 3.23% (UFFT)** — these are strong open-dataset within-paradigm results, not cross-subject workload.

**Covariance/Riemannian on fNIRS (the anti-pattern to avoid, documented):** Nguyen et al. 2025 (PMC12523035) forced Riemannian geometry onto fNIRS by building **channel covariance matrices** (Ledoit-Wolf shrinkage etc.), stacking HbO+HbR block-diagonal: 8-class 62% vs 42% traditional, 2-class 95% vs 78%. BUT n=7, no short-separation regression, and authors explicitly warn the HbO/HbR "complementarity likely reflects divergent artifact sensitivities, not independent neural signals." **Verdict: not a robust baseline; the amplitude/slope features are the honest fNIRS signal.**

---

## 4. n-BACK / MENTAL-WORKLOAD SPECIFIC

**Datasets & their baselines:**
- **Shin 2018 (Scientific Data 5:180003)** — 26 subjects, EEG+fNIRS (36 fNIRS ch @10 Hz), tasks: n-back (0/2/3-back), DSR, word generation. Community baselines: WG-vs-baseline up to **95.2%**; n-back **load** discrimination is much harder (multiclass 0/2/3-back typically 40–70%). Shin's own papers used **shrinkage-LDA on mean/slope-type features**. TU-Berlin hosts data at doc.ml.tu-berlin.de/simultaneous_EEG_NIRS.
- **Tufts fNIRS2MW (NeurIPS 2021 D&B, Huang et al.)** — largest workload set, **68 participants**, 30 s windows, n-back to induce workload, predict 2-class low-vs-high (0-back vs 2-back). Benchmark models: **Logistic Regression, Random Forest, DeepConvNet, EEGNet**, under subject-specific / generic / generic+finetune paradigms. Input = the multivariate fNIRS window (HbO+HbR + optical intensity variants). **Subject-independent accuracy is only ~62–64% for the deep nets** (from Table V of Wang et al. block-as-domain paper citing fNIRS2MW: DeepConv 63.75%, EEGNet 62.08% baseline; with block-wise domain adaptation DeepConv 67.56%, EEGNet 66.51%, MLPMixer 67.91%). **This is the single most important reality check: cross-subject n-back workload decoding tops out in the mid-60s%.** [subject-specific numbers not fetched from source PDF — flagged unverified; expect higher, ~70–80%.]
  - Code: github.com/tufts-ml/fNIRS-mental-workload-classifiers. Leaderboard-style follow-ups: NIRS-X adaptive learning framework (ACM 2024), block-as-domain adaptation (arXiv 2405.00213).
- **Aghajani et al.** — n-back cognitive load, SVM + moving window, **74.8% binary** (EEG+fNIRS fusion higher).

**Best-reported n-back methods:** within-subject binary low-vs-high with feature+LDA/SVM lands 70–85%; cross-subject with domain adaptation ~65–68%; 3–4 class load levels much lower (chance-adjusted).

---

## 5. CROSS-SUBJECT / TRANSFER

fNIRS is **amplitude-based**, so absolute ΔHbO magnitude varies hugely by subject (skull, hair, optode coupling). Standardization is essential:
- **Per-channel, per-subject z-score / standardization** of features (or of the raw signal) is the standard normalization. The MNE-NIRS pipeline's `Scaler` normalizes each channel by mean/std. This is the baseline transfer trick — do it before any cross-subject model.
- **Domain adaptation** is the field's main cross-subject tool (Yang et al. PMC7790507: Gromov-Wasserstein / fused-GW alignment; block-wise / block-as-domain adaptation arXiv 2405.00213 treats each block as a domain to cut intra-session variance; adversarial feature alignment). Gains are real but modest (~+4–5% on Tufts).
- **Transfer learning:** heterogeneous transfer for cross-subject motor-imagery (PMC11983500); generic-pretrain + per-subject fine-tune is the Tufts "generic+finetuning" paradigm.
- **Practical recipe for mindscape cross-subject:** z-score each channel feature within subject → train LDA/SVM on pooled subjects (LOSO eval) → optionally add a domain-adaptation or per-subject-calibration step. Expect ~60–68% on hard n-back workload, higher on rest-vs-task.

---

## 6. TOOLBOX CONVENTIONS

- **MNE-Python + MNE-NIRS** — the de-facto Python stack. MNE-NIRS adds GLM, Beer-Lambert conversion, scalp-coupling-index QC, short-channel handling. **Decoding module = it reuses MNE's `mne.decoding` (`Scaler`, `Vectorizer`, `SlidingEstimator`) + scikit-learn.** Canonical decoding tutorial (`plot_50_decoding.html`): `make_pipeline(Scaler(...), Vectorizer(), LogisticRegression(solver='liblinear'))`, 5-fold CV, ROC-AUC, on epoched HbO/HbR. **This is your ready-made Baseline #2.** GLM route (`plot_14_glm_components`) is for activation stats, not single-trial decode.
- **scikit-learn pipelines are the standard** for the classical route: `StandardScaler → LDA/SVM` on the hand-feature matrix.
- **Other toolboxes:** Homer3 (MATLAB, preprocessing/GLM, not a decoder), NIRS-KIT & NIRS Brain AnalyzIR (MATLAB, stats/connectivity), Artinis Oxysoft. None ship a canonical single-trial ML decoder — the field converges on **MNE-NIRS + scikit-learn** for decoding.
- pyRiemann/MOABB exist but are EEG-oriented; using them on fNIRS means the covariance anti-pattern of §3 — skip for the baseline.

---

## References (URLs)

1. Noori, Naseer et al. 2016, *Optimal Feature-Combination for LDA of fNIRS* — https://pmc.ncbi.nlm.nih.gov/articles/PMC4879140/ (mean+peak 93% HbO)
2. Kwon & Benerradi, *Benchmarking framework for ML classification from fNIRS data* 2024 — https://pmc.ncbi.nlm.nih.gov/articles/PMC10790918/ (mean/std/slope features; LDA≈CNN generalized 55–60%)
3. MNE-NIRS Decoding Analysis tutorial — https://mne.tools/mne-nirs/stable/auto_examples/general/plot_50_decoding.html (LogReg pipeline, HbO 89% ROC-AUC)
4. Hong et al., *Feature Extraction & Classification for Hybrid fNIRS-EEG BCI* 2018 — https://pmc.ncbi.nlm.nih.gov/articles/PMC6032997/ (feature list: mean/peak/slope/latency/skew/kurtosis/PSD)
5. Shin et al. 2018, *Simultaneous EEG-NIRS open dataset* (Scientific Data) — https://www.nature.com/articles/sdata20183 ; data: https://doc.ml.tu-berlin.de/simultaneous_EEG_NIRS/
6. Huang et al., *Tufts fNIRS2MW Dataset & Benchmark* (NeurIPS 2021 D&B) — https://tufts-hci-lab.github.io/code_and_datasets/fNIRS2MW.html ; paper https://openreview.net/forum?id=QzNHE7QHhut ; code https://github.com/tufts-ml/fNIRS-mental-workload-classifiers
7. *Block-as-Domain Adaptation for Workload Prediction from fNIRS* 2024 (fNIRS2MW baselines: DeepConv 63.75%, EEGNet 62.08%) — https://ar5iv.labs.arxiv.org/html/2405.00213v1
8. Wang, Zhang et al., *Transformer Model for fNIRS Classification (fNIRS-T)* IEEE JBHI 2022 — https://ieeexplore.ieee.org/document/9670659/ ; code https://github.com/wzhlearning/fNIRS-Transformer
9. *fNIRSNet* (delayed-HRF domain knowledge, 498 params) — via https://arxiv.org/html/2402.15266v2 (calibration study) ; review https://arxiv.org/pdf/2201.13371 (*Deep Learning in fNIRS: A review*)
10. Wickramaratne & Mahmud, CNN on GASF, 87.14% mental arithmetic — via review refs above
11. Yang et al., *Domain adaptation for cross-subject/session workload alignment (fNIRS)* — https://pmc.ncbi.nlm.nih.gov/articles/PMC7790507/
12. Nguyen et al. 2025, *Riemannian geometry boosts fNIRS classification* (covariance anti-pattern, n=7 caveats) — https://pmc.ncbi.nlm.nih.gov/articles/PMC12523035/
13. Aghajani, Omurtag et al., SVM n-back cognitive load 74.8% — via EEG+fNIRS workload review https://pmc.ncbi.nlm.nih.gov/articles/PMC9571712/

### Unverified / to-confirm flags
- Tufts **subject-specific** (not generic) accuracy numbers — not fetched from source PDF (PDF exceeded fetch size). Only cross-subject ~62–68% confirmed via secondary citation. Expect subject-specific ~70–80%.
- fNIRS-T exact per-dataset accuracy table — LOSO ~90% quoted secondhand, not read from primary table.
- "91.31% mean+slope" workload figure — from lit-review summary, primary source not individually opened.
