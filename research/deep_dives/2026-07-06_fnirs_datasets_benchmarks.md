# fNIRS Datasets & Benchmarks Landscape

**Date**: 2026-07-06
**Status**: partial
**Supersedes**: (none — new topic)

## TL;DR

Six major open-access fNIRS datasets exist: Tufts fNIRS2MW (68 subjects, n-back, CC-By-4.0), BenchNIRS framework with 5 datasets (10–30 subjects, motor/arithmetic/workload), PhysioNet multimodal n-back+music (n=5, 44 channels, 2025), OpenNeuro ds004830 (11 subjects, attention task, 2026). Cross-subject accuracies plateau ~38–60% on workload tasks; no active public fNIRS-only competitions found. OpenNeuro/openfnirs.org are primary discovery hubs.

## Question

Which open-access fNIRS datasets exist with published benchmarks, how large are they (subjects/trials/channels), what are their reported cross-subject accuracies, and are there live competitions or benchmark challenges?

## Findings

### Tier-1 Benchmark Datasets (Established, Multi-Study)

**Tufts fNIRS Mental Workload (fNIRS2MW)** [S1, S2]
- **Size**: 68 subjects (87 collected, 68 recommended), aged 18–44 years
- **Task**: n-back working memory (0-back, 2-back, 3-back) — cognitive workload
- **Trials/data**: ~40 trials per block, ~21 min fNIRS data per subject
- **Channels**: 8 measurement channels (2 hemoglobin types × 2 optical data types × 2 forehead locations)
- **Benchmark Protocol**: Subject-specific, generic (cross-subject), and generic + fine-tuning paradigms [S2]
- **Download**: https://tufts.box.com/s/1e0831syu1evlmk9zx2pukpl3i32md6r [S2]
- **License**: CC-By-4.0 [S2]
- **Status**: Live, NeurIPS 2021 Datasets & Benchmarks venue [S1]
- **Cross-subject accuracy**: Not explicitly numerical on landing page; benchmarked under transfer learning paradigms [S2]

**BenchNIRS Framework (5-Dataset Meta-Benchmark)** [S3, S4]
- **Framework**: Standardized ML pipeline testing (LDA, SVM, kNN, ANN, CNN, LSTM) on five open-access fNIRS datasets [S3]
- **Datasets included**:

| Dataset | Authors/Year | n Subjects | Trials/Class | Channels | Task | Cross-Subject LDA | Cross-Subject CNN |
|---------|--------------|-----------|--------------|----------|------|-------------------|-------------------|
| n-back | Herff et al. 2014 | 10 | 100 | 8 HbO + 8 HbR | 1/2/3-back | 40.7%* | 36.7% |
| n-back | Shin et al. 2018 | 26 | 234 | 36 HbO + 36 HbR | 0/2/3-back | 38.9%* | 39.3%* |
| Word generation | Shin et al. 2018 | 26 | 780 | 36 HbO + 36 HbR | baseline vs. WG | 59.6%* | 56.2%* |
| Mental arithmetic | Shin et al. 2016 | 29 | 870 | 36 channels (760/850 nm) | baseline vs. MA | 59.1%* | 57.9%* |
| Motor execution | Bak et al. 2019 | 30 | 750 | 20 HbO + 20 HbR | L/R hand + foot | 51.8%* | 47.7%* |

*Asterisks indicate p<0.05 (above chance) [S3, S4].
- **Key Verdict**: On cross-subject evaluation (likely Leave-One-Subject-Out), LDA wins or matches DL; no method beats ~60% on workload tasks [S3, S4]
- **License**: Open-source framework on PyPI [S5]
- **Implications**: Predicting unseen subject data is harder than literature reports suggest; small N + within-subject inflation common [S3]

### Tier-2 Curated Multimodal Datasets

**Shin et al. 2018 (EEG + fNIRS Simultaneous)** [S6, S7]
- **Size**: 26 subjects
- **Tasks**: n-back (0/2/3-back), Discrimination-Selection Response (DSR), Word Generation (WG) — multiple cognitive loads
- **Channels**: 64-channel EEG + fNIRS (exact fNIRS channel count not specified in abstract)
- **Sessions/trials**: Multiple sessions per task
- **Modality**: Multimodal (simultaneous EEG + fNIRS for fusion research) [S6]
- **Publication**: Scientific Data, DOI 10.1038/sdata.2018.3 (February 2018) [S6]
- **Download location**: Referenced in BenchNIRS; specific URL in Scientific Data paper [S6]
- **License**: Open access (CC-By or equivalent per Scientific Data policy) [S6]
- **Use case**: Establishes n-back as canonical workload task for cross-subject evaluation (0-back baseline, 2/3-back increasing load) [S6]

**PhysioNet: Multimodal Dataset for Working Memory in Music (2025)** [S8, S9]
- **Size**: 5 subjects with complete data (11 enrolled, filtered for multimodal completeness)
- **Sessions**: 2 per subject (calming + vexing music conditions)
- **Task**: 1-back vs. 3-back n-back working memory, 16 blocks/session, 22 trials/block
- **fNIRS channels**: 44 channels (total HbO/HbR + oxygenated/deoxygenated hemoglobin measured) [S8]
- **Physiological**: EDA, ECG, PPG, respiration, EMG, skin temperature, facial expressions, behavioral (RT, accuracy) [S8]
- **Download**: https://physionet.org/content/multimodal-nback-music/1.0.0/ [S8]
- **License**: Open Data Commons Attribution License v1.0 [S8]
- **Novelty**: First fNIRS+multimodal n-back dataset investigating music effect on cognitive workload [S8]
- **Sample concern**: n=5 is very small for cross-subject validation [S8]

### Tier-3 Task-Specific Recent Datasets (2024–2026)

**OpenNeuro ds004830: Spatial Attention Decoding (Complex Scene Analysis)** [S10, S11]
- **Size**: 11 subjects (90 trials most; 1 subject 180 trials); published Feb 2026
- **Task**: Visual attention on spatial location during complex scene viewing (not workload task)
- **Channels**: fNIRS in SNIRF format (exact channel count not specified in dataset summary)
- **Data format**: SNIRF (fNIRS), .mat (behavioral responses + answers)
- **Download**: https://openneuro.org/datasets/ds004830 [S11]
- **Paper**: Frontiers in Human Neuroscience 2024 [S10]
- **Use case**: Attention/spatial decoding, not workload; demonstrates SNIRF format adoption on OpenNeuro [S11]

**Bak et al. 2019: Motor Execution Finger & Foot Tapping** [S12, S13]
- **Size**: 30 subjects
- **Task**: Overt motor execution — unilateral left/right finger tapping + foot tapping (3-way classification)
- **Channels**: 20 HbO + 20 HbR (motor cortex, 8 sources + 8 detectors)
- **Trials/class**: 750
- **Reported accuracy (within-subject LOCO)**: 70.4% ± 18.4% (SVM ternary) [S12]
- **Publication**: Electronics 2019, open-access [S12]
- **Download**: Referenced in BenchNIRS & NITRC motor dataset repository [S12]
- **Status**: Widely cited as canonical motor imagery benchmark [S12, S13]

### Repository & Discovery Hubs (No Direct Benchmarks)

**OpenNeuro (BIDS-format fNIRS)** [S14]
- ~10+ curated fNIRS datasets in SNIRF/BIDS format
- No centralized cross-subject benchmark
- Primary venue for new datasets (e.g., ds004830, complex scene analysis)

**openfnirs.org Data Repository** [S15]
- Meta-database of open-access fNIRS datasets
- Indexed by task type (motor, workload, pain, emotion, etc.)
- Emphasis on BIDS compliance for OpenNeuro migration
- Maintained by fNIRS community but noted as "historical" [S15]

**PhysioNet (Multimodal)** [S9]
- fNIRS + EDA/ECG/PPG datasets (e.g., n-back+music 2025)
- Broader physiological scope than fNIRS-only

**Zenodo** [S16]
- Hosts fNIRS datasets but no unified benchmark
- Used for supplementary data to published papers

### Active Competitions & Challenges

**Kaggle**: No dedicated fNIRS competition found (searched 2025–2026) [S17]

**Grand Challenge / DrivenData / IEEE**: No active fNIRS-specific challenge found [S17]

**BCI 2025 Spring School**: Includes fNIRS workshops and BR41N.IO Hackathon, but not a formal published-dataset competition [S18]

**Verdict**: No live public fNIRS decoding competition with leaderboard [S17, S18]

## Red Flags & Interpretation Notes

1. **Cross-Subject Workload Ceiling**: BenchNIRS reports 38.9% LDA on n-back (chance 33%), 59.1% on mental arithmetic (chance 50%) — modest signal [S3, S4]. No method found achieving >65% cross-subject on 3-class n-back without fine-tuning [S3].

2. **Within-Subject Inflation**: Bak 2019 reports 70.4% within-subject vs. likely ~50–55% cross-subject (estimated from BenchNIRS pattern) — large generalization gap [S12, S13].

3. **Sample Size Reality**: Tier-1 datasets range n=10–68; PhysioNet 2025 only n=5 complete. Cross-subject validation (LOSO or 5-fold GroupKFold) requires n≥20 for statistical power [S3, S4, S8].

4. **DL Underperformance**: CNN/LSTM match or underperform LDA on cross-subject workload tasks in BenchNIRS; deep learning advantage appears only within-subject or with domain adaptation [S3, S4].

5. **Task Ambiguity**: "Workload" decoding conflates cognitive demand (n-back level) with signal quality (artifact, subject physiology). Mental arithmetic & word generation show higher accuracies (~59%) than n-back (~39%), suggesting task-dependent difficulty [S3, S4].

## Open Questions

- **Benchmark generalization**: Do methods trained on Shin 2018 (26 subj) transfer to Tufts fNIRS2MW (68 subj)?
- **Cross-dataset validation**: Has anyone evaluated BenchNIRS models on fNIRS2MW or vice versa?
- **Temporal feature validation**: Do advanced temporal features (functional PCA, wavelets, GLM-HRF) improve cross-subject workload (n-back) beyond LDA+mean/slope/peak baseline?
- **Active competitions**: Are there time-gated Kaggle/DrivenData fNIRS challenges launching 2026–2027?
- **Recent 2025–2026 datasets**: OpenNeuro ds004830 & PhysioNet n-back+music are very recent; cross-subject benchmarking against classical datasets needed.

## Sources

- [S1] Huang, Z., Wang, L., Blaney, G., Slaughter, C., McKeon, D., Zhou, Z., Jacob, R., Hughes, M.C. (2021). "The Tufts fNIRS Mental Workload Dataset & Benchmark for Brain-Computer Interfaces that Generalize." NeurIPS Datasets & Benchmarks. https://datasets-benchmarks-proceedings.neurips.cc/paper/2021/
- [S2] Tufts HCI Lab. "The Tufts fNIRS to Mental Workload Dataset." https://tufts-hci-lab.github.io/code_and_datasets/fNIRS2MW.html
- [S3] Benerradi, J., Clos, J., Landowska, A., Valstar, M.F., Wilson, M.L. (2023). "Benchmarking framework for machine learning classification from fNIRS data." Frontiers in Neuroergonomics 4:994969. https://www.frontiersin.org/journals/neuroergonomics/articles/10.3389/fnrgo.2023.994969/full
- [S4] Benerradi, J. (2023). "Benchmarking framework for machine learning classification from fNIRS data" (PhD thesis). University of Nottingham. https://nottingham-repository.worktribe.com/output/18230969/
- [S5] BenchNIRS on PyPI. https://pypi.org/project/benchnirs/1.3.1/
- [S6] Shin, J., von Lühmann, A., Kim, D.W., et al. (2018). "Data descriptor: Simultaneous acquisition of EEG and NIRS during cognitive tasks for an open access dataset." Scientific Data 5:180003. DOI: 10.1038/sdata.2018.3
- [S7] Shin et al. (2018) dataset on Pure Korea University. https://pure.korea.ac.kr/en/publications/data-descriptor-simultaneous-acquisition-of-eeg-and-nirs-during-c/
- [S8] Khazaei, Parshi, Alam, Amin, Faghih (2025). "A Multimodal Dataset for Investigating Working Memory in Presence of Music v1.0.0." PhysioNet. https://physionet.org/content/multimodal-nback-music/1.0.0/
- [S9] PhysioNet Multimodal Index. https://physionet.org/content/?topic=multimodal
- [S10] "fNIRS dataset during complex scene analysis." Frontiers in Human Neuroscience 2024. https://www.frontiersin.org/journals/human-neuroscience/articles/10.3389/fnhum.2024.1329086/full
- [S11] OpenNeuro ds004830: "Spatial Attention Decoding using fNIRS During Complex Scene Analysis." https://openneuro.org/datasets/ds004830
- [S12] Bak, S., Park, J., Shin, J., Jeong, J. (2019). "Open-Access fNIRS Dataset for Classification of Unilateral Finger- and Foot-Tapping." Electronics 8(12):1486. https://www.mdpi.com/2079-9292/8/12/1486
- [S13] Bak et al. (2019) motor dataset on NITRC. https://www.nitrc.org/frs/?group_id=1455
- [S14] OpenNeuro Search. https://openneuro.org/search
- [S15] openfnirs Data Repository. https://openfnirs.org/data/ ; openfnirs database summary: https://fnirs.org/openfnirs-database/
- [S16] Zenodo Research. https://zenodo.org/ (fNIRS datasets hosted but not centrally indexed)
- [S17] Kaggle Competitions (active 2025–2026). https://www.kaggle.com/competitions?listOption=active — no fNIRS-specific challenge found.
- [S18] BCI 2025 Spring School & BR41N.IO. https://www.gtec.at/event/spring-school-program-2026/ — workshops listed but no formal published-data benchmark challenge.
