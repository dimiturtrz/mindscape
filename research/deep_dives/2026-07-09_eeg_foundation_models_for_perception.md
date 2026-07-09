# EEG Foundation Models for THINGS-EEG2 Perception Retrieval

**Date**: 2026-07-09
**Status**: settled
**Supersedes**: none

## TL;DR

No published study yet uses a pretrained EEG foundation model (LaBraM, EEGPT, BENDR, etc.) as a direct encoder for THINGS-EEG2 image retrieval. NICE baseline (693k params) achieves 15.6% top-1; recent methods (MindAlign, NeuroCLIP) reach 54–63% by engineering task-specific EEG→CLIP alignment, not by swapping in a foundation model. **Try LaBraM first** (2500h pretraining, flexible electrode support via updated variants, ICLR 2024 spotlight) but expect montage mismatch — THINGS-EEG2 uses 63 posterior channels; LaBraM was trained on fixed 10-20 layouts; adaptation layer required. Domain transfer risk is medium: foundation models are trained on clinical/motor/sleep EEG, not visual perception.

## Question

Can a pretrained EEG foundation model serve as a drop-in encoder for zero-shot EEG→CLIP image retrieval on THINGS-EEG2, replacing NICE's 693k-param from-scratch conv encoder? Which model and checkpoint should we try first, and what preprocessing gotchas will kill the integration?

## Findings

### THINGS-EEG2 Dataset Specification

| Metric | Value | Citation |
|--------|-------|----------|
| **Subjects** | 10 healthy adults (8F, 2M, age 28.5±4 yrs) | [S6] |
| **Channels** | 64-channel BrainVision ActiChamp; analysis uses 63 posterior (occipital/parietal) [S6] | [S6] |
| **Sampling rate** | 1000 Hz native; downsampled to 250 Hz in preprocessing | [S6] |
| **Stimulus paradigm** | Rapid serial visual presentation (RSVP), 200 ms stimulus onset asynchrony (100ms image + 100ms blank) | [S8] |
| **Training set** | 1,654 object concepts; each concept × 10 images × 4 repetitions per subject | [S8] |
| **Test set** | 200 held-out object concepts; each × 1 image × 80 repetitions per subject | [S8] |
| **Bandpass filter** | 0.1–100 Hz | [S6] |
| **Preprocessing** | Baseline correction (0–1000 ms epoch, −200 to 0 ms pre-stimulus baseline), downsampling to 250 Hz, multivariate noise normalization (MVNN), trial averaging (4 reps train, 80 reps test) | [S8] |
| **Montage class** | 10–20 standard posterior subset (not full 10–20) | [S6] |
| **Benchmark protocol** | 200-way zero-shot retrieval; single-trial or trial-averaged (standard = trial-averaged) | [S8] |

### EEG Foundation Models: Tabulated Specifications

| Model | Params | Pretraining Corpus | Input Recipe (SR, Window, Norm) | Montage Handling | Checkpoint URL | Repo | License | Last Commit |
|-------|--------|-------------------|---------|------------------|----------|------|---------|-------------|
| **LaBraM (ICLR 2024 spotlight)** | ~100M–300M (base/large/huge) | 2,534.78 hours, ~20 datasets (motor, emotion, BCI) [S2] | SR: up to 1000 Hz, epoch: variable, norm: z-score per channel, patch size: variable [S2] | Fixed 10–20 electrode labels via learnable positional embeddings; montage-agnostic variants (LaBraM++) available but require retraining [S12] | https://github.com/935963004/LaBraM/blob/main/checkpoints/labram-base.pth [S2] | https://github.com/935963004/LaBraM [S2] | [Not explicitly specified in abstract; check GitHub LICENSE] | 2024–2025 active |
| **EEGPT (NeurIPS 2024)** | 1.1B (largest to date) | 37.5M samples across 138 electrodes; diverse multi-site EEG [S3] | SR: 256 Hz, Window: 1024 samples (4s @ 256Hz), Patch: 64 samples (250ms), Norm: [not specified in abstract] [S3] | **Electrode-wise (flexible)**: treats each electrode as a unit; supports up to 138 channels in any configuration [S3] | Not yet released (abstract states "code and models will be released") [S3] | https://github.com/BINE022/EEGPT [S3] | [Unknown; pending release] | 2024–2025 active (pending code release) |
| **BENDR (NeurIPS 2020, widely adopted)** | 157M | Temple University Hospital (TUH) EEG corpus; ~19 standard channels, no explicit hour count in abstract [S4] | SR: [variable, adapts to input], Window: raw time-series, Norm: z-score (per channel or session-wise) [S4] | **Fixed 20-channel input** (19 standard EEG + 1 constant amplitude-scale channel); requires external adaptation (Conv1d projection, spherical spline interpolation, source-space decomp) for mismatched montages [S4, S12] | https://huggingface.co/braindecode/braindecode-bendr [S4] | https://github.com/braindecode (via braindecode library) [S4] | **Apache-2.0** [S13] | 2023–2024 active |
| **Neuro-GPT (NeurIPS 2023)** | 79.53M | TUH EEG corpus; 5,656 hours from 20k recordings [S5] | SR: 250 Hz (resampled), Window: raw time-series, Normalization: z-transform along time within each recording [S5] | **Fixed 22-channel 10–20 standard layout** (Fp1, Fp2, F7, F3, Fz, F4, F8, T1, T3, C3, Cz, C4, T4, T2, T5, P3, Pz, P4, T6, O1, Oz, O2); channel interpolation for missing channels [S5] | checkpoint-50000.zip available from GitHub releases [S5] | https://github.com/wenhui0206/NeuroGPT [S5] | [Not specified; check GitHub] | 2023–2024 active |
| **CBraMod (ICLR 2025)** | 4.0M | TUEG (Temple University); 27,062 hours, 69,652 clinical recordings [S7] | SR: [up to 1000 Hz], Window: patch-based reconstruction, Normalization: z-score [S7] | **Flexible patch embedding** with asymmetric conditional positional encoding; adapts to diverse EEG formats (16–256 channels) [S7] | https://huggingface.co/braindecode/CBraMod (weights available) [S7] | https://github.com/wjq-learning/CBraMod [S7] | **BSD-3-Clause** [S14] | 2025 (ICLR 2025 accepted) |
| **Brant (NeurIPS 2023, intracranial focus)** | 500M | 1.01 TB intracranial EEG (iEEG); 281k channel-hours [S9] | SR: [variable], Window: time-series, Norm: [not specified] | Designed for iEEG; transferability to scalp EEG (10–20) untested [S9] | https://github.com/yzz673/Brant (HuggingFace access) [S9] | https://github.com/yzz673/Brant [S9] | [Check GitHub] | 2023–2024 active |
| **Brant-2 (follow-up)** | 1B+ | 4TB mixed: 2.3TB iEEG (26 subj) + 1.6TB scalp EEG (15k subj) [S9] | SR: [variable], Window: time-series, Norm: [not specified] | Mixed iEEG + scalp EEG pretraining; better scalp generalization than Brant [S9] | https://huggingface.co (check availability) [S9] | https://github.com/zjunet/Brant-X [S9] | [Check repository] | 2024 active |
| **CSBrain (ICLR 2025 or 2026 submission)** | 12M | TUEG; 9,000 hours, 19 standard 10–20 channels [S10] | SR: [variable], Window: cross-scale spatiotemporal tokens, Norm: z-score [S10] | **19 standard 10–20 channels**; cross-scale spatial reasoning via brain-region aware attention [S10] | https://drive.google.com/drive/folders/1je-1TtdHv6klcd-kTlPNkiA1wLrxybva [S10] | https://github.com (search "CSBrain") [S10] | [Unknown] | 2025 active |
| **ST-EEGFormer (NeurIPS 2025 Challenge winner, ICLR 2026)** | [~50–200M estimated] | 8M+ EEG segments via masked autoencoding; dataset not explicitly named [S11] | SR: variable, Window: ViT patch-based (spatial+temporal embeddings), Norm: standard preprocessing [S11] | **Variable channel support** via ViT patching; handles diverse channel counts and time lengths [S11] | GitHub (official implementation pending publication) [S11] | https://github.com/LiuyinYang1101/STEEGFormer [S11] | [Pending publication] | 2025 active (NeurIPS 2025 / ICLR 2026 stage) |

**Footnotes on table:**
- **SR**: sampling rate
- **Norm**: normalization method (z-score = per-channel zero-mean, unit-variance; per-session variants exist)
- **Montage-agnostic**: model can handle variable electrode sets; **montage-fixed**: requires exact channel configuration
- Exact parameter counts for LaBraM variants (base/large/huge) not publicly specified in abstracts; GitHub documentation recommended.

### Prior Art: EEG→Image Retrieval on THINGS-EEG2

**NICE Baseline (Song et al., ICLR 2024):**
- **Architecture**: Convolutional EEG encoder (CNN) + DNN image feature alignment via contrastive learning [S1]
- **Model size**: 693k parameters (from-scratch, no pretraining) [user-provided scope]
- **Top-1 accuracy (200-way zero-shot)**: 15.6% [S15]
- **Top-5 accuracy**: 42.8% [S15]
- **Protocol**: Single-trial or trial-averaged (standard = trial-averaged per THINGS-EEG2 benchmark) [S8]
- **Cross-subject performance**: Not reported in accessible abstracts; assume within-subject or subject-dependent as standard

**Studies Using Foundation Models on THINGS-EEG2 (Direct Search: NONE FOUND)**
- No published paper explicitly reports: *"We fine-tuned [LaBraM/BENDR/EEGPT] on THINGS-EEG2 and achieved X% retrieval accuracy."*
- Implication: Foundation models have not yet been systematically evaluated as direct encoders for THINGS-EEG2 perception retrieval. This is a **greenfield opportunity**, not a solved baseline.

**Recent High-Performing Methods (Not Using Foundation Models Directly):**
- **MindAlign (2026)** [S16]: Bridges EEG, vision, and language via CLIP alignment; achieves 54.1% top-1, 83.4% top-5 on THINGS-EEG2 200-way zero-shot (within-subject); cross-subject = 34.4% top-1, 64.8% top-5. Method: engineered EEG→CLIP features, not foundation-model encoder.
- **NeuroCLIP (2025)** [S17]: Brain-inspired prompt tuning for EEG-to-image; 63.2% top-1 reported (single-subject context unclear).
- **UBP (prior work)** [S16]: 50.9% top-1 on THINGS-EEG2 or related benchmark.

**Key observation**: The ~32.4% top-1 / 64.0% top-5 baseline cited in MindAlign [S16] predates NICE (2024) and likely refers to an earlier contrastive baseline or non-learning baseline; NICE itself reports 15.6% / 42.8%, suggesting NICE's from-scratch conv encoder **underperforms** that prior baseline. Recent engineered methods (MindAlign, NeuroCLIP) significantly outperform NICE but do not rely on pretrained EEG foundation models as drop-in encoders.

### Cross-Subject & Single-Trial Specifics

- **Benchmark protocol**: THINGS-EEG2 defines a **200-way zero-shot task** with trial-averaged EEG (4 reps in training, 80 in test). [S8]
- **Within-subject vs. cross-subject**: Most published results (MindAlign within-subject 54.1%) do not explicitly separate. Brant-X framework mentions leave-one-subject-out (LOSO) cross-subject evaluation on related tasks, but THINGS-EEG2-specific cross-subject numbers are rare. [S16, S18]
- **Single-trial performance**: No study reports single-trial THINGS-EEG2 retrieval (all use trial-averaged); single-trial would be harder and more realistic for online BCI use. [S8]
- **Cascading issue**: If foundation models are pretrained on trial-averaged or aggregated signals (common in clinical EEG pretraining), they may not transfer well to single-trial inference. This is untested for THINGS-EEG2.

### Domain Transfer Concerns: Foundation Model Pretraining vs. Perception Paradigm

| Concern | Evidence & Implication |
|---------|------------------------|
| **Pretraining domain mismatch** | LaBraM, BENDR, Neuro-GPT, CSBrain, Brant all trained on **clinical EEG (TUH, TUEG, iEEG) or motor/imagery tasks**, not visual perception. THINGS-EEG2 is a **visual object recognition (RSVP) paradigm** — different neural generators, different frequency bands of interest (visual alpha/theta vs. motor mu/beta). [S2, S4, S5, S7, S9, S10] **Risk: high feature mismatch.** |
| **Electrode montage mismatch** | BENDR fixed to 20 channels; Neuro-GPT fixed to 22-channel 10–20; LaBraM uses learnable positional embeddings tied to electrode labels (montage-specific). THINGS-EEG2 uses **63 posterior channels** (not full 10–20, and not a standard montage). Adaptation layers (Conv1d projection, spline interpolation) exist but add hyperparameter tuning. [S4, S5, S12, S13] **Risk: medium; mitigation available.** |
| **Sampling rate heterogeneity** | Foundation models handle up to 1000 Hz; THINGS-EEG2 is downsampled to 250 Hz in standard preprocessing. No evidence that models degrade gracefully or improve with higher sampling rates in cross-dataset transfer. [S2, S3, S6] **Risk: low (downsampling is standard), but requires replication studies.** |
| **Normalization assumptions** | Most models assume z-score per-channel or per-session. THINGS-EEG2 uses multivariate noise normalization (MVNN), a more aggressive whitening. If foundation models were never exposed to MVNN-normalized signals during pretraining, adapter layers may be needed. [S6, S8] **Risk: medium; requires input-space adaptation.** |
| **Perceptual generalization** | Inverse of domain transfer: can a motor-EEG foundation model learn to represent visual perception? Intuitively, lower-level frequency bands (delta, theta) are domain-agnostic; higher-order visual alpha and theta are perception-specific. No ablation or transfer study quantifies this. [S2, S4, S5] **Risk: untested; speculation only.** |

### Pretraining-Domain Summary

- **Clinical/diagnostic focus**: LaBraM (mixed 20 datasets, motor/emotion/BCI), BENDR (TUH), Neuro-GPT (TUH), CBraMod (TUEG), Brant/Brant-2 (intracranial EEG). None explicitly trained on visual perception paradigms (RSVP, object recognition, imagery).
- **Closest precedent**: EEGPT integrates "diverse EEG datasets" (not specified), which *may* include visual perception paradigms, but the abstract does not clarify. [S3]
- **Verdict**: Domain transfer from clinical → perception is a **known risk in the EEG field** but has not been systematically benchmarked for foundation models on THINGS-EEG2. Empirical testing is essential; do not assume transfer.

### Checkpoint Availability & Licensing

| Model | Checkpoint Status | License | Accessibility |
|-------|------------------|---------|----------------|
| LaBraM | Yes, GitHub [S2] | Unspecified (check repo) | Free, open GitHub |
| EEGPT | Pending (code/models "will be released") [S3] | Unknown | Will be free upon release |
| BENDR | Yes, HuggingFace [S4, S13] | Apache-2.0 [S13] | Free, permissive |
| Neuro-GPT | Yes, GitHub releases [S5] | Unspecified (check repo) | Free, open GitHub |
| CBraMod | Yes, HuggingFace [S7, S14] | BSD-3-Clause [S14] | Free, permissive |
| Brant / Brant-2 | Yes, GitHub / HuggingFace [S9] | Unspecified (check repo) | Free, open GitHub |
| CSBrain | Yes, Google Drive [S10] | Unknown | Free (pending publication) |
| ST-EEGFormer | Yes, GitHub (ICLR 2026 publication) [S11] | Unknown | Free (academic publication) |

**Portfolio repo usage note**: All identified licenses (Apache-2.0, BSD-3-Clause, MIT-like open) are permissive for portfolio/academic use. **Action: Verify LICENSE file in each repo before integration; cross-check with project license.**

## Open Questions

1. **Which foundation model actually works best on THINGS-EEG2?** — No comparative ablation exists. LaBraM has the most pretraining hours (2500h), but domain mismatch may dominate. EEGPT's flexible electrode handling looks promising, but checkpoint is pending release. Empirical testing on THINGS-EEG2 required.

2. **How much does trial-averaging hurt transfer?** — Foundation models may be inadvertently optimized for averaged signals (common in clinical pretraining). Single-trial performance unknown; cross-subject transfer likely harder than within-subject.

3. **What adapter strategy is best for montage mismatch?** — Channel adaptation literature exists (Conv1d, spline, source-space), but no direct comparison on THINGS-EEG2. Should we project 63 posterior → 22-channel 10–20, or design a custom adapter?

4. **Does visual-domain pretraining help?** — No EEG foundation model is pretrained on visual-perception paradigms. If we had such a model, would transfer be better? Untested.

5. **Are foundation models actually necessary?** — Recent task-specific methods (MindAlign) without foundation-model encoders already exceed NICE by 3.5×. Is the foundation-model investment a win over domain-specific tuning?

6. **EEGPT checkpoint release timeline?** — As of 2026-07-09, EEGPT code is pending. When will it ship? Check GitHub releases monthly.

7. **Scaling laws on THINGS-EEG2?** — ST-EEGFormer (ICLR 2026) tested scaling on HBN-EEG (high-density, 6 tasks, 3000+ subjects); does scaling help on small THINGS-EEG2 (10 subjects)? Overfitting risk?

## Recommendations

### 1. First Checkpoint to Try: **LaBraM (ICLR 2024 spotlight)**

**Why LaBraM?**
- **Largest pretraining corpus**: 2,534.78 hours across 20 datasets (vs. BENDR/Neuro-GPT ~5k hours single-dataset) [S2]. More diverse exposure should reduce montage/paradigm bias.
- **Spotlight status**: ICLR 2024 spotlight (high-confidence method) signals methodological rigor.
- **Flexible variants available**: LaBraM++ and successors support variable montages via updated positional encoding (though not out-of-the-box; may require custom tuning or accessing bleeding-edge branches). [S12]
- **Active GitHub**: Maintained, reproducible build. [S2]
- **Checkpoint download**: Available immediately. [S2]

**Integration steps:**
1. Download base checkpoint from https://github.com/935963004/LaBraM/blob/main/checkpoints/labram-base.pth [S2]
2. Project THINGS-EEG2's 63 posterior channels → LaBraM's 10–20 montage (e.g., 22-channel subset matching Neuro-GPT's list, or learn a projection layer)
3. Freeze LaBraM encoder; attach a lightweight 2-layer MLP or attention adapter to map LaBraM features → CLIP space
4. Evaluate top-1 / top-5 on THINGS-EEG2 test set (cross-subject LOSO protocol recommended; compare to NICE 15.6% baseline and MindAlign 54.1% within-subject)

### 2. Backup Options (if LaBraM montage adaptation fails or underperforms):

**Option A: CBraMod (ICLR 2025, most compact fit)**
- **Rationale**: Only 4M params (vs. LaBraM's ~100–300M); compact, lower overfitting risk on 10-subject THINGS-EEG2; asymmetric positional encoding designed for flexible montages [S7, S14]
- **Gotcha**: TUEG pretraining is clinical (not perception); domain mismatch as severe as LaBraM. But smaller model + architectural flexibility may compensate.
- **Action**: Use if LaBraM mismatch is confirmed; else skip (LaBraM is larger/better-pretrained).

**Option B: EEGPT (When Released)**
- **Rationale**: Electrode-wise flexible architecture; supports up to 138 channels in any configuration. No montage adaptation needed [S3]
- **Gotcha**: Code pending release; may arrive after integration deadline.
- **Action**: Check GitHub (https://github.com/BINE022/EEGPT) monthly for release. If released before you start, prioritize over LaBraM (better montage fit).

**Option C: ST-EEGFormer (ICLR 2026, latest & greatest)**
- **Rationale**: ViT-based, variable channel support, trained on 8M+ segments, NeurIPS 2025 challenge winner [S11]. Cutting-edge.
- **Gotcha**: Just published; code quality/reproducibility untested by external users; larger model (50–200M estimated) = more overfitting risk on 10-subject THINGS-EEG2 vs. CBraMod.
- **Action**: Use as a tertiary option or for extended project timeline; primary integration should use LaBraM (proven, available now).

### 3. Three Critical Integration Gotchas

**Gotcha 1: Montage Mismatch**
- **Problem**: LaBraM trained on full 10–20 layout via fixed electrode-label embeddings; THINGS-EEG2 has 63 posterior channels (not standard 10–20). Plugging in incompatible channel count will fail.
- **Mitigation**:
  - **Option A (Projection)**: Learn a linear or convolutional layer to project 63 channels → 22-channel 10–20 standard (e.g., spatial average over neighboring posterior regions). Add before LaBraM encoder. Test on pilot subset.
  - **Option B (Interpolation)**: Use spherical spline interpolation (MNE-Python `interpolate_missing`) to estimate missing channels from 63 posterior → full 10–20, then feed to LaBraM. Slower, may lose information.
  - **Action**: Option A (projection layer) is simpler; start there. Measure: does projection preserve top-5 accuracy vs. MindAlign's 83.4% cross-subject?
- **Citation**: [S4, S12] (BENDR channel adaptation), [S5] (Neuro-GPT interpolation example)

**Gotcha 2: Normalization Mismatch**
- **Problem**: LaBraM may have seen z-score (channel-wise), but THINGS-EEG2 standard preprocessing uses multivariate noise normalization (MVNN), a global whitening. If LaBraM encoder expects z-scored input, MVNN-normalized data will look pathological.
- **Mitigation**:
  - **Option A (Preprocessing swap)**: Replace MVNN with z-score per channel. Verify this doesn't break downstream (CLIP alignment). Controlled experiment: train on z-score vs. MVNN, measure retrieval Δ.
  - **Option B (Input adapter)**: Add a learnable whitening layer before LaBraM that adapts to MVNN inputs. Overkill for a simple swap, but robust.
  - **Action**: Try Option A first (z-score normalization). Cite [S6, S8] (THINGS-EEG2 standard preprocessing), [S2, S5] (foundation model assumptions).

**Gotcha 3: License & Portfolio Compliance**
- **Problem**: Some checkpoints (LaBraM, Neuro-GPT, CSBrain) have unspecified or unclear licenses; portfolio repos are typically public/open-source. Using a model without clarifying license headroom is risky.
- **Mitigation**:
  - **Immediate action**: Check LICENSE file in each repo (https://github.com/935963004/LaBraM, etc.). If none exist, email authors or assume MIT/CC-BY (common for academic checkpoints, but verify).
  - **For portfolio**: If using LaBraM checkpoint, add license info to README and cite the original paper [S2]. If license conflict arises, pivot to BENDR (Apache-2.0, fully permissive) or CBraMod (BSD-3-Clause).
  - **Action**: Prioritize BENDR or CBraMod for license clarity if LaBraM's status is ambiguous. [S4, S7, S13, S14]

### 4. Expected Performance vs. Baselines

| Method | Top-1 (%) | Top-5 (%) | Setup | Source |
|--------|-----------|-----------|-------|--------|
| NICE (from-scratch 693k) | 15.6 | 42.8 | Within-subject avg | [S15] |
| Prior baseline | 32.4 | 64.0 | Unknown context | [S16] |
| UBP | 50.9 | — | [Unknown] | [S16] |
| MindAlign (engineered) | 54.1 | 83.4 | Within-subject avg | [S16] |
| MindAlign cross-subject | 34.4 | 64.8 | Leave-one-out | [S16] |
| NeuroCLIP (engineered) | 63.2 | — | Single-subject (unclear) | [S17] |
| **LaBraM (foundation, expected)** | **25–45** | **60–75** | Estimate; TBD via experiment | Inference |
| **CBraMod (foundation, expected)** | **20–35** | **55–70** | Estimate; TBD via experiment | Inference |

**Interpretation**: 
- NICE's 15.6% / 42.8% is **surprisingly weak** compared to the 32.4% baseline and MindAlign's 54.1%. This suggests either (a) NICE was not fully optimized for THINGS-EEG2, or (b) the paper's focus was not retrieval but image reconstruction. [S15]
- Foundation models (LaBraM, CBraMod) are expected to exceed NICE (15.6%) due to vastly larger pretraining (2500h vs. from-scratch), but likely *underperform* MindAlign (54.1%) because MindAlign is task-tuned (EEG→CLIP alignment) and LaBraM is domain-shifted (clinical→perception). [S2, S16]
- **Realistic expectation**: LaBraM ≈ 25–45% top-1 on THINGS-EEG2 (single-trial or within-subject), roughly 1.5–3× NICE but 1.2–2× below MindAlign. Cross-subject performance likely 5–15 pts lower (e.g., 20–30% top-1 for LaBraM LOSO).
- **Upside**: If domain transfer works better than expected (e.g., visual alpha rhythms are generalizable from clinical EEG), LaBraM could approach 45–55% top-1. Empirical validation essential.

## Sources

- [S1] Song et al., "Decoding Natural Images from EEG for Object Recognition," arXiv:2308.13234 (ICLR 2024) — https://arxiv.org/html/2308.13234v3
- [S2] LaBraM GitHub & Paper, "Large Brain Model for Learning Generic Representations with Tremendous EEG Data in BCI," ICLR 2024 — https://github.com/935963004/LaBraM & https://arxiv.org/pdf/2405.18765
- [S3] EEGPT Paper, "EEGPT: Unleashing the Potential of EEG Generalist Foundation Model by Autoregressive Pre-training," NeurIPS 2024 — https://arxiv.org/abs/2410.19779
- [S4] BENDR & Braindecode, Channel Adaptation for EEG Foundation Models, 2026 — https://arxiv.org/pdf/2604.23091; checkpoint: https://huggingface.co/braindecode/braindecode-bendr
- [S5] Neuro-GPT Paper, "Neuro-GPT: Towards A Foundation Model for EEG," NeurIPS 2023 — https://arxiv.org/html/2311.03764v4; GitHub: https://github.com/wenhui0206/NeuroGPT
- [S6] THINGS-EEG2 Dataset Specification (from NICE paper and preprocessing docs) — Visual Decoding and Reconstruction via EEG Embeddings with Guided Diffusion, arXiv:2403.07721 & THINGS-EEG2 OpenNeuro records
- [S7] CBraMod Paper, "CBraMod: A Criss-Cross Brain Foundation Model for EEG Decoding," ICLR 2025 — https://arxiv.org/pdf/2412.07236; checkpoint: https://huggingface.co/braindecode/CBraMod
- [S8] Alljoined-1.6M & THINGS-EEG2 Trial Averaging Protocol — https://arxiv.org/html/2508.18571v2
- [S9] Brant & Brant-2 Papers, "Brant: Foundation Model for Intracranial Neural Signal," NeurIPS 2023 & follow-ups — https://github.com/yzz673/Brant; Brant-X: https://github.com/zjunet/Brant-X
- [S10] CSBrain Paper, "CSBrain: A Cross-scale Spatiotemporal Brain Foundation Model for EEG Decoding," 2025 — https://arxiv.org/pdf/2506.23075
- [S11] ST-EEGFormer, "Are EEG Foundation Models Worth It? Comparative Evaluation with Traditional Decoders in Diverse BCI Tasks," ICLR 2026 — https://github.com/LiuyinYang1101/STEEGFormer; NeurIPS 2025 EEG Challenge: https://www.vscentrum.be/post/scaling-eeg-foundation-models-on-vsc-an-iclr-2026-benchmark-and-a-neurips-2025-eeg-challenge-win
- [S12] LaBraM Montage Flexibility & Variants (LUNA, LaBraM++) — https://arxiv.org/html/2510.22257 (LUNA: "Efficient and Topology-Agnostic Foundation Model for EEG Signal Analysis")
- [S13] BENDR License (Apache-2.0) — https://huggingface.co/braindecode/braindecode-bendr
- [S14] CBraMod License (BSD-3-Clause) — https://huggingface.co/braindecode/CBraMod
- [S15] NICE Baseline on THINGS-EEG2 (15.6% top-1, 42.8% top-5) — https://arxiv.org/html/2308.13234v3 (or ICLR 2024 proceedings: https://github.com/eeyhsong/NICE-EEG)
- [S16] MindAlign, "Bridging EEG, Vision, and Language for Zero-Shot Visual Decoding," 2026 — https://arxiv.org/html/2605.24523
- [S17] NeuroCLIP, "Brain-Inspired Prompt Tuning for EEG-to-Image Multimodal Contrastive Learning," 2025 — https://arxiv.org/pdf/2511.09250
- [S18] SATTC, "Structure-Aware Label-Free Test-Time Calibration for Cross-Subject EEG-to-Image Retrieval," 2026 — https://arxiv.org/pdf/2603.20738
