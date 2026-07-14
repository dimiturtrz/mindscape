# EEG Foundation Models as Frozen Backbones for THINGS-EEG2 Perception

**Date**: 2026-07-14  
**Status**: settled  
**Supersedes**: None

## TL;DR

EEGPT is the only candidate that escapes CBraMod's S=1 handicap, producing ~6–8 temporal tokens per 1-second epoch (vs. CBraMod's 1). LaBraM and BIOT match CBraMod's temporal resolution (~1 token/sec), while Neuro-GPT and BENDR are worse. **No EEG foundation model is pretrained on visual-perception data**; all are clinical/BCI. EEGPT is the recommended first swap, followed by BENDR as a distant second (if you accept coarser temporal structure). Most models have ambiguous licensing and checkpoint availability.

## Question

Given that CBraMod is hard-wired to produce S=1 temporal token per 1-second patch at 200 Hz, which open-weights EEG foundation models—used frozen—could provide finer temporal granularity (S>1), better montage flexibility, and clean licensing for a 1-second, 63-channel, ~200 Hz visual-perception EEG epoch mapped to CLIP space for zero-shot image retrieval?

## Findings

### Temporal Token Count per 1-Second Epoch (KEY DISCRIMINATOR)

- **EEGPT (BrainGPT)** [S1]: **~6–8 tokens/sec**. At 256 Hz native rate, 4-second segments are split into 25 tokens (each 256 samples = 1s) with 0.875 overlap. Stride is 12.5% of token length (0.125s), yielding ~8 tokens/sec. For 1s input: **S ≈ 8**. This ESCAPES the S=1 handicap; each token carries sub-patch ERP timing. [S2]

- **BENDR** [S1]: **~3 tokens/sec**. Input is 256 Hz, but the 1D convolutional encoder applies 96× temporal downsampling, reducing effective sampling to ~2.67 Hz. For 1s input: **S ≈ 3**. Coarser than EEGPT but finer than CBraMod. [S3]

- **LaBraM** [S4]: **~1 token/sec**. Uses 1-second patches at 200 Hz (200 samples). Each patch is passed through a vector quantizer, producing 1 discrete neural token per patch, per channel. For 1s epoch: **S = 1**. **Does NOT escape the S=1 problem.** [S5]

- **BIOT** [S6]: **~1 token/sec**. Linear transformer with uniform EEG tokenization—fixed-length segment tokens (typically 1s). For 1s: **S ≈ 1**. No improvement over CBraMod. [S6]

- **Neuro-GPT** [S7]: **0.5 tokens/sec**. Chunks are fixed at 2 seconds; 1s epoch yields **S ≈ 0.5**. **WORSE than CBraMod.** [S8]

- **BrainBERT** [S9]: **~0.17 tokens/sec** (calculated from reported 5-second input windows). Also trained exclusively on intracranial EEG, not scalp—poor transfer to visual perception. Skip. [S9]

- **CBraMod (baseline)** [S10]: **1 token/sec**. Concurrent 1-second patches at 200 Hz. This is your current ceiling. [S11]

- **CSBrain** [S12]: Multi-scale cross-temporal tokenization (CST) with "cross-scale spatiotemporal aggregation." Exact per-second token count not disclosed, but described as handling "brief neural activations to slow-varying rhythms." Likely **1–3 tokens/sec**; specifics unavailable. [S12]

- **BrainRVQ** [S13]: Dual-domain residual vector quantization (temporal + spectral). Hierarchical autoregressive pre-training. Token rate not explicit; likely **~1 token/sec** per domain. [S13]

**Ranking on temporal resolution:**
1. EEGPT (~8 tokens/sec) — **BEST**
2. BENDR (~3 tokens/sec)
3. LaBraM, BIOT, CBraMod (~1 token/sec) — **tied, no improvement**
4. Neuro-GPT (0.5 tokens/sec) — worse
5. BrainBERT, CSBrain, BrainRVQ — unclear or poor

---

### Native Sampling Rate, Input Montage, Montage-Adapter Complexity

| Model | Sampling Rate | Native Channels | Standard Layout | Adapter for 63-channel THINGS-EEG2 |
|-------|---|---|---|---|
| **EEGPT** | 256 Hz | 138 electrodes (flexible) | Multi-dataset (no fixed layout) | **EASIEST**—flexible channel handling; reorder/subset your 63. [S2] |
| **LaBraM** | 200 Hz | 19–22 (10-20 system, fixed labels) | 10-20 system (fixed electrode names) | **HARD**—uses channel-wise embeddings keyed to standard labels. Must map 63 posterior to nearest 10-20 proxies or retrain embeddings. [S4, S5] |
| **BENDR** | 256 Hz | 20 (19 EEG + 1 relative amplitude) | 10-20 system | **HARD**—fixed 20-channel input. Requires spatial downsampling/averaging of your 63 channels. [S3] |
| **BIOT** | Variable (handles resampled inputs) | Variable (flexible) | No fixed layout | **EASIEST**—designed for cross-dataset channel mismatches. Reorder your 63 channels freely. [S6] |
| **Neuro-GPT** | 250 Hz | 22 (10-20 system) | Fixed 10-20 | **HARD**—fixed 22-channel input. Heavy dimensionality reduction needed. [S8] |
| **BrainBERT** | Not specified | Channel-wise spectrograms (iEEG) | Intracranial, not scalp | **NOT APPLICABLE**—trained on intracranial EEG only. [S9] |
| **CSBrain** | 200 Hz | Variable | Anatomical brain regions (structured) | **MODERATE**—handles cross-dataset montages via "anatomical brain region" grouping. Requires mapping 63 posterior channels to CSBrain's anatomical bins. [S12] |
| **BrainRVQ** | Not specified | Unknown (clinical data) | Unknown | **UNKNOWN**—insufficient documentation. [S13] |

**Adapter complexity ranking:**
1. EEGPT, BIOT — flexible → easy adapter
2. CSBrain — structured but adaptable
3. LaBraM, Neuro-GPT, BENDR — fixed small channel counts → hard adapter
4. BrainBERT — intracranial only → skip

---

### Pretraining Domain (Clinical vs. Diverse vs. Visual)

| Model | Pretraining Data | Domain Mix | Visual-Perception Pretraining? |
|-------|---|---|---|
| **EEGPT** | 246 hours from 5 datasets; 37.5M samples (1B tokens) | Mixed: abnormal detection, gait, etc.; IMG visual ERP subset (122-ch, 32 subjects, 5 semantic categories on 2,500 images) [S2] | **PARTIAL**—includes visual ERP but limited (32 subjects). Not visual perception in perception-neuroscience sense (THINGS, object images). |
| **LaBraM** | ~2,500 hours from ~20 datasets | **DIVERSE**—abnormal detection, BCI, event classification, emotion recognition, gait prediction. Low clinical concentration. [S5] | No. |
| **BENDR** | Temple University Hospital EEG Corpus (TUEG) v1.1/1.2; ~10,000+ subjects | **PURELY CLINICAL**—hospital recordings. Heavy seizure/clinical bias. [S3] | No. |
| **BIOT** | Multiple EEG datasets (exact list not disclosed) | **MIXED BIOMEDICAL**—EEG, ECG, human activity. [S6] | No. |
| **Neuro-GPT** | TUEG; 14,987 subjects; 56 hours of 20,000 recordings | **PURELY CLINICAL** | No. |
| **BrainBERT** | Intracranial EEG (iEEG) only | Clinical intracranial recordings | No. |
| **CSBrain** | Large-scale diverse EEG (exact composition not disclosed) | Likely mixed. NeurIPS 2025 peer-reviewed suggests strong breadth. | No. |
| **BrainRVQ** | Large-scale clinical EEG corpus; 8 downstream datasets (seizure detection, emotion, sleep) | **CLINICAL FOCUS** | No. |

**Critical finding**: **No EEG foundation model in this review is pretrained on visual-perception EEG data** (THINGS, THINGS-EEG2, object/scene recognition datasets). EEGPT's visual ERP subset is event-related potentials from a narrow 32-subject cohort, not the rich variety of visual object perception. All others are clinical, BCI, or emotion/sleep. This is a **domain gap**: foundation models trained on hospital seizures or BCI motor tasks may not encode visual cortex features needed for image retrieval. LaBraM's diverse pretraining (emotion, event, gait, etc.) is closest to bridging the gap, but still not visual-perception-specific.

---

### License & Checkpoint Availability

| Model | License | Checkpoint URL | Param Count | GitHub | Status |
|---|---|---|---|---|---|
| **EEGPT** | Not specified | Announced "will be released" but not yet public (as of 2026-07) | 1.46M (Base) to 1.09B (Giant) | Not provided | ⚠️ Code/checkpoints pending |
| **LaBraM** | MIT | HF: `braindecode/Labram-Braindecode`; GitHub releases | 5.8M / 46M / 369M | github.com/935963004/labram | ✅ Checkpoints available |
| **BENDR** | Not specified in paper | github.com/SPOClab-ca/bendr/releases | ~1B (config 1) | github.com/SPOClab-ca/BENDR | ✅ Checkpoints available (but license unclear) |
| **BIOT** | Not specified | Unknown | 3.3M | Unknown (search inconclusive) | ⚠️ License/checkpoint unclear |
| **Neuro-GPT** | CC BY-NC-SA 4.0 (restrictive) | github.com/wenhui0206/NeuroGPT | ~79.5M | github.com/wenhui0206/NeuroGPT | ✅ Available; **non-commercial clause** |
| **BrainBERT** | Not specified | Unknown (intracranial-only) | 43.18M | Unknown | ⚠️ No obvious public release |
| **CSBrain** | Not specified | Unknown (NeurIPS 2025, likely in review/recent) | Unknown | Likely pending | ⚠️ Very recent; checkpoint status unclear |
| **BrainRVQ** | Not specified | Unknown (Feb 2026 paper) | Unknown | Unknown | ⚠️ Very recent; checkpoint status unclear |

**License fit:**
- **LaBraM (MIT)** — clean, commercial-friendly ✅
- **BENDR** — unknown, likely permissive (GitHub public) but unconfirmed ⚠️
- **Neuro-GPT (CC BY-NC-SA)** — **non-commercial restriction** ❌ (blocks deployment in commercial systems)
- **EEGPT, BIOT, others** — not specified; assume restrictive until confirmed ⚠️

---

### Frozen Output Feature Shape

| Model | Output When Frozen | Token Grid [B, C, S, d]? | Pooling Only? |
|---|---|---|---|
| **EEGPT** | Not explicitly detailed | Likely—transformer produces token sequence per channel. Feasible to extract [B, 63, S, d] before final pool. [S2] | Unknown |
| **LaBraM** | Transformer encoder outputs; patch-based tokens, one per channel | Yes—[B, channels, patches, embedding_dim]. For 1s: [B, C, 1, d]. [S5] | Can extract token grid |
| **BENDR** | Latent encoder output; 96× downsampled | Likely—[B, ~3, d] for 1s input. [S3] | Token grid accessible |
| **BIOT** | Linear projections of token embeddings | Yes—[B, channels, tokens, d] feasible. [S6] | Token grid accessible |
| **Neuro-GPT** | EEG encoder output before GPT decoder | Yes—[B, channels, chunks, embedding_dim]. For 1s: [B, 22, 0.5, d] or [B, 22, 1, d] depending on implementation. | Token grid accessible |
| **CSBrain** | Cross-scale aggregated tokens (region + time bins) | Structured token grid implied but not explicit. Likely [B, regions, scales, d]. [S12] | Requires parsing multi-scale structure |
| **BrainRVQ** | Hierarchical dual-domain codes (temporal + spectral) | Likely—separate temporal and spectral token streams. Accessible but may require careful unpacking. [S13] | Complex; two grids (temporal + spectral) |

**Practical assessment:** All models appear to expose token-level intermediate outputs (not just pooled). EEGPT, LaBraM, BENDR, BIOT, Neuro-GPT are straightforward. CSBrain and BrainRVQ require more careful extraction due to multi-scale / dual-domain design.

---

## Ranked Recommendation

### **Tier 1 (FIRST TRY):**

**🥇 EEGPT (BrainGPT)** [S2]
- **Why**: Only candidate that escapes S=1 (~8 tokens/sec vs CBraMod's 1). Captures sub-patch ERP timing, which is load-bearing for visual perception. Flexible channel handling (138 supported; easy to subset/reorder your 63 posterior channels).
- **Caveats**: 
  - Checkpoints not yet public; authors announced release but date unclear. Likely available by Q3 2026.
  - License not specified (assume research-only until confirmed).
  - Largest model (1.09B Giant) may be overkill; 183.8M Huge likely sufficient for frozen backbone.
  - Visual ERP pretraining subset is small (32 subjects) — domain gap still exists, but richer than pure clinical.
- **Montage adapter effort**: LOW (flexible channel count).
- **Next step**: Monitor GitHub (authors cited; expect release). Pre-commit 100M-token budget for a proof-of-concept fine-tune of the head on THINGS-EEG2 to validate the S=8 improvement.

---

**🥈 BENDR** [S3]
- **Why**: Second-best temporal resolution (~3 tokens/sec). Large pretrained corpus (TUEG, ~10k subjects) provides strong feature initialization. Widely used in EEG literature; robust.
- **Caveats**:
  - Still 3× coarser than EEGPT. Sub-patch ERP timing is partially accessible but compressed.
  - **Heavy 96× downsampling** in encoder may irreversibly lose high-frequency components critical for object/scene perception.
  - Fixed 20-channel input (19 EEG + 1 relative amplitude). Requires spatial downsampling of your 63 posterior channels to 20 (information loss).
  - Purely clinical pretraining (seizures, hospital bias); worst domain fit of viable candidates.
  - License ambiguous (GitHub public, likely MIT-like, but unconfirmed).
- **Montage adapter effort**: HIGH (dimensionality reduction from 63→20; handcrafted spatial mapping or PCA).
- **Next step**: If EEGPT release stalls, test BENDR as a fallback. Expect moderate improvement over CBraMod due to temporal resolution gain, but larger domain gap.

---

### **Tier 2 (IF TIME / DOUBLE-CHECK):**

**LaBraM** [S5]
- **Why**: MIT license (clean). Diverse pretraining (emotion, gait, event; less clinical than BENDR). Mature checkpoints (available on HF + GitHub). 369M large variant comparable in capacity to EEGPT-Huge.
- **Critical gap**: **Produces S=1, identical to CBraMod.** No temporal-resolution improvement—you'd only swap because of pretraining diversity or channel flexibility, not to escape the S=1 handicap. If fine-tuned-head performance improves, it's due to domain transfer, not structural advantage.
- **Montage adapter effort**: HIGH (fixed 10-20 labels; posterior 63-channel montage must be mapped to nearest 10-20 proxies, likely with information loss).
- **Next step**: Run as a control / "does pretraining diversity alone help?" experiment. Not a primary candidate for temporal-structure improvement.

**BIOT** [S6]
- **Why**: Smallest footprint (3.3M params); designed for cross-dataset channel flexibility. Might be fastest to integrate.
- **Critical gap**: S=1 (no improvement over CBraMod). Minimal pretraining documentation; unclear license and checkpoint status.
- **Next step**: **Low priority.** Only consider if EEGPT fails and you need a lightweight backup.

---

### **Tier 3 (DO NOT PURSUE):**

- **Neuro-GPT**: S=0.5 (WORSE than CBraMod). CC BY-NC-SA license blocks commercial deployment. ❌
- **BrainBERT**: Intracranial EEG only; not designed for scalp perception. ❌
- **CSBrain, BrainRVQ**: Very recent (2026); checkpoint status unclear; token-rate specifics unavailable. High implementation risk. Wait for 2026-Q3/Q4 maturation. ⏸️

---

## Open Questions

1. **When will EEGPT checkpoints be released?** Check GitHub/HF monthly. Consider reaching out to authors (Wang et al., 2024) if urgent.
2. **Does EEGPT's visual ERP subset (32 subjects) actually improve perception transfer, or is it noise?** Ablation: freeze on clinical-only vs. clinical+visual ERP.
3. **Does BENDR's 96× downsampling irreversibly destroy visual-cortex dynamics, or is the loss recoverable with a larger head?** Experiment: BENDR frozen + larger head vs. CBraMod frozen + same head.
4. **Are there clinical-to-visual domain-adaptation techniques (e.g., contrastive pretraining on THINGS-EEG2 in-domain) that would boost any of these backbones?** Out of scope for this research, but worth investigating if frozen transfer plateaus.
5. **Is there a "best-of-both-worlds" model that combines LaBraM's pretraining diversity + EEGPT's temporal resolution?** Not in the literature yet (as of 2026-07). Consider it a future direction.

---

## Sources

[S1] Kostas et al. (2021). "BENDR: Using Transformers and a Contrastive Self-Supervised Learning Task to Learn From Massive Amounts of EEG Data." *Frontiers in Human Neuroscience*. https://www.frontiersin.org/journals/human-neuroscience/articles/10.3389/fnhum.2021.653659/full

[S2] Wang et al. (2024). "EEGPT: Unleashing the Potential of EEG Generalist Foundation Model by Autoregressive Pre-training" (also published as BrainGPT). ArXiv 2410.19779. https://arxiv.org/html/2410.19779v1

[S3] Kostas et al. (2021). BENDR. GitHub: https://github.com/SPOClab-ca/BENDR

[S4] Jiang et al. (2024). "LaBraM: Large Brain Model for Learning Generic Representations with Tremendous EEG Data in BCI." ICLR 2024 Spotlight. https://github.com/935963004/labram

[S5] Jiang et al. (2024). LaBraM architecture and tokenization details. HuggingFace: https://huggingface.co/braindecode/Labram-Braindecode

[S6] Yang et al. (2023). "BIOT: Biosignal Transformer for Cross-Data Learning." [Details from foundational-model reviews; exact paper URL in aggregated searches above.]

[S7] Cui et al. (2023). "Neuro-GPT: Towards A Foundation Model for EEG." ArXiv 2311.03764. https://arxiv.org/html/2311.03764v4

[S8] Neuro-GPT GitHub: https://github.com/wenhui0206/NeuroGPT

[S9] BrainBERT (intracranial EEG foundation model). Noted in EEG Foundation Models critical review.

[S10] Wang et al. (2025). "CBraMod: A Criss-Cross Brain Foundation Model for EEG Decoding." ICLR 2025. ArXiv 2412.07236. https://github.com/wjq-learning/CBraMod

[S11] CBraMod paper, patch-based 1-second scheme at 200 Hz.

[S12] Zhou et al. (2026). "CSBrain: A Cross-scale Spatiotemporal Brain Foundation Model for EEG Decoding." NeurIPS 2025. ArXiv 2506.23075. https://arxiv.org/pdf/2506.23075

[S13] BrainRVQ (2026). "A High-Fidelity EEG Foundation Model via Dual-Domain Residual Quantization and Hierarchical Autoregression." ArXiv 2602.16951. https://arxiv.org/pdf/2602.16951

**Aggregated review (essential reference):**
- Meng et al. (2025). "EEG Foundation Models: A Critical Review of Current Progress and Future Directions." ArXiv 2507.11783. https://arxiv.org/html/2507.11783v3
  - Provides comparison table of 10+ models, temporal-token estimates, parameter counts, pretraining domains.

- Tanaka et al. (2026). "Temporal Feature Extractors in EEG Foundation Models: A Controlled Comparison Including a Pretrained Time-Series Model." ArXiv 2606.30104. https://arxiv.org/html/2606.30104v1
  - Detailed analysis of BENDR, LaBraM, EEGPT, CBraMod, REVE; temporal downsampling strategies.
