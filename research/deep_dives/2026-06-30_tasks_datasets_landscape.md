# Non-invasive neural decoding — tasks, datasets, benchmarks, build order

> Deep-dive · 2026-06-30 · session-anchor for mindscape Stage 0.
> Scope: the field's task taxonomy, the public dataset per task, the DCASE-equivalent
> challenge culture, difficulty ranking, and the recommended build order. All ⚠️ items from
> `docs/PLAN.md` verified against primary/strong sources (cited inline). Read-only research pass —
> no code touched.

---

## TL;DR (the decisions this note backs)

- **Start: Motor Imagery on BCI Competition IV-2a, via MOABB + Braindecode.** Standardized, cheap,
  non-trivial, has a published cross-subject gap to reproduce honestly. = Stage 0 = public-flip trigger.
- **The harness is the project, not the model.** Build the eval layer dataset-agnostic
  (`(y_true, y_pred, y_prob, subject_id)` in → accuracy + ECE + per-subject/session diagnostics out),
  decoupled from MOABB, so Stage 1 datasets feed the same eval untouched.
- **The field's own consensus backs the thesis.** NeurIPS 2025 EEG Foundation Challenge (>1000 teams)
  is literally "cross-task → cross-subject generalization" — mindscape's honest-eval angle, externally
  validated. Don't chase accuracy SOTA; contribute the OOD/calibration measurement + the efficient deploy.

---

## 1. Task taxonomy (ML-engineer framing: signal in → output → ML shape)

| # | Task | In → Out | ML shape | Key dataset | Maturity |
|---|------|----------|----------|-------------|----------|
| 1 | **Motor imagery (MI)** | 2–4 s EEG epoch → limb class | multiclass classification | BCI IV 2a/2b | mature (within-subj) |
| 2 | **P300 / ERP speller** | post-stimulus epoch → target vs not | binary detection + accumulation | BNCI, EPFL | near-solved |
| 3 | **SSVEP / cVEP** | continuous EEG over flicker → which freq | frequency classification (CCA/FBCCA→DL) | Nakanishi, Wang | mature, high bitrate |
| 4 | **Visual / semantic** | epoch @ image onset → class or CLIP embedding | contrastive / embedding regression, zero-shot | THINGS-EEG2, THINGS-MEG | hot, OOD-fragile |
| 5 | **Speech / language** | continuous MEG/EEG → phonemes/words/text | seq2seq, contrastive / end-to-end | MEG-MASC, LibriBrain | **frontier** |
| 6 | **Affective / cognitive state** | longer EEG window → state/score | classification / regression | DEAP, SEED, Sleep-EDF | mature (clinical) |
| 7 | **Auditory attention (AAD)** | EEG + audio envelopes → attended stream | stimulus-reconstruction / correlation | DTU AAD, KU Leuven | niche |

### The cross-cutting axes (where the actual engineering value sits — paradigm-independent)
- **Within vs cross-subject/session transfer** — the 15–20 pt accuracy cliff. *The honest-eval contribution.*
- **Calibration / uncertainty under shift** — ECE; underreported everywhere. Trust in any comms signal needs it.
- **Preprocessing** (filter / epoch / artifact: EOG, line noise) — MNE; ~half the accuracy lives here, not the model.
- **Domain adaptation** — Riemannian alignment, transfer learning; adapt to a new subject without full retrain.
- **Efficiency / on-device** — latency, size, power for real-time. *mindscape's Stage 2 differentiator.*
- **Decoder arch** (CNN→TCN→transformer→foundation models) — commodity; NOT where the contribution is.

---

## 2. Difficulty ladder (simple → hard)

```
EASY  P300/ERP detection ─ binary, big robust signal, Riemannian baseline near-solved
  │   SSVEP/cVEP ───────── freq classification, CCA, highest bitrate, low ML
  │   Motor imagery ────── 2–4 class, MOABB-standard, EEGNet ~4 lines   ← STAGE 0
  │   Affective/state ──── classification, longer windows, DEAP/SEED/Sleep-EDF
  │   Auditory attention ─ binary but needs stimulus reconstruction
  │   Visual/semantic ──── embedding regression, zero-shot, OOD-fragile, MEG helps  ← STAGE 1
HARD  Speech/language ──── seq2seq, MEG-hungry, SOTA still CER 32%      ← STAGE 1 / Lys thesis
```

Difficulty driver = **output cardinality + cross-subject transfer**, not the paradigm. Binary same-subject = easy;
open-vocab seq2seq cross-subject = frontier. MI is the right Stage 0: hard enough to be non-trivial (4-class,
real OOD gap to measure) but standardized enough to be cheap (MOABB). P300/SSVEP are *too* solved — nothing
honest left to contribute.

---

## 3. Dataset per task — all public

| Task | Dataset | Access | Adapter via | Verified specs |
|------|---------|--------|-------------|----------------|
| MI | **BCI IV 2a** (=MOABB `BNCI2014_001`) | open, registration | MOABB (1 line) | 9 subj · 4 class (L/R hand, feet, tongue) · 22 EEG ch @ 250 Hz · 2 sess × 288 trials |
| MI | **BCI IV 2b** | open | MOABB | 9 subj · 2 class (L/R hand) · 3 bipolar ch (C3/Cz/C4) @ 250 Hz · 5 sess — good edge target |
| MI | PhysioNet MMI, Schirrmeister2017 (HGD) | open | MOABB | the other MOABB-standard MI sets (2nd-dataset generalization check) |
| P300 | BNCI, EPFL | open | MOABB | — |
| SSVEP | Nakanishi, Wang | open | MOABB | — |
| Visual/semantic | **THINGS-EEG2** | open (OSF) | MNE/BIDS | 10 subj · 64 ch @ 1 kHz · 16,740 images (train ×4, 200-class test ×80) |
| Visual/semantic | **THINGS-MEG** | open (OpenNeuro) | MNE/BIDS | 4 subj · 275-ch CTF @ 1.2 kHz · 22,448 images |
| Speech/language | **MEG-MASC** (Gwilliams/King) | open, **CC-BY-4.0**, Donders | MNE/BIDS | 27 subj · 2 h naturalistic speech each · word+phoneme timestamps · BIDS |
| Speech/language | **LibriBrain** | open, registration | PNPL/MNE | 50+ h within-subject MEG; built to scale speech decoding |
| Affective/state | DEAP, SEED, Sleep-EDF | open (some email request) | MNE | emotion / sleep staging |
| AAD | DTU AAD, KU Leuven | open | MNE | — |

**Data is not the bottleneck.** MOABB tasks come free (download + standard split + leaderboard ceiling).
THINGS/MEG-MASC are BIDS → small custom adapter, handled by the `paths.yaml` one-root convention.

---

## 4. The DCASE-equivalents (challenge/benchmark culture — yes, the field has it)

**Tier 1 — standing benchmark (the "ImageNet/MOABB of BCI"):**
- **MOABB** (Mother of All BCI Benchmarks) — continuous leaderboard, reproducible pipelines, many
  datasets/paradigms. Within-session mean acc ± std. Founded Barachant/Jayaram. *Stage 0 reproduce target.*

**Tier 2 — annual competitions (the literal DCASE-analogue: time-boxed, leaderboard, paper):**
- **NeurIPS 2025 EEG Foundation Challenge** — cross-task → cross-subject decoding + psychopathology
  prediction. 128-ch, **3000+ subjects, multi-TB**. >1000 teams, >8000 submissions (winner 0.887).
  *Its task IS mindscape's thesis (cross-subject generalization) — externally validated.*
- **PNPL Competition 2025** — speech detection + phoneme classification on **LibriBrain**. The speech-decoding
  DCASE; maps to Lys thesis. Gives a baseline + harness.
- **BCI Competition I–IV** — the OG (BCI IV-2a originates here); like early DCASE.

**Tier 3 — foundation-model benchmarks (emerging):**
- **Brain4FMs**, **Neuroprobe** — benchmark EEG foundation models. Field trending to pretrain-once-decode-many.

**Implication:** field consensus (NeurIPS 2025) names **cross-task + cross-subject generalization** as THE open
problem — mindscape's honest-eval angle verbatim. Cite it. Optionally *enter* EEG2025/PNPL later = strongest
possible builder-who-ships signal for Lys (live leaderboard).

---

## 5. SOTA landscape (the ceiling to quarantine against — DON'T chase)

**MI, BCI IV-2a** (4-class, chance 25%):
- Within-subject SOTA ~**88.5%** (two-stage transformer); transformer pack 83–88%; EEGEncoder 86.5%.
- Cross-subject SOTA ~**74%** (EEGEncoder) — the hard wall; ~14 pt gap persists even at SOTA.
- Reproduce baseline (EEGNet/ShallowConvNet) ~**67–74% within** — correct to sit ~15 pt under SOTA. SOTA = ceiling, not target.

**Non-invasive brain-to-text** (Lys thesis):
- **Meta Brain2Qwerty (Feb 2025) = SOTA.** MEG, end-to-end DL, 9 subj, 22k sentences. **CER 32% avg, 19% best**;
  EEG far worse (CER 67%); word acc ~61% (v2) vs ~8% prior. **MEG ≫ EEG** for speech — modality gap is large/published.
- North star to cite, not compete with.

---

## 6. Build order (mapped to role ROI — from `docs/PLAN.local.md`)

1. **Stage 0 — MI on BCI IV-2a via MOABB + Braindecode + verified eval harness.**
   Lowest effort, proves the spine, **triggers public flip** → makes the Lys application real (closes the
   "no BCI project" gap) + backs Beacon Biosignals. First commit = `paths.yaml` + MOABB adapter + harness
   running on 2a (within-subj acc + cross-subj acc + ECE + per-subject diagnostics). **Do first.**
2. **Stage 0.5 — prove harness generalizes.** Point the *same* harness at a 2nd MOABB MI set
   (PhysioNetMI / Schirrmeister2017). Runs on data it wasn't written for → harness is real. Cheap credibility.
3. **Stage 1 — speech/semantic + honest OOD/calibration.** THINGS-EEG2 first (EEG, cheaper than MEG), then
   MEG-MASC / LibriBrain for speech. Only new code = BIDS adapter; eval harness already exists. Strongest
   Lys-specific signal (their thesis = communication). The honest-eval headline number.
4. **Stage 2 — efficient on-device decoder.** Quantize/distill the Stage-0 decoder, ONNX → edge runtime,
   latency/size/(power) vs full precision, parity-gated. The differentiator nobody else in the pool brings;
   ties to Lys's bandwidth/efficiency framing.

### The one real decision: MOABB-first vs roll-your-own loader
**MOABB-first.** Free verified ceiling (honesty rule #1), free cross-subject splits (headline number, no
split-leakage risk), less plumbing → faster public flip. Risk: don't over-fit the harness to MOABB's API —
keep the eval layer taking plain arrays, decoupled, so Stage 1 BIDS data feeds it untouched.

---

## Sources
- BCI Competition IV — https://www.bbci.de/competition/iv/
- MOABB docs — http://moabb.neurotechx.com/docs/ ; benchmark paper — https://arxiv.org/html/2404.15319v1
- Braindecode BCIC IV 2a example — https://braindecode.org/0.6/auto_examples/plot_bcic_iv_2a_moabb_trial.html
- Cross-subject DL benchmark (EEGNet v4 ~51%) — https://link.springer.com/chapter/10.1007/978-3-031-87657-8_15
- MI SOTA: two-stage transformer 88.5% — https://www.nature.com/articles/s41598-025-06364-4 ; EEGEncoder — https://arxiv.org/html/2404.14869v1
- THINGS scaling-laws (EEG2/MEG specs) — https://arxiv.org/pdf/2501.15322
- MEG-MASC (Sci Data) — https://www.nature.com/articles/s41597-023-02752-5 ; code — https://github.com/kingjr/meg-masc
- Meta Brain2Qwerty — https://ai.meta.com/research/publications/brain-to-text-decoding-a-non-invasive-approach-via-typing/ ; arxiv — https://arxiv.org/pdf/2502.17480
- LibriBrain / PNPL — https://arxiv.org/pdf/2506.02098
- NeurIPS 2025 EEG Foundation Challenge — https://eeg2025.github.io/eeg2025.github.io/ ; paper — https://arxiv.org/html/2506.19141v1 ; NeurIPS comps — https://blog.neurips.cc/2025/06/27/neurips-2025-competitions-announced/
</content>
</invoke>
