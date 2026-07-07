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
  │   Visual/semantic ──── embedding regression, zero-shot, OOD-fragile, MEG helps  ← STAGE 1 (perception)
HARD  Speech/language ──── seq2seq, MEG-hungry, SOTA still CER 32%      ← STAGE 1 (comms)
```

The difficulty driver is **output cardinality × cross-subject transfer**, not the paradigm label: binary
within-subject is easy; open-vocabulary seq2seq across subjects is the frontier. Those two axes line up
cleanly — **perception** (view / read, stimulus-driven, decodable) sits below **communication** (produce /
imagine, endogenous, hard). MI is the right Stage 0: non-trivial (4-class, a real OOD gap to measure) yet
standardized and cheap (MOABB). P300/SSVEP are *too* solved — little honest headroom left to contribute.

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
- **NeurIPS 2025 EEG Foundation Challenge** — the marquee event: cross-task → cross-subject decoding plus
  psychopathology prediction, 128-ch, **3000+ subjects, multi-TB**, >1000 teams / >8000 submissions
  (winner 0.887). *Its task IS mindscape's thesis — cross-subject generalization, externally validated.*
- **PNPL Competition 2025** — speech detection + phoneme classification on **LibriBrain**. The
  speech-decoding analogue of DCASE — a ready-made baseline + harness for the communication axis.
- **BCI Competition I–IV** — the original series (BCI IV-2a originates here); the field's early-DCASE.

**Tier 3 — foundation-model benchmarks (emerging):**
- **Brain4FMs**, **Neuroprobe** — benchmarks for EEG foundation models; the field is trending toward
  pretrain-once, decode-many.

**Implication:** the field's own consensus (NeurIPS 2025) names **cross-task + cross-subject generalization**
as THE open problem — which is mindscape's honest-eval angle almost verbatim, so cite it as external
validation. Entering EEG2025 / PNPL later is optional upside (a live public leaderboard is a strong ship-it
signal), but the contribution stands on its own without a leaderboard entry.

---

## 5. SOTA landscape (the ceiling to quarantine against — DON'T chase)

**MI, BCI IV-2a** (4-class, chance 25%):
- Within-subject SOTA ~**88.5%** (two-stage transformer); the transformer pack lands 83–88%; EEGEncoder 86.5%.
- Cross-subject SOTA ~**74%** (EEGEncoder) — the hard wall: a ~14 pt gap persists even at SOTA.
- Baseline to reproduce (EEGNet / ShallowConvNet) ~**67–74% within-subject** — correctly ~15 pt under SOTA;
  SOTA is the ceiling, not the target.

**Non-invasive brain-to-text decoding** (the communication frontier — the hardest rung):
- **Meta Brain2Qwerty (Feb 2025) = SOTA.** MEG, end-to-end DL, 9 subj, 22k sentences. **CER 32% avg, 19% best**;
  EEG far worse (CER 67%); word acc ~61% (v2) vs ~8% prior. The **MEG ≫ EEG** speech gap is large and
  published — a standing caution against expecting brain-to-text on consumer EEG.
- A north star to cite for scale + ceiling, not a target to compete with here.

---

## 6. Build order (capability ROI)

1. **Stage 0 — MI on BCI IV-2a via MOABB + Braindecode + a verified eval harness.**
   Lowest effort, highest return: it proves the whole spine end-to-end and closes the "no BCI project" gap
   with a first credible decoding result on public data. First commit = `paths.yaml` + a MOABB adapter +
   the harness running on 2a (within-subject acc + cross-subject acc + ECE + per-subject diagnostics).
   **Do this first.**
2. **Stage 0.5 — prove the harness generalizes.** Point the *same* harness at a 2nd MOABB MI set
   (PhysioNetMI / Schirrmeister2017). If it runs on data it was never written for, the harness is real —
   cheap, high-signal credibility.
3. **Stage 1 — semantic/visual, then speech, with honest OOD + calibration.** THINGS-EEG2 first (EEG, far
   cheaper than MEG), then MEG-MASC / LibriBrain for speech. The only new code is a BIDS adapter; the eval
   harness already exists. This is the **communication axis** (brain-to-text = the frontier), where the
   honest cross-subject headline number carries the most weight.
4. **Stage 2 — an efficient on-device decoder.** Quantize / distill the Stage-0 decoder, export to ONNX → an
   edge runtime, and report latency / size / (power) against full precision, parity-gated. The differentiator
   most decoding work skips — the on-device bandwidth + efficiency angle.

### The one real decision: MOABB-first vs roll-your-own loader
**MOABB-first.** It buys a free verified ceiling (honesty rule #1) and free cross-subject splits (the headline
number, with no split-leakage risk), and less plumbing means a working spine sooner. The one risk: don't
over-fit the harness to MOABB's API — keep the eval layer taking plain arrays, decoupled, so Stage 1 BIDS
data feeds it untouched.

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
