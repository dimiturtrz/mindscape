# EEG-fNIRS Hybrid Fusion SOTA for Mental-Workload / n-back Decoding

**Date**: 2026-07-01
**Status**: SETTLED (verdict firm; one open replication item)
**Extends**: fnirs_decoding_sota.md, fnirs_landscape.md, fnirs_decoding_methods.md
**Anchor**: BenchNIRS / Benerradi 2023 (proper cross-subject fNIRS n-back ≈ chance; published fNIRS accuracies inflated by within-session/personalised validation)
**Repo standing rule applied**: a fusion result that looks too good vs honest single-modality baselines is treated as improper eval until proven otherwise.

---

## TL;DR VERDICT (read this first)

1. **SOTA fusion architecture** = multi-branch (per-modality) CNN encoders + a **cross-modal / bidirectional attention** block + joint head. Named exemplars: **MBC-ATT** (Front. Hum. Neurosci. 2025), **TSMMF/BCMT** bidirectional cross-modal transformer (ESWA 2024/25), plus older **decision-fusion CNN-LSTM-GRU** (Cogn. Neurodyn. 2023) and **MSVD** feature fusion (Front. Hum. Neurosci. 2020). Cross-attention transformers are the genuine frontier; the rest is incremental.

2. **Every headline "90-98% on Shin n-back" number is WITHIN-SUBJECT** (train/test split *inside* each subject, or trials pooled with no subject grouping). Under BenchNIRS logic these are inflated and **not comparable to our cross-subject GroupKFold**.

3. **Does fusion beat the best single modality under HONEST cross-subject eval on n-back workload?** — **No credible published evidence that it does, for 3-class 0/2/3-back.** Zero papers report a clean LOSO/GroupKFold 0/2/3-back fusion-vs-both-single-modalities comparison. The only honest cross-subject fusion "wins" are on *emotion* datasets (~+6 pts, still fragile), and on n-back the one honest signal we have (2-back vs 3-back, within-subject) shows fusion **under-performing** fNIRS.

4. **Our null is DEFENSIBLE and consistent with the literature.** Fusion ≈ 0.474 not beating fNIRS-alone (3-class, chance 0.333, subject-wise GroupKFold) is exactly what the honest-eval framing predicts. The field's fusion gains live almost entirely in the within-subject regime that BenchNIRS already discredited for fNIRS.

5. **If we implement one method**: a **compact intermediate-fusion model** — shallow per-modality CNN encoders + a *single* cross-attention block + heavy regularization + per-subject standardization, under strict GroupKFold. It is the least-bad deep candidate at our scale (~700 epochs / 26 subjects). Expect it to *not* beat fNIRS-alone; its value is a defensible negative, not a win.

---

## 1. Leading fusion architectures (named, ranked SOTA vs incremental)

Fusion stages, standard taxonomy: **early/feature** (concat raw/features → one model), **intermediate/hierarchical** (per-modality encoders → learned joint representation, e.g. cross-attention), **late/decision** (per-modality classifiers → combine probabilities).

| Model / paper | Venue, year | Fusion type | Architecture | Task/dataset | Notes |
|---|---|---|---|---|---|
| **MBC-ATT** [S1] | Front. Hum. Neurosci. 2025 (PMC12504385) | intermediate | multi-branch CNN per modality + **cross-modal attention** | Shin 2018 **0/2/3-back** | Closest thing to "SOTA cross-attention on the exact dataset". Within-subject only. |
| **TSMMF (BCMT)** [S2] | Expert Syst. Appl. 2024/25 | intermediate | intra-modal extractors + **bidirectional Cross-Modal Transformer** + attention fusion | EEG-fNIRS **emotion** | Reports **cross-subject** numbers (rare). Code on GitHub (ThreePoundUniverse/TSMMF-ESWA). Genuine frontier design. |
| **DC-AGIN / attention GIN + contrastive** [S3] | Brain Sciences 2026 | intermediate | **graph isomorphism network** + dual contrastive learning | EEG-fNIRS emotion | Explicit LOSO number → see §3. Graph fusion exemplar. |
| Decision-fusion **CNN-LSTM-GRU** [S4] | Cogn. Neurodyn. 2023 (PMC11297873) | late | 7 DL variants, prob-level fuse | Shin **task-type** (n-back vs DSR vs WG) | 96% but NOT workload levels + pooled split (see §2/§3). |
| **MSVD** [S5] | Front. Hum. Neurosci. 2020 (PMC7753369) | feature + "system" | multi-resolution SVD features → KNN | Shin 0/2/3-back | Classic feature fusion. Within-subject 10-fold. |
| Multi-domain + multi-level progressive [S6] | Front. Hum. Neurosci. 2022 (PMC9388144) | hierarchical/stacked | primary+secondary learners | Shin 2017 **MI/MA** (not n-back) | Within-subject 8-fold. |
| Connectivity + ML [S7] | Sensors 2022 (PMC9571712) | feature | functional-connectivity features | Shin 0/2/3-back (binary pairs) | Honest caveat: fusion *underperforms* on hardest pair. |

**Genuine SOTA vs incremental.** The frontier is **cross-attention / bidirectional cross-modal transformers** (MBC-ATT, TSMMF) and **graph fusion** (DC-AGIN) — they learn *inter-modality* dependencies rather than concatenating. Tensor fusion / bilinear pooling appear in the broader multimodal-ML literature but I found no rigorous EEG-fNIRS n-back instantiation; on n-back they'd be incremental at best given data scale. Late-fusion CNN-LSTM-GRU and MSVD are incremental/legacy.

## 2. Reported accuracies on Shin n-back (exact numbers)

**Critical framing:** "n-back" is used two incompatible ways in this literature:
- **(A) Workload levels** = 3-class **0-back vs 2-back vs 3-back** (chance 33.3%) — this is *our* task.
- **(B) Task-type** = **n-back vs DSR vs WG** (chance 33.3%) — a much easier "which activity" problem. Papers routinely blur these.

| Study | Task | Classes | EEG-only | fNIRS-only | **Fusion** | Fusion Δ vs best single |
|---|---|---|---|---|---|---|
| MBC-ATT [S1] | **0/2/3-back (A)** | 3 | 91.58% | (not reported alone) | **98.13%** | +6.55 (ablation) |
| MSVD [S5] | **0/2/3-back (A)** | 3 | — | — | **96.67%** (system-based) | feature-fusion 85.45% |
| Connectivity [S7] | 0-back vs 2-back | 2 | 70% | 68% | **77%** | +7 |
| Connectivity [S7] | 0-back vs 3-back | 2 | 71% | 72% | **83%** | +11 |
| Connectivity [S7] | **2-back vs 3-back** | 2 | 55% | **61%** | 59% | **−2 (fusion LOSES)** |
| Decision-fusion [S4] | n-back vs DSR vs WG **(B)** | 3 | 80% | 80% | **96%** | +16 |
| sLDA (WG vs baseline) [S8] | word-gen, binary | 2 | 76.9% | 74.3% | **80.7%** | +3.8 |

Observations: on the *actual* 0/2/3-back workload task, reported fusion is 96-98% — **implausible** vs BenchNIRS's ~39% cross-subject fNIRS n-back and our own reproduced 0.392. On the *hardest* real contrast (2 vs 3-back, the loads closest in difficulty) fusion **fails to beat fNIRS**, which is the tell that the "easy" gains ride on load-magnitude separability, not on genuine multimodal synergy.

## 3. Evaluation rigor per number (the decisive audit)

| Study | Validation scheme (quoted/paraphrased) | Verdict |
|---|---|---|
| **MBC-ATT** [S1] | "within-subject partitioning… independent training and testing on the dataset of **each subject**"; 80/20 random split + 5-fold on train. | **WITHIN-SUBJECT.** Random 80/20 within a subject ⇒ trials from the same subject/session in train *and* test ⇒ inflated. 98% is not cross-subject. |
| **MSVD** [S5] | "10-fold cross-validation… No LOSO." | **WITHIN-SUBJECT.** 96.67% inflated. |
| **Decision-fusion CNN-LSTM-GRU** [S4] | "train/test = 53/25 samples" pooled; no subject grouping stated; also task-type (B) not workload. | **NOT subject-separated + wrong task.** 96% is easy 3-task discrimination with likely subject leakage. Discard for our purposes. |
| **Connectivity** [S7] | "five-fold CV… 80/20… No LOSO." | **WITHIN-SUBJECT.** 77/83% inflated; but note the honest 2v3 failure. |
| **Multi-domain progressive** [S6] | "8-fold within individual subjects." | **WITHIN-SUBJECT** + not n-back (MI/MA). |
| **sLDA WG/baseline** [S8] | Shin-style within-subject chronological split, binary WG task. | Within-subject; modest (+3.8) and the most honest-looking of the pooled set, but not workload levels. |
| **DC-AGIN (emotion)** [S3] | Reports BOTH: subject-dependent 5-fold **96.98%** → **LOSO 62.56%**. | **The Rosetta Stone.** Same model, same data: honest LOSO is **−34.4 points**. This is the exact BenchNIRS collapse, quantified, on an EEG-fNIRS fusion model. |
| **TSMMF/BCMT (emotion)** [S2] | Cross-subject: **76.15%** fusion vs EEG 70.09% vs fNIRS 63.71%. | **HONEST cross-subject; fusion beats single by ~+6.** But: emotion (not n-back), emotion datasets carry their own stimulus/subject-leakage risks, and +6 is modest and unreplicated on workload. |

**Rule enforced:** every 90%+ Shin n-back fusion number resolved to within-subject on inspection. Confirmed as predicted.

## 4. Does fusion actually help under honest cross-subject eval? — VERDICT

**Clear answer, not a hedge:**

- **On 3-class 0/2/3-back workload, cross-subject: there is NO published honest evidence that fusion beats the best single modality.** No paper reports a clean LOSO/GroupKFold 0/2/3-back run with EEG-only, fNIRS-only, and fusion side-by-side. Every 0/2/3-back fusion win is within-subject.
- **The direct within-honest signal we do have on n-back is negative**: on 2-back vs 3-back (the discrimination that isn't just "hard vs easy"), fusion (59%) **loses** to fNIRS-alone (61%) even *within-subject* [S7].
- **The only honest cross-subject fusion gains are on emotion**, not workload: TSMMF +6 pts [S2] (real but modest/unreplicated), and DC-AGIN's own honest number is 62.56% LOSO after a 34-pt drop [S3]. These do **not** transfer as evidence for n-back workload, and the +6 could itself shrink under the 5-fold-GroupKFold protocol we use.
- **Mechanistically consistent:** EEG and fNIRS share the same neurovascular source under sustained load; the extra modality often adds correlated signal + noise, not orthogonal information — so cross-subject, where you can't memorize a subject's baseline, the marginal modality frequently fails to pay for its added variance.

**Therefore our null is defensible.** Fusion ≈ 0.474 not beating fNIRS-alone under subject-wise GroupKFold is fully consistent with (a) BenchNIRS's near-chance honest fNIRS n-back, (b) the total absence of honest cross-subject n-back fusion wins, and (c) the one quantified honest collapse (−34 pts LOSO). We are not missing a known method that would flip this; we are correctly reproducing what honest eval does to fusion hype.

**Caveat for intellectual honesty (do not oversell the null):** our two schemes are *trivial* (prob-averaging late fusion; concat→LDA feature fusion). "Trivial fusion doesn't beat fNIRS" is weaker than "no fusion beats fNIRS." A properly-regularized intermediate cross-attention model *might* claw back a point or two (cf. TSMMF's honest +6 on emotion). The defensible claim is: **no honest cross-subject n-back fusion win exists in the literature, and our null matches that; a modern intermediate-fusion model is unlikely to overturn it at n=26 but hasn't been tried by us.**

## 5. Implementability on our data (~700 epochs, 26 subj, EEG 28ch + fNIRS 72ch)

**Overfit reality check.** MBC-ATT/TSMMF-class models have 10^5–10^6 params and were validated within-subject where effective N is huge. We have ~702 epochs total, ~27/subject, and must hold out whole subjects. A full cross-attention transformer will memorize and collapse cross-subject — exactly the DC-AGIN −34-pt story.

**Ranked candidates (least-bad first):**

1. **Keep trivial fusion + frame the null (recommended).** Cheapest, and the literature backs it. Report: within-subject fusion literature is inflated; honest cross-subject n-back fusion wins do not exist; our GroupKFold null reproduces this.
2. **Compact intermediate fusion (if we implement one deep method).** Shallow per-modality CNN encoders (2 conv blocks each; EEG on its temporal grid, fNIRS on its slower grid — resample fNIRS or use separate stride, then a small learned temporal pool so the two encoders emit fixed-length tokens), **one** cross-attention block (few heads, small d_model ≤ 32), dropout ≥ 0.5, weight decay, early stopping on an inner GroupKFold. Per-subject z-scoring per channel *before* the split. This mirrors MBC-ATT structurally at ~1/100th the capacity. Expected: ≈ fNIRS-alone, maybe ±2%. Value = a rigorous, reported attempt, not a win.
3. **Avoid**: full transformers (TSMMF/BCMT), graph nets (DC-AGIN), tensor/bilinear fusion — all need far more subjects; they will produce great within-subject numbers and a cross-subject collapse we'd then have to explain.

**PyTorch feasibility:** option 2 is a ~1–2 day build; the risk is not compute, it's that any positive result must survive nested GroupKFold or it's noise. Gate any reported gain on: does it beat fNIRS-alone on the *same* outer folds, with CIs that exclude zero.

---

## Open item
- No paper reports honest LOSO/GroupKFold **0/2/3-back** with per-modality + fusion breakdown. If we build option-2 and run it cleanly, *we* would be producing that missing datapoint — worth doing precisely because the gap exists.

## Sources
- [S1] Multimodal MBC-ATT: cross-modality attentional fusion of EEG-fNIRS for cognitive state decoding. *Front. Hum. Neurosci.* 2025;19:1660532. PMC12504385. https://pmc.ncbi.nlm.nih.gov/articles/PMC12504385/ — Shin 2018 0/2/3-back; **within-subject**; EEG 91.58% → fusion 98.13%.
- [S2] A bidirectional cross-modal transformer representation learning model for EEG-fNIRS multimodal affective BCI (TSMMF/BCMT). *Expert Syst. Appl.* 2024/25. https://www.sciencedirect.com/science/article/abs/pii/S0957417424029488 — cross-subject emotion 76.15% vs EEG 70.09% / fNIRS 63.71% (+6.06/+12.44). Code: github.com/ThreePoundUniverse/TSMMF-ESWA.
- [S3] EEG-fNIRS Cross-Subject Emotion Recognition Based on Attention Graph Isomorphism Network and Contrastive Learning (DC-AGIN). *Brain Sciences* 2026;16(2):145. https://doi.org/10.3390/brainsci16020145 — subject-dependent 96.98% → **LOSO 62.56%** (same model, −34.4 pts).
- [S4] Deep learning networks based decision fusion model of EEG and fNIRS for classification of cognitive tasks. *Cogn. Neurodyn.* 2023. PMC11297873. https://pmc.ncbi.nlm.nih.gov/articles/PMC11297873/ — Shin task-type (n-back vs DSR vs WG), pooled 53/25 split, fusion 96% (CNN-LSTM-GRU). Not workload levels; not subject-separated.
- [S5] Hybrid EEG-fNIRS BCI Fusion Using Multi-Resolution SVD (MSVD). *Front. Hum. Neurosci.* 2020;14:599802. PMC7753369. https://pmc.ncbi.nlm.nih.gov/articles/PMC7753369/ — Shin 0/2/3-back, 10-fold within-subject, system-fusion 96.67%.
- [S6] Improved classification of EEG-fNIRS multimodal BCI via multi-domain features and multi-level progressive learning. *Front. Hum. Neurosci.* 2022. PMC9388144. https://pmc.ncbi.nlm.nih.gov/articles/PMC9388144/ — Shin 2017 MI/MA, within-subject 8-fold, 96.74%/98.42%.
- [S7] EEG/fNIRS Based Workload Classification Using Functional Brain Connectivity and ML. *Sensors* 2022;22(19):7623. PMC9571712. https://pmc.ncbi.nlm.nih.gov/articles/PMC9571712/ — Shin 0/2/3-back binary pairs, within-subject 5-fold; fusion 77% (0v2), 83% (0v3), **59% (2v3, loses to fNIRS 61%)**.
- [S8] Assessment of mental workload by EEG+fNIRS / hybrid sLDA on Shin data (WG vs baseline binary): EEG 76.9%, fNIRS 74.3%, fusion 80.7%. https://www.researchgate.net/publication/305807924
- Anchor: Benerradi et al. BenchNIRS. *Front. Neuroergonomics* 2023;4:994969 (cross-subject fNIRS n-back LDA 38.9%; reproduced in-repo 0.392).
- Context: CM-GGT (26-subj n-back/DSR/WG task-type, 93.2%) — task-type not workload; noted, not used.
