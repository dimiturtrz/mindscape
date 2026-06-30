# M3 · Probability & statistics → the eval/honesty layer

Working fluency, tied to the contribution. Reference: **StatQuest** (cross-entropy, ROC/PR, calibration);
*Mathematics for ML* (Deisenroth) for the probability spine.

## The core ideas
- **Softmax** `e^{xᵢ}/Σe^{xⱼ}` → a distribution: every value **≥0** (exp is positive; negatives → small
  positives), and **sums to 1**. All-equal logits → uniform (e.g. 4 zeros → 0.25 each, *not* zero).
- **Cross-entropy** `CE = −log(p_true)` — penalizes by *how much* prob mass you put off the truth, not just
  right/wrong. It's the loss (not accuracy) because accuracy is a step function (no gradient); CE is smooth →
  gradient pushes the true class's probability up.
- **Cohen's κ** `= (acc − chance)/(1 − chance)` — accuracy minus luck. 4-class chance 0.25; our cross-subject
  `(0.382−0.25)/0.75 = 0.176`. κ=0 = chance, κ=1 = perfect, κ<0 = worse than guessing. Report it at 4 classes
  because raw accuracy flatters. It's a **skill score** in form — same template as the **Brier *skill*
  score** `(B−B_base)/(0−B_base)`. (Raw Brier = MSE on probs, a proper scoring rule ~ calibration; Tjur's D =
  mean-prob separation = discrimination; PR-AUC = ranking — those are *probabilistic*, unlike κ's hard-label
  agreement.)
- **ECE** — "when it says 90% sure, is it right 90% of the time?" Per prediction: confidence = max class-prob,
  correct = 0/1. **Bin predictions by confidence**; per bin compare **accuracy (mean of the 0/1s)** vs
  **mean confidence**; `ECE = Σ (n_bin/N)·|acc − conf|`. GT isn't binned — it's *averaged within confidence
  bins*. Perfect → ECE 0; overconfident (our pre-calibration nets) → conf > acc → positive ECE.
- **Temperature scaling** — logits ÷ T before softmax. T>1 → logits closer → softmax flatter → **lower
  confidence**. **argmax unchanged** (same positive scaling preserves order) → accuracy untouched, only ECE
  moves. Same flattening as LLM "creativity" temperature, different use (calibrate vs sample).
- **Estimate variance / LOSO.** A single mean hides the per-subject spread (0.34–0.84) — report std/range.
  **LOSO = Leave-One-Subject-Out** CV (train on 8, test on the held-out 9th, rotate) = the honest
  generalization estimate (test subject never in training = a new user). Same family as systole's **LOVO**
  (Leave-One-Vendor-Out) = Leave-One-*Group*-Out, group = the shift axis (subject/vendor/session/site).

---

## Quiz log

### 2026-06-30 — quiz M3
**Score ~3.5/6 — fluent in ML-adjacent stats (cross-entropy, softmax formula, temperature), gaps in the
eval-specific trio (κ, ECE binning, LOSO acronym).**

1. *Softmax* — ✓ formula; missed the two distribution properties (≥0, sum=1); error "all-zeros→all-zero"
   (it's uniform 0.25).
2. *Cross-entropy* — ✓ good (measures *how wrong*; accuracy can't train, no gradient). Sharpened to `−log p_true`.
3. *κ* — ✗ (taught fully: chance-corrected accuracy, skill-score form).
4. *ECE* — ✓ *why* (confident+wrong = dangerous); ✗ *how* (taught the confidence-binning, conf vs accuracy).
5. *Estimate variance / LOSO* — ✓ mean needs std/range; LOSO acronym was new (= the cross-subject scheme,
   recognized as systole's LOVO family).
6. *Temperature* — ✓ correct mechanism + argmax-preservation; reconciled with LLM "creativity" temperature.

**Cross-links drawn:** κ ≈ Brier skill score (skill-score template); Brier/Tjur/PR-AUC are probabilistic
(calibration/discrimination/ranking) vs κ's hard-label agreement; LOSO = LOVO (leave-one-group-out); ECE
bins by confidence (not GT). **Takeaway: κ + ECE + LOSO *are* the honesty layer** — the project's headline
(`within 0.598 → cross 0.382, κ 0.176, ECE under shift`) is the contribution stated in stats.
