# BCI IV-2a SOTA recipe — what the published numbers actually do (and why ours are below)

> Deep-dive · 2026-06-30 · grounds the "what are we doing differently" diagnosis in primary sources.
> Triggered by our results sitting ~23-28 pts under claimed SOTA. Conclusion: the gap is **pipeline +
> model-usage**, not data or approach — and our recorded ATCNet ceiling was slightly too high.

---

## Our numbers (honest, T→E hold-out, 4-class)
- CSP+LDA within: **0.598** / κ0.463 ; cross-subject (LOSO): **0.382** / κ0.176
- ATCNet within (crop-aug, early-stop): **0.584** / κ0.445

## Published bar (primary sources)
- **ATCNet (Altaheri 2023): 81.1–82.0%** — *10-run average*, **500 epochs, NO early stopping**, hold-out
  session-1→session-2 (the SAME protocol we use). [repo](https://github.com/Altaheri/EEG-ATCNet)
- **EEGNet: ~71%** on 2a 4-class with proper preprocessing. [braindecode](https://braindecode.org/0.6/auto_examples/plot_bcic_iv_2a_moabb_trial.html)
- **FBCSP: ~65% / κ0.57** (BCI IV winner).
- The 0.88 "transformer SOTA" is a separate outlier claim; the **realistic strong bar is ATCNet ≈ 0.81**.

> Correction: our `reference.yaml` had ATCNet at 0.85 — the actual reported figure is **0.81** (10-run
> mean). Fixed.

---

## What the SOTA pipeline does that ours does NOT (primary-source confirmed)

| # | SOTA recipe | ours | confirmed cost |
|---|---|---|---|
| 1 | **Exponential moving standardization** (`factor_new=1e-3, init_block_size=1000`) on the continuous signal | whole-epoch per-channel z-score | braindecode's *canonical* 2a preprocessing — a real, named difference |
| 2 | **500 epochs, NO early stopping** (ATCNet) | early-stop ~ep 90 on in-session val | **we under-train** — confirmed against the repo |
| 3 | ATCNet's **own built-in convolutional sliding-window augmentation** | our *external* 2s crop layer, net built for crop_len=500 | our crop **fights** ATCNet's design (built for full ~1125-sample trial) |
| 4 | **10-run average** | single seed | ±3-5 pts; our mean is dragged by collapse subjects (s5 0.25, s6 0.38) |
| 5 | bandpass **4–38 Hz** | 4–40 Hz | negligible |
| 6 | eval = hold-out session-1→2 | **same** | ✓ our protocol is correct/comparable — NOT the gap |

**The eval protocol is NOT the excuse.** ATCNet's 0.81 is the *same* T→E hold-out we run. So our ~23-pt
gap to ATCNet is genuine pipeline weakness, not an honesty penalty. (Honesty only costs vs the papers
that use within-session CV — those inflate ~5-10 pts; not our comparison here.)

---

## Diagnosis (grounded)
The gap is **training pipeline + model-usage**, ranked:
1. **No exponential-moving standardization** (the braindecode-standard preprocessing). [#1]
2. **Under-training** — early-stop at ~90 ep vs their 500-ep full schedule. [#2]
3. **External crop conflicts with ATCNet's internal sliding-window aug.** [#3]
4. **Single seed** — mean dragged by 2-3 collapse subjects; SOTA averages 10 runs. [#4]

Not data (identical), not the harness/eval (correct + honest), not architecture availability
(braindecode ships the real nets).

## The fix (ranked, closes most of the gap)
1. **Exponential-moving standardization** in preprocessing (carry braindecode's `factor_new=1e-3`).
2. **ATCNet uncropped, full 500-epoch schedule, no/loose early-stop** — let it use its own window aug.
3. **Keep external crops only for nets without internal aug** (EEGNet/Shallow).
4. **Seed-averaging (3-5 runs)** → reportable mean±std, fixes the collapse-subject drag.

## What this means for the project
The plan says don't chase SOTA — but our numbers shouldn't read as *weak* either. These four fixes are
standard practice, not leaderboard-chasing; they make the decoder a **fair representative** (~0.75-0.80
range expected for ATCNet) so the real contribution — the **within→cross gap (−0.216)** + calibration —
stands on a credible decoder rather than an undertrained one.

## Sources
- EEG-ATCNet repo (500 ep, no early stop, 81%, hold-out) — https://github.com/Altaheri/EEG-ATCNet
- Braindecode 2a recipe (4-38 Hz + exponential_moving_standardize, factor_new=1e-3) — https://braindecode.org/0.6/auto_examples/plot_bcic_iv_2a_moabb_trial.html
- Braindecode data-augmentation search on 2a — https://braindecode.org/dev/auto_examples/advanced_training/plot_data_augmentation_search.html
- EEGNet 71% on 2a — https://deepwiki.com/amrzhd/EEGNet/3-bci-competition-iv-2a-dataset
