# Workload — n-back mental workload (Shin, EEG · fNIRS · fusion)

The **state** rung of the [field-map](../../../README.md): decode which n-back load (0/2/3-back) a subject
holds in working memory, from the Shin hybrid set where EEG and fNIRS were recorded **simultaneously** — so
both modalities decode *one identical task* and any difference is the **modality, not the task**. The root
README carries the headline (the re-centering transfer fix, EEG 0.45 → 0.60 cross-subject) and the fusion
result table; this page is the depth behind them.

```bash
uv run python -m neuroscan.tasks.workload.run_fnirs --exp nback_fnirs_cross       # fNIRS amplitude → LDA
uv run python -m neuroscan.tasks.workload.run_fusion --exp nback_fusion           # EEG+fNIRS, the sweep
```

## What actually carries the fNIRS signal — a feature-importance search

The baseline uses **mean + slope + peak**, the field-standard triple. But *which* of those actually matters?
We let a search tell us — **Optuna as a wrapper feature-selector** over 15 per-channel temporal descriptors,
one weight ∈ [0,1] per family, scored by subject-grouped 5-fold CV
([`optuna_search`](feature_importance/optuna_search.py)). A search *maximises*, so its peak accuracy (~0.50) is
optimistic and reported as such — the robust deliverable is the **importance ranking**, and it's **stable**
(top-5 Jaccard 0.20 at 30 trials → **0.67 at 200**, across 3 re-runs). It runs a fixed `shrinkage=0.4` LDA
(constant across trials, so it can't confound the weight comparison) — a *different* classifier from the
`shrinkage="auto"` LDA in the fixed-recipe CV below, so the ~0.50 here and the ~0.46 there aren't on one scale.

**Finding — dynamics ≫ amplitude** (200 trials × 3 seeds):

| family | importance | best-trial weight | reading |
|---|---|---|---|
| **slope** | **0.35** | **0.91** | the workhorse — kept on, dominates |
| time-to-peak | 0.25 | 0.22 | influential but best trials **suppress** it (hurts if included) |
| range | 0.13 | 0.41 | moderate |
| late-slope | 0.09 | 0.76 | mild help (response *shape*, not size) |
| **mean** | 0.03 | **0.15** | near-dead — the search **drops** it |
| peak | 0.02 | 0.58 | minor |

The workload signal is in the **rate and shape of the hemodynamic rise (`slope`)**, *not* its magnitude —
`mean` and `peak`, two-thirds of the standard triple, carry almost nothing here. Mechanistically clean: `mean`
over the −2→20 s window blends baseline + rise + plateau and dilutes the contrast, while `slope` reads the
rise directly — matching the fNIRS literature's emphasis on regression/slope features.

**Two follow-ups confirmed it — and corrected it.** A **differentiable** version
([`differentiable`](feature_importance/differentiable.py), torch/CUDA, softmax weights + entropy sparsity
penalty) leaves `slope` the last family standing — and exposes a limit: **per-channel** weights (1080) are
statistically unidentifiable on 702 blocks (they stay uniform), so the per-family view is the right
resolution. And a **fixed-recipe CV** — no search, no selection optimism
([`recipes`](feature_importance/recipes.py), 3×5-fold GroupKFold):

| recipe | acc | κ |
|---|---|---|
| **dynamics** (slope + early/late-slope) | **<!--r:fnirs_recipe_dynamics_shin2017_nback.acc-->0.466<!--/r-->** | <!--r:fnirs_recipe_dynamics_shin2017_nback.kappa-->0.199<!--/r--> |
| full (15 families) | <!--r:fnirs_recipe_full_shin2017_nback.acc-->0.464<!--/r--> | <!--r:fnirs_recipe_full_shin2017_nback.kappa-->0.196<!--/r--> |
| amplitude — mean+slope+peak (baseline) | <!--r:fnirs_recipe_amplitude_shin2017_nback.acc-->0.460<!--/r--> | <!--r:fnirs_recipe_amplitude_shin2017_nback.kappa-->0.190<!--/r--> |
| slope only | <!--r:fnirs_recipe_slope_only_shin2017_nback.acc-->0.446<!--/r--> | <!--r:fnirs_recipe_slope_only_shin2017_nback.kappa-->0.169<!--/r--> |
| mean only | <!--r:fnirs_recipe_mean_only_shin2017_nback.acc-->0.392<!--/r--> | <!--r:fnirs_recipe_mean_only_shin2017_nback.kappa-->0.088<!--/r--> |
| peak only | <!--r:fnirs_recipe_peak_only_shin2017_nback.acc-->0.376<!--/r--> | <!--r:fnirs_recipe_peak_only_shin2017_nback.kappa-->0.064<!--/r--> |

The **slope *trajectory*** (rise rate + early/late shape, 3 features) **ties the full 15-family bank and edges
the standard triple**, while `mean` and `peak` alone barely clear chance. So two-thirds of the field-standard
triple is redundant; the real recipe is *shape, not magnitude*. (Assume-wrong in action: the search
*suggested* slope; the no-search CV *tempered* the claim — slope-*alone* sits a hair under the triple, it's the
trajectory that carries it.)

**Does keeping the time axis help?** A windowed decoder ([`WindowedFnirs`](../../../baselines/fnirs/windowed.py))
sub-windows the response instead of collapsing it. Ordered 3-window concat lifts **within**-subject +2.5 pp
(0.448 → 0.473) — but the gain is subject-idiosyncratic and only *ties* the collapse cross-subject; MIL pooling
sits *below* it (~0.39, no localized cue in a smooth response). So the collapse isn't lazy — it's matched to
the signal, and the temporal gain it leaves is trapped *within* subject.

## The graded-load ceiling — physiology, not a method gap

We attacked the ~0.46 cross-subject 3-class number **four ways** — a 15-family feature bank, windowed temporal
representations, CBSI physiological-noise cleaning, and a GLM-β HRF model — and **every one ties the
mean+slope+peak baseline** (~<!--r:fnirs_lda_cross_subject_shin2017_nback.acc-->0.454<!--/r-->), which itself
matches the field benchmark (reproduced BenchNIRS 0.389 → 0.392). The wall is a **single boundary: 2-back vs
3-back is at chance even *within* subject** (Ishii 2013: 0.61 within, 3-class 0.50 within; ours ≈ chance). So
it is **not a transfer problem** — domain adaptation can only move signal that exists within a subject, and
here there is none to move. The mechanism is physiology: a fixed ~2.2 s stimulus SOA + neurovascular
saturation + the inverted-U load response mean 2- and 3-back produce the **same-shaped hemodynamic plateau,
differing only in a tiny, sign-flipping, subject-relative height**. fNIRS reads *whether* working memory is
engaged, not *how hard* — a **load detector, not a meter**.

This closes the "lift the weak modality" path in fusion below: **no lever** — DA has nothing to transfer, and
better features can't extract absent signal. It **matches the field**: no *leakage-free* published result beats
~chance+10 pts on cross-subject Shin n-back (the 0.83–0.96 numbers are within-subject / leaky splits). The
deliverable is the **rigorous negative + the mechanism**. fNIRS earns its keep where the hemodynamic response
*is* the signal (engagement detection, motion-robust ambulatory BCI); graded WM *level* cross-subject is a poor
task for it.

## Fusion — complementarity is real, output-space fusion barely captures it

Both decoders run on the **same aligned epochs** — EEG **re-centered Riemann** (zero-shot per-subject) + fNIRS
mean/slope/peak — under one **5-fold GroupKFold** (fusion needs per-epoch EEG↔fNIRS pairing, which LOSO's
single-subject test sets make too small to read). The role table lives in the root README; the open question
is the **complementarity** — the two still **fail on different blocks**:

| complementarity (same 5-fold) | value |
|---|---|
| best single modality | <!--r:fusion_cross_subject_kfold_shin2017_nback.best_single-->0.580<!--/r--> |
| late fusion (what naive averaging gets) | <!--r:fusion_cross_subject_kfold_shin2017_nback.late-->0.587<!--/r--> |
| **oracle — *either* modality correct** | **<!--r:fusion_cross_subject_kfold_shin2017_nback.oracle_either-->0.752<!--/r-->** |
| oracle headroom over best single | **<!--r:fusion_cross_subject_kfold_shin2017_nback.oracle_either-fusion_cross_subject_kfold_shin2017_nback.best_single-->+0.172<!--/r-->** |
| error correlation (φ) | <!--r:fusion_cross_subject_kfold_shin2017_nback.err_corr-->0.107<!--/r--> |
| EEG-only-right / fNIRS-only-right / both-wrong | 0.28 / 0.17 / <!--r:fusion_cross_subject_kfold_shin2017_nback.both_wrong-->0.248<!--/r--> |

**The headroom is large and mostly uncaptured.** A per-trial oracle picking the right modality would hit
**0.752** — **+17 pts** over the best single — with near-independent errors (φ ≈ 0.11): EEG uniquely rescues
~28 % of blocks, fNIRS ~17 %, only ~25 % beat both. So the ceiling isn't the data, it's the **fusion
mechanism**. We swept every **output-space** combiner (stacking + temperature fit on an inner GroupKFold — no
test leakage):

| output-space aggregator | acc | vs best single |
|---|---|---|
| mean (late) | <!--r:fusion_cross_subject_kfold_shin2017_nback.mean-->0.587<!--/r--> | +0.007 |
| **product (naïve Bayes)** | **<!--r:fusion_cross_subject_kfold_shin2017_nback.product-->0.595<!--/r-->** | **+0.015** |
| confidence-weighted | <!--r:fusion_cross_subject_kfold_shin2017_nback.conf_weight-->0.580<!--/r--> | ±0.000 |
| max-confidence pick | <!--r:fusion_cross_subject_kfold_shin2017_nback.maxconf_pick-->0.591<!--/r--> | +0.011 |
| stacking (meta-LDA, nested CV) | <!--r:fusion_cross_subject_kfold_shin2017_nback.stacking-->0.587<!--/r--> | +0.007 |
| calibrated mean | <!--r:fusion_cross_subject_kfold_shin2017_nback.cal_mean-->0.587<!--/r--> | +0.007 |
| calibrated conf-weighted | <!--r:fusion_cross_subject_kfold_shin2017_nback.cal_conf_weight-->0.580<!--/r--> | ±0.000 |

**Several combiners now marginally *beat* best-single (product +1.5 pp)** — a real shift from the earlier
weak+weak version where *every* combiner lost. Mechanistic: re-centering made EEG's **confidence informative**
(correct-vs-wrong gap +<!--r:fusion_cross_subject_kfold_shin2017_nback.eeg_conf_gap-->0.132<!--/r--> for EEG vs
+<!--r:fusion_cross_subject_kfold_shin2017_nback.fnirs_conf_gap-->0.038<!--/r--> for fNIRS), so product/max-pick
can *partially* tell which modality to trust. The lesson: **output-space fusion works to the extent the
modalities are calibrated**, and fails when confidence is noise.

But the gains are within fold noise and far under the oracle, and an **input-level gate**
([`fusion_gate.py`](fusion_gate.py)) reading reliability from the raw signals only *ties* EEG-alone
(**<!--r:fusion_gate_cross_subject_kfold_shin2017_nback.gate-->0.573<!--/r-->**), capturing none of the headroom.
A *second, independent* EEG fix confirms the ~0.58 strength from the feature side — per-subject unsupervised
z-scoring of the subject-idiosyncratic band-power recovers EEG from
<!--r:calibration_ablation_shin2017_nback_eeg.eeg_raw-->0.407<!--/r--> to
**<!--r:calibration_ablation_shin2017_nback_eeg.eeg_zcalib-->0.511<!--/r-->** (held-out calibration-half, no
leakage) / **<!--r:calibration_ablation_shin2017_nback_eeg.eeg_ztrans-->0.581<!--/r-->** (transductive)
([`calibration_ablation.py`](calibration_ablation.py)). So — measured — neither the combiners nor the gate
capture the complementarity on this strong+weak pair; **boundary-aware routing and source-space fusion are the
open, untested routes**.

Two caveats. (1) The oracle is an **upper bound** — headroom *exists*, not claimable. (2) The literature offers
no free lunch: every published Shin n-back fusion number (96–98 %) is **within-subject** (inflated as BenchNIRS
predicts), the one leakage-free EEG-fNIRS fusion LOSO figure *drops* 34 pts (DC-AGIN 96.98 → 62.56 %), and on
the hardest real contrast (2- vs 3-back) fusion *loses* to fNIRS — so a learned model must be small (compact
cross-attention, the only thing that fits n=26) and gated on **strict nested GroupKFold**, or it reproduces
that collapse.

Full audit + citations → [`research/`](../../../research/deep_dives/2026-07-01_eeg_fnirs_fusion_sota.md).
