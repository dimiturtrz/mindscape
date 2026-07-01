# M5 · Hemodynamics → why the fNIRS signal is the way it is

The hemodynamic counterpart to [M4](M4_neurophysiology.md) (which did EEG's electrophysiology). *Enough to
explain the fNIRS signal and its failure modes*, not tissue optics from scratch. Grounded in
[`research/deep_dives/2026-07-01_fnirs_fundamentals.md`](../../research/deep_dives/2026-07-01_fnirs_fundamentals.md).
Reference: Scholkmann 2014 (review), Ferrari & Quaresima 2012, Pinti 2020.

## The core ideas
- **Source = blood oxygenation, read by light.** fNIRS shines near-infrared light into the scalp and measures
  what comes back. It doesn't see neurons — it sees the **hemodynamic response** to their activity: ΔHbO
  (oxy-) and ΔHbR (deoxy-hemoglobin) concentration changes. An **optical**, **indirect** signal (vs EEG's
  direct electrical one).
- **The optical window (~650–950 nm).** In this band tissue is relatively transparent: hemoglobin caps the
  bottom (absorbs below ~650), water the top (absorbs above ~950). So NIR light penetrates far enough to
  reach cortex. In tissue, **scattering ≫ absorption (~100×)** — light doesn't travel straight; photons take
  a **banana-shaped path** from source to detector.
- **Shallow — cortex only.** Sensitivity peaks at ~**half the source–detector separation**. Standard adult
  separation ~**2.5–3 cm** → reaches ~1.5–3 cm deep = the **cortical surface only**. fNIRS **never** sees deep
  structures. (Wider separation = deeper but weaker signal.)
- **Two wavelengths + the modified Beer–Lambert law.** HbO and HbR have different absorption spectra that
  **cross at the isosbestic point (~800 nm)**. Use **two wavelengths** (one either side) → two equations,
  two unknowns → solve for ΔHbO and ΔHbR. The **modified Beer–Lambert law (MBLL)** converts ΔOD →
  concentration using a **differential pathlength factor (DPF ≈ 6** adult, wavelength/age-dependent**)**.
  Continuous-wave fNIRS gives only **relative** changes (the scattering term cancels) — the DPF assumption is
  the main quantitative caveat.
- **Neurovascular coupling — the crux, and why it's SLOW.** Neural activity → local demand → **functional
  hyperemia** (blood flow *over*-supplies) → **HbO ↑, HbR ↓** (they anti-correlate — the hallmark of a real
  response). The **hemodynamic response function (HRF)**: onset ~1–2 s, **peak ~5–8 s**, undershoot, recovery
  ~15–25 s. This is why fNIRS is **slow** (~10 Hz sampling is plenty) and why trials are long. **Same
  physiology fMRI-BOLD reads** (BOLD ∝ −HbR) — fNIRS is portable optical BOLD. (An "initial dip" is debated.)
- **Systemic physiology = the big confound.** The signal is contaminated by **cardiac (~1 Hz)**, **respiration
  (~0.2–0.3 Hz)**, and **Mayer waves (~0.1 Hz** blood-pressure oscillation — **the worst, it overlaps the
  task band)**, plus scalp blood flow. **Short-separation channels (~8 mm)** sample scalp-only hemodynamics to
  regress it out (our Shin dataset has none → a stated limitation).
- **Modality tradeoff.** fNIRS: **portable + hemodynamic + shallow + slow**, robust to motion/electrical noise,
  good for **prefrontal cognition / motor / naturalistic + infant** studies. vs EEG (electrical, ms-fast,
  volume-conduction-blurred) · fMRI (same hemodynamics, mm-spatial, huge + immobile). fNIRS = "wearable fMRI-
  lite."

## The synthesis (why M5 changes the decoding)
The fNIRS signal is **slow, amplitude-coded, and shallow**. Every downstream choice flips vs EEG:
- **The class signal is in the AMPLITUDE** (how high ΔHbO rises), not oscillatory power/covariance. → decode
  with **mean + slope + peak features → LDA**, *not* CSP/Riemann (which center out the mean = the whole
  signal). This is the **method-must-match-the-signal-physics** lesson, made concrete.
- **Slow HRF** → long windows (peak ~5–8 s), few trials, tiny datasets → simple features beat deep nets.
- **Systemic physiology** (Mayer waves in-band) is fNIRS's version of EEG's artifacts — the honest-eval
  caveat (can inflate accuracy; needs short-separation channels we don't have).
- Interview answer to "why does fNIRS need different methods than EEG": it measures a **slow hemodynamic
  amplitude** (neurovascular coupling), not a fast electrical oscillation — so amplitude features, not
  covariance/spatial filters.

---

## Quiz log

### 2026-07-01 — quiz M5 (domain section) · ~5/6, one skipped
Strong mechanistic grasp; proactively added the mitochondria/aerobic-efficiency point.

1. *What it measures* — ✓✓ ΔHbO/ΔHbR concentration; indirect (downstream blood) vs EEG's direct electrical.
2. *Neurovascular coupling* — ✓ energy chain + **aerobic ≈ 20× anaerobic ATP** (correct, self-added). Missing
   the crux: the brain **over-supplies** O₂ → **HbO net RISES** at the active site (not falls) → that's why
   HbO↑/HbR↓ **anti-correlate**.
3. *Why light / shallow* — ✓✓ optical window (blood low / water high), scattering ≫ absorption → reflectance
   near source + can't go deep. Add: NIR reads oxygenation *because* HbO/HbR absorb NIR differently.
4. *Why slow* — ✓✓ hemodynamics lag the spiking; ~5 s delay; long trials; peak then slow decay. (peak ~5–8 s).
5. *fNIRS vs fMRI* — ⬜ skipped. Same physiology; fMRI-BOLD ∝ −HbR (paramagnetic), magnetic vs optical sensor.
6. *Confounds* — ✓ cardiac/respiration filterable; got **Mayer ~once/10 s**. Reinforced: Mayer is worst
   *because* ~0.1 Hz **overlaps the task/HRF band** → can't filter → needs short-separation channels.

**To lock:** over-supply → net HbO rise (anti-correlation); Mayer-worst = in-band; fMRI-BOLD = HbR/paramagnetic.

### 2026-07-01 — quiz M5 (maths section) · ~5/6
1. *Beer–Lambert* — ✓✓ `I=I₀·10^(−εcL)` (exponential form); "3 unknowns, can't get most." (linear: A=log(I₀/I)=εcL).
2. *Why plain BL breaks* — ✓ path unknown after scatter + inhomogeneous; DPF≈6×distance. Fix: differences
   cancel the **unknown scatter-loss `G`** (not `c`) → leaves Δc.
3. *Relative only* — ✓✓ unknown `G`/path → absolute c unrecoverable, only changes survive.
4. *Two wavelengths* — ✓✓ oxy/deoxy differ by λ; **2 eqns 2 unknowns → invert 2×2 extinction matrix**.
5. *Isosbestic straddle* — ✓✓ at ~800 nm ε_HbO=ε_HbR → rows collapse → singular; straddle = condition the solve.
6. *Features / why covariance fails* — ~½ on the day, but the follow-up produced the **richer answer** (below).

**Why covariance fails on fNIRS — the full, two-axis answer** (student connected the second axis):
Covariance is a *second-order, zero-mean, cross-channel* statistic. fNIRS breaks it on **two independent axes**:
- **Order:** the signal is **first-order (the mean HbO level)** — covariance **centers the mean out**.
- **Cross-channel:** fNIRS is **spatially local** (no volume conduction) → off-diagonals ≈ 0 → nothing for
  CSP's spatial filters to *unmix*; a diagonal covariance = just per-channel variance (the wrong statistic).
They converge: covariance's best case here is per-channel *variance*, but the signal is per-channel *mean*.
**Caveat (sharpens it):** the covariance is NOT purely diagonal — **systemic physiology (Mayer/cardiac) is
global**, correlating all channels → strong off-diagonals, but that's **noise**, so covariance methods lock
onto the artifact. Empirically consistent: Riemann = 0.329 ≈ chance.

**To lock:** `G` cancels in differences; covariance fails on BOTH order (mean centered out) AND locality
(no cross-channel signal structure; the off-diagonals that exist are systemic noise).

### approaches section — pending
_(next: classical mean+slope+peak→LDA vs DL 1D-CNN/LSTM/fNIRS-T; why DL rarely beats features on small data.)_
