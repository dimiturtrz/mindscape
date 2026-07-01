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

_pending — take the M5 quiz to fill this in (numbers to know cold: optical window 650–950 nm, isosbestic
~800 nm, DPF ≈ 6, HRF peak ~5–8 s, Mayer ~0.1 Hz, separation ~3 cm → ~cortex only)._
