# fNIRS Fundamentals — the physics & neurophysiology (teaching module)

**Date:** 2026-07-01
**Purpose:** The hemodynamic counterpart to the EEG-electrophysiology module. Covers the *underlying science* of functional near-infrared spectroscopy — light-in-tissue, the chromophores + modified Beer-Lambert law, neurovascular coupling, optodes/montage, physiological noise, and a cross-modality comparison. Decoding methods are OUT of scope (see `2026-07-01_fnirs_decoding_methods.md`).
**Audience:** taught module + quiz. Intuition-first, with exact quiz-able numbers and citations. Unverified/approximate claims are flagged **[≈]** or **[debated]**.

---

## Intuition in 5 bullets

1. **fNIRS shines dim red/near-infrared light through the skull and measures how much comes back.** Blood is the main thing in the head that absorbs this light, and oxygenated vs deoxygenated hemoglobin absorb *differently* — so the returning light tells you how much of each is present under the optode.
2. **Two colors, two unknowns.** Oxy-Hb (HbO) and deoxy-Hb (HbR) each have a distinct absorption spectrum that cross at ~800 nm (the isosbestic point). Measure absorption change at ≥2 wavelengths, solve 2 equations → get ΔHbO and ΔHbR separately.
3. **Light takes a "banana"-shaped path** from source to detector; it only samples the outer ~1.5–3 cm — scalp, skull, and the cortical surface. **fNIRS sees cortex, never deep structures.** Depth ≈ half the source–detector distance.
4. **The signal is blood flow, not spikes.** Neurons fire → local vessels dilate → fresh oxygenated blood floods in (functional hyperemia) → **HbO rises, HbR falls.** This hemodynamic response is *slow*: peaks ~5–8 s after onset. This is the **same physiology fMRI-BOLD reads** (BOLD ∝ −HbR).
5. **The brain signal is buried in body noise** — heartbeat (~1 Hz), respiration (~0.3 Hz), Mayer waves (~0.1 Hz), and scalp blood flow — much of it in the extracerebral tissue the light must cross twice. Short-separation channels + filtering are how you dig the neural signal out.

---

## 1. Near-infrared light in tissue

### The optical ("therapeutic") window ~650–950 nm
Biological tissue is relatively transparent to light in a narrow band, roughly **650–950 nm** (some sources say ~600/650 up to ~900/1000 nm). The window is bounded by two absorbers:
- **Below ~650 nm:** hemoglobin absorbs strongly (this is *why blood looks red* — it eats blue/green, reflects red; in the deep red/NIR its absorption finally drops).
- **Above ~950 nm:** **water** absorbs strongly (and lipids). Since tissue is mostly water, this sets the upper bound.

In between, both water and hemoglobin absorb weakly, so photons can travel *centimeters* instead of being absorbed within microns. This is what makes trans-cranial optical measurement possible at all. *(Scholkmann 2014; Ferrari & Quaresima 2012.)*

### Absorption vs scattering — scattering dominates
Two things happen to a photon in tissue:
- **Absorption** (μ_a, absorption coefficient): the photon's energy is taken up by a chromophore (Hb, water, lipid, cytochrome). This is the *useful signal* — it carries concentration information.
- **Scattering** (μ_s, scattering coefficient): the photon bounces off refractive-index boundaries (cell membranes, mitochondria, organelles) and changes direction *without losing energy*.

**In brain tissue, scattering >> absorption** — roughly ~100× larger. Photons scatter many times per centimeter (order ~10+ scattering events/cm; the reduced scattering coefficient μ_s′ ≈ 1 mm⁻¹ order). This is the defining fact of the field: light does **not** travel in a straight line. It **diffuses**. Total attenuation = absorption + scattering + geometry, and only the absorption part is informative — the modified Beer-Lambert law (§2) exists precisely to separate them.

### The "banana" path and penetration depth
Because light diffuses, the ensemble of detected photons — those that entered at the source and happened to random-walk their way to a detector a few cm away — occupy a curved, **banana-shaped ("photon banana") volume** arcing from source down into tissue and back up to the detector. The detector preferentially collects photons that dipped down and came back.

- **Depth rule of thumb:** the region of maximum sensitivity sits at roughly **half the source–detector separation** in depth, at the **midpoint** between source and detector laterally. **[≈]** So a 3 cm separation → sensitivity peaks ~1.5 cm deep.
- **Penetration:** on the order of a **few mm up to ~15–20 mm** into the head for adult montages (openfnirs). Enough to reach gyral crowns of cortex through scalp+skull; **not** enough for sulcal depths, white matter, or subcortex.
- **Separation ↔ depth tradeoff:** larger separation → banana reaches deeper into brain, BUT the number of surviving photons falls **exponentially** with distance → weaker, noisier signal. Recommended adult separation is **~2.5–3 cm** (often quoted 3–3.5 cm) as the sweet spot: deep enough to reach cortex, bright enough to measure. Infants use smaller separations (thinner skull → cortex is closer). At ~55 mm separation the banana intersects only the outer ~17 mm of brain but with very few photons. *(Strangman/Colin27 PLoS ONE 2013.)*

### Why only the cortical surface
Two compounding reasons: (1) photon count decays exponentially with pathlength, so deep photons essentially never make it back to a surface detector; (2) the light must pass **twice** through scalp+skull (down and back up), which are themselves absorbing/scattering. Net: **fNIRS is a cortical-surface method (~top 1.5–3 cm)** — great for gyral cortex under the optode (prefrontal, motor, occipital, temporal surface), blind to hippocampus, thalamus, deep nuclei, and sulcal walls.

---

## 2. The chromophores + the modified Beer-Lambert law

### HbO vs HbR spectra and the isosbestic point
The two dominant chromophores that *change* with brain activity are:
- **HbO** = oxygenated hemoglobin (oxy-Hb, HbO₂, O2Hb)
- **HbR** = deoxygenated hemoglobin (deoxy-Hb, HHb, Hb)

Their molar extinction coefficient spectra ε(λ) differ across the optical window:
- Below ~800 nm, **HbR absorbs more** than HbO.
- Above ~800 nm, **HbO absorbs more** than HbR.
- They **cross at the isosbestic point ≈ 800 nm** (commonly cited **~805 nm**; the crossing is broad/near-flat so sources give 798–808 nm). At the isosbestic point absorption is *independent of oxygenation* — it tracks **total hemoglobin** (HbT = HbO + HbR) only. **[≈ exact value ~800–805 nm]**

### Why TWO wavelengths are mandatory
At a single wavelength, a measured absorption change could be caused by *either* a change in HbO *or* HbR (or any mix) — one equation, two unknowns → unsolvable. Using **two wavelengths** (one on each side of the isosbestic point, e.g., a shorter ~660–760 nm where HbR dominates and a longer ~830–880 nm where HbO dominates) gives **two equations, two unknowns**, solvable for both ΔHbO and ΔHbR:

```
ΔA(λ1) = DPF·r·[ ε_HbO(λ1)·ΔC_HbO + ε_HbR(λ1)·ΔC_HbR ]
ΔA(λ2) = DPF·r·[ ε_HbO(λ2)·ΔC_HbO + ε_HbR(λ2)·ΔC_HbR ]
```

Best sensitivity comes from wavelengths spread *away* from 800 nm (poorly chosen pairs both near 800 nm give ill-conditioned equations → "hemoglobin cross-talk," where error in one leaks into the other). Common commercial pairs: 760 & 850 nm, 690 & 830 nm, etc.

### The modified Beer-Lambert law (MBLL)
Classic Beer-Lambert (A = ε·C·L) assumes light travels a straight path length L and there's no scattering — **false in tissue**. The **modified** Beer-Lambert law patches this empirically for a highly scattering medium:

```
ΔOD(λ) = ΔA(λ) = log10(I_baseline / I_measured) = ε(λ) · ΔC · d · DPF(λ)   ( + G )
```

where:
- **ΔOD / ΔA** = change in optical density (attenuation), the measured quantity.
- **d** = geometric source–detector separation (e.g., 3 cm).
- **DPF** = **differential pathlength factor** — the multiplier that turns the straight-line distance d into the true (longer) mean photon path. **True path = d × DPF.** Because scattering makes photons wander, the real path is several times d.
- **G** = an unknown scattering-loss offset term; it **cancels** when we take *differences* (ΔOD), which is exactly why fNIRS reports **changes** (ΔHbO, ΔHbR), not absolute concentrations. This is the key trick and the key limitation.

### DPF / PPF — the approximation that bites
- **DPF (differential pathlength factor):** ratio of mean detected-photon pathlength to inter-optode distance. **Typical adult value ≈ 6** (range often ~5.5–6.5; **~6.26 at 807 nm** for adults per Duncan et al. 1996). It is **wavelength-dependent** (higher at shorter λ), **age-dependent** (infants lower, ~4–5; rises with age), and varies by region and person. **[≈]**
- **PPF (partial pathlength factor):** DPF assumes the whole banana is the same tissue; really the light crosses scalp/skull *and* brain, and only the brain portion carries neural signal. PPF accounts for the fraction of path in the layer of interest and is more correct for layered heads, but harder to know.
- **Why it's an approximation:** in continuous-wave (CW) fNIRS — the common, cheap, portable kind — you measure only *intensity*, so you **cannot measure the true pathlength**; you must *assume* a DPF from tables. A wrong DPF scales your concentrations and, worse, if wrong differently across wavelengths, injects HbO↔HbR cross-talk. Time-domain (TD) and frequency-domain (FD) systems *can* measure pathlength directly (expensive, less portable). **This DPF assumption is the single biggest quantitative caveat of CW-fNIRS.**

### Units
Concentration changes come out in **molar units**: typically **µmol/L (micromolar, µM)** or equivalently expressed with the pathlength baked in as **mmol/L·mm** if DPF is unknown. Real cortical task responses are small — order **~0.1–1 µM** ΔHbO. HbO changes are usually **larger in magnitude** than HbR changes (HbR change ~half of HbO, opposite sign) — a useful sanity check for a genuine hemodynamic response.

---

## 3. Neurovascular coupling — the crux

### From spikes to blood
Neurons don't store energy; when a patch of cortex becomes active, its metabolic demand jumps and, within ~1–2 s, local arterioles **dilate** and **cerebral blood flow (CBF) surges** to that patch. This is **functional hyperemia**, and the machinery linking neural activity → vascular response is **neurovascular coupling** (mediated by astrocytes, nitric oxide, K⁺, adenosine, and neuronal signaling onto smooth muscle/pericytes).

**Crucially the flow increase *overcompensates*** — more oxygenated blood is delivered than the tissue actually extracts. So oxygen supply outstrips demand and the local blood becomes *more* oxygenated during activity. In fNIRS terms:
- **HbO ↑ (increases)** — fresh oxygenated blood floods in. (Larger signal.)
- **HbR ↓ (decreases)** — the washed-in oxy-blood dilutes/flushes out deoxy-Hb. (Smaller, opposite.)
- **HbT ↑** (total blood volume rises).

This canonical **HbO-up / HbR-down** anti-correlated pattern is *the* signature of a cortical activation in fNIRS. (If both move together, suspect systemic/scalp artifact.)

### The hemodynamic response function (HRF)
The temporal shape of this response to a brief stimulus is the **HRF**:
- **Onset delay:** ~**1–2 s** before the hemodynamic response begins after neural onset.
- **Time-to-peak:** rises to a peak at **~5–8 s** (commonly quoted 4–6 s for the rise, plateau/peak ~6–10 s; canonical fMRI HRF peaks ~5 s). **[≈ ranges vary by region/study]**
- **Post-stimulus undershoot:** after the stimulus ends, the response dips **below baseline** and recovers slowly.
- **Return to baseline:** full recovery takes **~15–25 s**. The whole HRF spans ~20–30 s.

### Why fNIRS is intrinsically SLOW
The limiting timescale is **vascular, not optical**. Light and the detector are effectively instantaneous; the *blood* takes seconds to arrive and clear. So even though fNIRS samples fast (often **~10 Hz**, up to 50–100 Hz), its **effective temporal resolution is ~seconds** — you cannot resolve events faster than the HRF blurs them together. This is the fundamental reason fNIRS (like fMRI) can't see millisecond dynamics the way EEG can. Fast sampling still helps: it lets you *see and remove* heartbeat/respiration (§5).

### The "initial dip" [debated]
Some studies report a brief, early **HbR increase / HbO decrease** (an "initial dip") in the first ~1–2 s *before* the big hyperemic HbO rise — interpreted as oxygen extraction by spiking neurons *before* fresh blood arrives, hence more spatially specific to the active neurons. It is **small, inconsistently observed, and controversial** — reliability depends on hardware sensitivity, region, and stimulus. **Do not rely on it; flag as debated in teaching.**

### Same signal as fMRI-BOLD
fMRI **BOLD** (Blood-Oxygen-Level-Dependent) contrast is driven by **deoxyhemoglobin's paramagnetism**: HbR distorts the local magnetic field and shortens T2*. When activity flushes HbR out, the BOLD signal *increases*. So **BOLD ∝ −HbR (approximately)**. fNIRS measures the *same underlying hemodynamics optically* and gives HbR **directly** (plus HbO, which BOLD doesn't isolate). Practical framing for the module: **fNIRS and fMRI read the same neurovascular story; fMRI weights HbR magnetically with mm resolution but is huge/immobile, while fNIRS reads both HbO+HbR optically at the surface, portably.** The two show strong spatial/temporal agreement in validation studies — fNIRS is often called "wearable/portable BOLD."

---

## 4. Optodes + montage

- **Optode** = an optical contact point on the scalp. Two kinds:
  - **Source (emitter):** **LEDs** (cheap, common in CW systems) or **laser diodes** (narrower spectrum), emitting at the chosen 2+ wavelengths.
  - **Detector:** a photodetector — **avalanche photodiodes (APDs)**, **photomultiplier tubes (PMTs)**, or **silicon photodiodes** — that count returning photons.
- **Channel = one source–detector pair.** The measurement is attributed to the **midpoint** between that source and detector (where the banana's sensitivity peaks), at ~half-separation depth. N sources × M detectors within range → many channels; overlapping (high-density) layouts enable **diffuse optical tomography (DOT)** with better spatial resolution.
- **Standard separation ≈ 3 cm** (adults) → samples cortex. This is the "long"/regular channel that carries brain + systemic signal.
- **Short-separation ("short") channels ≈ 8 mm** (ideal ~8.4 mm; generally <15 mm): so close that the banana **never reaches the brain** — it samples **only scalp/extracerebral** blood flow. Used as a **regressor** to remove systemic/scalp contamination from the long channels (§5). Best practice pairs a short channel near each long channel.
- **Montage / cap layout:** optodes are held in a cap, usually referenced to the **EEG 10–20 (or 10–10) system** for reproducible placement, targeting regions of interest (e.g., prefrontal for cognition, C3/C4 motor, occipital for vision). Coverage is limited to where you place optodes and is **cortical-surface only**; whole-head high-density systems exist but trade off channel count, weight, and setup time. Hair is the practical enemy (blocks light) — prefrontal (hairless forehead) is the easiest, most reliable target, which is why so much fNIRS is prefrontal cognition.

---

## 5. Physiological ("systemic") noise

The neural hemodynamic response is *small* and sits on top of much larger body-driven blood-volume oscillations. Because light crosses **scalp and skull twice**, a lot of this noise is **extracerebral** (scalp blood flow) yet lands right in the signal. Main components and frequencies:

| Source | Frequency | Notes |
|---|---|---|
| **Cardiac (heartbeat)** | **~1 Hz (0.8–1.5 Hz)** | Strongest, sharpest peak; every pulse pushes blood through. Actually *useful* — its visibility is a signal-quality check that the optode is coupled. Removable by low-pass filter. |
| **Respiration** | **~0.2–0.3 Hz (0.15–0.4 Hz)** | Breathing modulates blood volume/CO₂. |
| **Mayer waves** | **~0.1 Hz (~0.09–0.1 Hz)** | Spontaneous **arterial blood-pressure** oscillations (baroreflex). **The nastiest confound.** |
| **Very-low-frequency / vasomotion** | **~0.01–0.05 Hz** | Slow drifts, autoregulation. |

### Why this is hard
- **Frequency overlap with the signal.** A typical block/task design produces neural responses in the **~0.01–0.1 Hz** band — which **overlaps Mayer waves (~0.1 Hz), respiration, and VLF drift.** You cannot simply band-pass them out without also attenuating the neural response. Cardiac (~1 Hz) is safely *above* the task band → easy to low-pass; Mayer waves are the ones that hurt because they sit *inside* the task band.
- **Task-correlated systemics.** Worse: cognitive/effortful tasks themselves raise heart rate, blood pressure, and scalp flow — producing systemic changes **time-locked to the task**, which can masquerade as (or swamp) a real cortical activation. This is a notorious source of **false positives** in fNIRS, especially prefrontal cognition.
- **Extracerebral origin.** Much of it is skin/scalp, not brain — hence the value of **short-separation channels** (§4) as a direct measurement of the scalp component to regress out. Other tools: band-pass filtering (e.g., ~0.01–0.1/0.5 Hz), adaptive filtering, PCA/ICA, GLM with short-channel regressors, and simultaneous physiological recording (pulse, respiration belt).

---

## 6. Comparison — fNIRS vs EEG vs fMRI

| Dimension | **fNIRS** | **EEG** | **fMRI (BOLD)** |
|---|---|---|---|
| **Signal source** | Hemodynamic (optical): ΔHbO/ΔHbR absorption of NIR light | Electrical: post-synaptic potentials, scalp voltages | Hemodynamic (magnetic): HbR paramagnetism (BOLD ∝ −HbR) |
| **What it reflects** | Blood oxygenation/volume → *indirect*, downstream of neural activity | Neural activity *directly* (mass post-synaptic currents) | Blood oxygenation → *indirect*, downstream |
| **Temporal resolution** | **~seconds** (limited by HRF, though sampled ~10 Hz) | **~milliseconds** (best) | **~1–2 s** (slowest; TR-limited + HRF) |
| **Spatial resolution** | **~1–3 cm** (mm-scale with high-density DOT) | **~cm** (poor; volume conduction, inverse problem) | **~1–3 mm** (best) |
| **Depth** | **Cortical surface only, ~1.5–3 cm** | Cortical surface (biased to gyri/superficial) | **Whole brain**, incl. deep structures |
| **Portability** | **High** — wearable, battery, some fiberless | **High** — wearable/dry-electrode | **None** — 3–5 tonne magnet, shielded room |
| **Motion tolerance** | **Good** — works standing/walking/infants | Moderate — muscle/movement artifacts | **Very poor** — must lie still |
| **Cost** | **Low–moderate** (one-time, ~$10k–$200k device) | **Low** (cheapest) | **Very high** (~$1M+ scanner; **$500–1000+/scan**) |
| **Silent / safe / non-invasive** | Yes; safe for repeated/infant/bedside | Yes | Loud, claustrophobic, no metal; safe but restrictive |
| **Typical use** | Portable prefrontal cognition, motor cortex, infant/developmental, naturalistic/social, BCI, clinical bedside | ERPs, sleep, epilepsy, fast dynamics, oscillations/BCI | Whole-brain functional mapping, deep structures, clinical imaging |

### What fNIRS is uniquely good at
- **Portable, naturalistic, real-world neuroimaging** — subjects can sit, stand, walk, talk, interact face-to-face, or do tasks outside a scanner. Enables **hyperscanning** (two brains at once), mobile/ambulatory studies.
- **Infants and children** — silent, tolerant of movement, safe for repeated sessions; a workhorse of developmental cognitive neuroscience where fMRI is impractical.
- **Prefrontal cortex cognition** — forehead is hairless → best signal quality; huge literature on working memory, executive function, mental workload.
- **Motor and clinical bedside monitoring** — motor cortex, rehab, stroke, ICU/neonatal oxygenation.
- **Both HbO *and* HbR** — richer hemodynamic picture than BOLD's HbR-only contrast.
- **Middle ground:** better spatial specificity than EEG, better temporal/portability/naturalistic than fMRI. Combines beautifully with EEG (electrical + hemodynamic; different noise, complementary).

### Hard limits (teach these plainly)
- **Cortical surface only** — no deep structures, no sulcal depth. ~top 1.5–3 cm.
- **Slow** — HRF-limited to ~seconds; can't resolve fast neural dynamics (that's EEG's job).
- **Indirect** — measures blood, not neurons; neurovascular coupling can be altered by age, disease, drugs, caffeine, CO₂.
- **Systemic/scalp contamination** — task-correlated blood-pressure/scalp flow → false positives without short-channel correction.
- **Lower spatial resolution** than fMRI; sensitive to hair, optode coupling, head geometry.
- **CW quantification is relative + DPF-approximate** — reports *changes*, not absolute concentrations; DPF assumption scales everything (§2).

---

## References (prefer reviews)

- **Scholkmann, F., et al. (2014).** *A review on continuous wave functional near-infrared spectroscopy and imaging instrumentation and methodology.* NeuroImage 85, 6–27. — https://metabolight.org/wp-content/uploads/2018/02/2014-Scholkmann-et-al.-A-review-on-continuous-wave-functional-near-infrared-spectroscopy-and-imaging-instrumentation-and-methodology.pdf (optical window, isosbestic ~800 nm, scattering, DPF).
- **Ferrari, M., & Quaresima, V. (2012).** *A brief review on the history of human fNIRS development and fields of application.* NeuroImage 63(2), 921–935. — https://pubmed.ncbi.nlm.nih.gov/22510258/
- **Pinti, P., et al. (2020).** *The present and future use of fNIRS for cognitive neuroscience.* Ann. N.Y. Acad. Sci. 1464(1), 5–29. — https://pmc.ncbi.nlm.nih.gov/articles/PMC6367070/ (basics, naturalistic use, challenges).
- **Villringer, A., & Chance, B. (1997).** *Non-invasive optical spectroscopy and imaging of human brain function.* Trends Neurosci. 20(10), 435–442. (foundational optical/hemodynamic).
- **Boas, D. A., Elwell, C. E., Ferrari, M., Taga, G. (2014).** *Twenty years of functional near-infrared spectroscopy: introduction for the special issue.* NeuroImage 85, 1–5.
- **openfnirs — Modified Beer-Lambert Law.** https://openfnirs.org/2024/01/01/modified-beer-lambert-law/ (MBLL equation, DPF definition, 2-wavelength system).
- **openfnirs — Penetration depth.** https://openfnirs.org/2024/01/01/penetration-depth/ (~few mm to 15–20 mm).
- **NIRx Learning Center — DPF vs PPF.** https://help.nirx.de/hc/en-us/articles/20214286260380 (DPF≈6 adult, PPF distinction).
- **Strangman, G. E., et al. (2013).** *Depth sensitivity and source–detector separations for NIRS based on the Colin27 brain template.* PLoS ONE 8(8):e66319. — https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0066319 (separation↔depth, banana).
- **Yücel, M. A., et al. / short-channel regression.** *Short-separation channels & systemic removal.* e.g., PMC7523733, PLoS ONE PMC7757903. (8.4 mm short channels).
- **Mayer-wave characterization (Neurophotonics/bioRxiv 2021).** https://pmc.ncbi.nlm.nih.gov/articles/PMC8652350/ (~0.09–0.1 Hz).
- **Duncan, A., et al. (1996).** *Measurement of cranial optical path length as a function of age using phase resolved NIRS.* Pediatr. Res. — (adult DPF≈6.26 @807 nm, age dependence). *[value cited from field, verify exact table before quizzing on 6.26]*

### Verification flags
- Isosbestic point: **~800–805 nm** — sources vary (798–808); teach "~800 nm." **[≈]**
- DPF adult ≈ 6 (6.26 @807 nm, Duncan 1996) — widely used default; **wavelength/age dependent**, verify exact number before making it a quiz answer. **[≈]**
- HRF time-to-peak "~5–8 s" — spans literature (rise 4–6 s, peak/plateau 6–10 s); region- and study-dependent. **[≈]**
- Initial dip — **[debated]**, do not present as established.
- Penetration "half of source–detector separation" is a rule-of-thumb, not exact; true sensitivity profile is a broad banana. **[≈]**
