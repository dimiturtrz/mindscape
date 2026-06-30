# M4 · Neurophysiology → why the EEG signal is the way it is

The bio bedrock — *enough to explain the signal and its failure modes*, not wet-lab neuroscience.
Grounded in [`research/deep_dives/2026-06-30_motor-imagery-neuroscience.md`](../research/deep_dives/2026-06-30_motor-imagery-neuroscience.md).
Reference: any intro EEG/neuro text; Buzsáki *Rhythms of the Brain* for depth.

## The core ideas
- **Source.** Scalp voltage = the summed **postsynaptic potentials (PSPs)** of **aligned pyramidal neurons**
  firing in **synchrony** — NOT their action-potential spikes. Spikes are too brief/async to sum; slow PSPs
  of perpendicular-to-cortex (parallel) cells add their dipoles into a measurable field (~µV).
- **Volume conduction** — the field spreads through brain/skull/scalp (skull smears most), so each electrode
  records a **blurred MIXTURE** of sources, **zero-lag** (EM field ~light speed, not a slow acoustic wave).
  *This is the single most consequential fact:* it's why CSP/spatial-filtering exists, why zero-lag
  covariance is the right tool, and why localization is hard.
- **Why slow.** The measured signal is the sum of **slow PSPs** + population **synchrony** → rhythms in the
  slow bands (<~100 Hz). EEG literally can't see ~1 ms spikes (they don't sum coherently). → 250 Hz suffices.
- **Spatial resolution + the inverse problem.** Few electrodes, but the *fundamental* limit is volume-
  conduction smearing. Recovering brain sources from scalp voltages = the **inverse problem** — ill-posed
  (many source configurations → the same scalp pattern, no unique solution).
- **Artifacts (often bigger than signal).** **EOG** (eyes/blinks — eyeball is a dipole; usually the worst),
  **EMG** (muscle; broadband high-freq), **line noise** (50/60 Hz), **motion/sweat/electrode drift**, ECG.
  → preprocessing (band-pass + artifact handling) is half the battle.
- **Modality tradeoff.** EEG: fast (ms) + portable + cheap, spatially poor → *the* real-time BCI tool. MEG:
  ms + better spatial (skull doesn't smear magnetism), not portable. fMRI: mm spatial, seconds-slow
  (hemodynamic), huge. fNIRS: wearable hemodynamic (light → blood oxygen), slow + shallow.

## The synthesis (why M4 underpins M1–M3)
EEG is a **blurred, slow, low-SNR, population-level** measurement. Every downstream choice follows: **filter**
(low-SNR + slow → M1), **spatially unmix via CSP** (volume conduction → M2), **evaluate honestly under
per-subject anatomy** (smearing + idiosyncrasy → M3), and the **cross-subject gap is physics + biology**, not
a bug. Interview answer to "why is EEG decoding hard / why doesn't it transfer": volume conduction (mixed +
poorly localized) + low SNR + per-person anatomy.

---

## Quiz log

### 2026-06-30 — quiz M4 (re-graded after a fairer read)
**Score ~4/6 — strong intuitions on the newest field.**

1. *Source* — ✓✓ pyramidal + many-because-tiny + **"fire in clusters" (synchrony)**. Added: PSPs-not-spikes,
   cell alignment.
2. *Mixing* — ✓✓ "outside looking in" (valid remote-sensing intuition) + **zero-lag = EM not acoustic** (the
   deep point, correct). Added the term: volume conduction.
3. *Why slow* — ✓ "coordinated firing of thousands" (synchrony half). Added: slow PSPs not fast spikes.
4. *Spatial res* — ✓ "few + distant." Added: volume-conduction smearing as the real limit + the **inverse
   problem**.
5. *Artifacts* — said "no idea" but **guessed gel (electrode drift) + movement (motion) — both real**. Added:
   EOG (eyes), EMG (muscle), line noise.
6. *Tradeoff* — ✓✓ EEG temporal + fMRI spatial/huge; noted MEG uncovered; **proactively raised fNIRS + its
   Lys relevance**. Filled in the full EEG/MEG/fMRI/fNIRS table.

**New additions to bank:** PSPs-not-spikes, the **inverse problem** (name), the EOG/EMG/line-noise artifact
trio, the modality tradeoff table.
