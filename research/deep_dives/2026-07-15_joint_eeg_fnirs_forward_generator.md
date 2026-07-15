# Joint EEG + fNIRS Forward Generator — shared-latent source-space simulator

Date: 2026-07-15
Scope: primary-source survey for building a paired EEG+fNIRS forward generator driven by ONE
shared cortical-source latent (parcels → EEG via lead field, fNIRS via HRF + sensitivity). Feeds
the "source-space fusion engine" idea. Honest bottom line up front: the EEG half is fully covered
by `mne.simulation`; the fNIRS half has good HRF tooling but the **cortical-parcel → fNIRS-channel
sensitivity mapping is the load-bearing, least-standard piece** — no off-the-shelf function does
exactly it, and a coarse geometric approximation is defensible for a validation testbed.

---

## 1. MNE-Python EEG source simulation — the recipe (fully covers the EEG half)

Current best API is `mne.simulation.SourceSimulator` → `simulate_raw` → `add_noise`. This is the
canonical, maintained path (stable docs, MNE 1.11).

Sources:
- Generate simulated source data (SourceSimulator example): https://mne.tools/stable/auto_examples/simulation/source_simulator.html
- Generate simulated raw data: https://mne.tools/stable/auto_examples/simulation/simulate_raw_data.html
- Compare simulated vs estimated source activity (validation metrics): https://mne.tools/1.8/auto_examples/simulation/plot_stc_metrics.html

Minimal recipe (reuse your EXISTING fsaverage template forward — no need to rebuild it):

```python
import mne
from mne.simulation import SourceSimulator, simulate_raw, add_noise

# fwd = your existing fsaverage template forward operator (mne.Forward)
# info = the sensor Info you want (montage, sfreq)
src   = fwd["src"]                 # source space lives inside the forward
tstep = 1.0 / info["sfreq"]

ss = SourceSimulator(src, tstep=tstep)
# labels: Desikan-Killiany parcels — you already produce 68 DK parcels
labels = mne.read_labels_from_annot("fsaverage", parc="aparc", subjects_dir=SUBJECTS_DIR)
for label, waveform in zip(active_labels, source_time_series):   # waveform in Ampere-metres
    ss.add_data(label, waveform, events)   # events: (n_events, 3) = [sample, 0, event_id]

raw = simulate_raw(info, ss, forward=fwd)          # lead-field projection to sensors
cov = mne.make_ad_hoc_cov(raw.info)
add_noise(raw, cov, iir_filter=[0.2, -0.2, 0.04])  # colored sensor noise, in place
```

Key facts:
- `SourceSimulator(src, tstep)` — `src` = the source space (get it from `fwd["src"]`).
- `add_data(label, source_time_series, events)` — waveform in **Ampere-metres** (dipole moment);
  `events` shape `(n_events, 3)`, columns `[sample, unused, event_id]`. Call repeatedly, one per
  active parcel; overlapping labels sum. This is exactly the "plant activity on parcels" primitive.
- `simulate_raw(info, source_simulator, forward=fwd)` — reuses your lead field, returns `RawArray`.
  (`simulate_evoked(fwd, stc, info, cov)` is the alternative if you build an `stc` directly and want
  averaged/evoked output instead of continuous raw.)
- `add_noise(raw, cov, iir_filter=...)` — additive sensor noise; `iir_filter` colors it (drift).
  Use `mne.make_ad_hoc_cov(info)` or a real noise cov.
- Validation: `plot_stc_metrics` shows MNE ships spatial-localization error metrics comparing planted
  vs recovered `stc` — directly reusable to score your fusion decoder against known ground truth.

There is nothing missing here for the EEG branch. The waveform generation (ERP/oscillation shapes)
is the only thing MNE leaves to you — that's where SEREEGA concepts (§4) are worth borrowing.

---

## 2. Neurovascular coupling / fNIRS forward — HRF is easy, spatial map is the risk

Two sub-problems: (a) temporal — neural drive → HbO/HbR via HRF convolution (well-tooled);
(b) spatial — cortical source → fNIRS channel sensitivity (the hard part).

### (a) Temporal: HRF convolution — use the SAME shared latent as the neural drive
The neurovascular link papers universally model fNIRS as `HbO(t) = HRF(t) * neural_drive(t)`, where
the canonical HRF is the SPM/Glover **double-gamma** (a peak gamma minus an undershoot gamma). HbR is
the negatively-scaled, slightly-lagged counterpart (typ. HbR ≈ −0.4·HbO with a small extra lag).

Ready-made tooling:
- **`mne_nirs.simulation.simulate_nirs_raw`** — signature:
  `simulate_nirs_raw(sfreq=3.0, amplitude=1.0, annot_desc='A', sig_dur=300.0, stim_dur=5.0,
  isi_min=15.0, isi_max=45.0, ch_name='Simulated', hrf_model='glover')`. Internally convolves a
  boxcar with the HRF (`hrf_model='glover'`, delegates HRF to **nilearn**) and returns an MNE `Raw`
  of simulated haemoglobin. `amplitude` in micromolar; accepts arrays for multiple conditions.
  Docs: https://mne.tools/mne-nirs/stable/ (module `mne_nirs.simulation`), PyPI https://pypi.org/project/mne-nirs/
- For arbitrary continuous drive (not just boxcars), skip `simulate_nirs_raw` and convolve directly
  with **`nilearn.glm.first_level.glover_hrf` / `spm_hrf`** (or `spm_hrf.m`'s double-gamma). This lets
  you feed your shared parcel neural drive (e.g. the envelope/power of the same source waveform that
  drives EEG) straight into the fNIRS branch.
- MNE-NIRS also ships an **autoregressive (AR) noise** fNIRS simulator and notes a **sensitivity /
  Jacobian matrix** can spatially spread local hemodynamic responses across channels — this is the
  hook for the spatial step below. (blog: https://artinis.com/blogpost-all/2021/fnirs-analysis-toolbox-series-mne-python)
- `neuRosim::canonicalHRF` (R) documents the exact double-gamma parameterization if you want to
  reimplement in numpy: https://rdrr.io/cran/neuRosim/man/canonicalHRF.html
- HRF-model background for fNIRS: https://openfnirs.org/2024/01/01/hemodynamic-response-function/ ;
  optimal fNIRS HRF: https://pmc.ncbi.nlm.nih.gov/articles/PMC4468613/

### (b) Spatial: cortical-source → fNIRS-channel sensitivity (THE hard/uncertain piece)
The general fNIRS forward relation is linear: **Y = A · ΔC**, where `A` is the sensitivity (Jacobian)
matrix mapping voxel/vertex absorption (or concentration) change to per-channel optical-density
change. Everything hinges on getting an `A` that maps YOUR 68 DK parcels → fNIRS channels.

Options, cheapest → most rigorous:
- **Coarse geometric approximation (recommended for a testbed; no photon transport):** place fNIRS
  optodes on the scalp (a standard 10-20/10-10 montage co-registered to fsaverage). For each channel
  = source-detector pair, take its scalp midpoint, project inward ~1.5-2 cm to the cortex, and set the
  parcel→channel weight as a Gaussian of the distance from the parcel centroid to that projected point,
  truncated to the banana-shaped region between source and detector. This yields a full parcel×channel
  `A` from geometry alone — defensible, fast, and good enough to validate a fusion decoder that only
  needs "channels near a parcel see it, distant ones don't." Document it as an approximation.
- **Photon-model sensitivity (rigorous, heavier dep):** **Cedalion** (`cedalion.dot.ForwardModel`)
  wraps **pmcx** (Monte-Carlo, GPU) and **NIRFASTer/NIRFASTerFF** (FEM) to compute the fluence per
  voxel and assemble the sensitivity matrix `A` on a real head model; supports precomputed/cached
  fluence so you pay the Monte-Carlo cost once.
  - Cedalion image-reconstruction example (shows `A`, `Adot`): https://doc.ibs.tu-berlin.de/cedalion/doc/dev/examples/head_models/40_image_reconstruction.html
  - Precomputed fluence: https://doc.ibs.tu-berlin.de/cedalion/doc/dev/examples/head_models/46_precompute_fluence.html
  - Cedalion tutorial paper (arXiv 2601.05923): https://arxiv.org/abs/2601.05923
  - NIRFASTerFF (pip-installable FEM photon modeling): https://pmc.ncbi.nlm.nih.gov/articles/PMC12587457/
  - AtlasViewer/HOMER2 Monte-Carlo sensitivity (MATLAB reference for the concept): https://pmc.ncbi.nlm.nih.gov/articles/PMC4478785/
  You would compute `A` once on fsaverage, then reduce it to parcel resolution by averaging/summing the
  voxel columns within each DK parcel → a parcel×channel matrix directly compatible with the EEG side's
  68-parcel latent.
- **Other tools:** NIRS Brainstorm/**Nirstorm** computes the two-wavelength forward model from MRI;
  **fOLD** (https://www.nature.com/articles/s41598-018-21716-z) gives per-ROI optode sensitivity tables
  (region → 10-10 optode) that can seed the geometric weights without any simulation.

---

## 3. Joint / shared-latent EEG+fNIRS simulators — no turnkey one; the coupling pattern is standard

There is **no published, off-the-shelf paired EEG+fNIRS forward simulator driven by one shared source
latent with released code** (unlike the mature single-modality tools). What exists:
- Biophysical multimodal forward models (mostly EEG-fMRI lineage) that explicitly model the
  **neurovascular coupling cascade**: a shared electrophysiological latent drives EEG linearly (lead
  field) and hemodynamics via an HRF/balloon model. Review framing: https://www.jneurosci.org/content/32/18/6053
- **EEG-informed HRF modeling for fNIRS**: EEG spectral-envelope / band-power is mapped to
  hemodynamics via **gamma transfer functions**, validated on *simulated* EEG-fNIRS data. This is the
  closest published "one latent → both modalities" recipe. https://ieeexplore.ieee.org/document/9959633/
  and broadband NIRS+EEG neurovascular/neurometabolic framework: https://www.nature.com/articles/s41598-021-83420-9
- Subject-specific EEG-fNIRS neurovascular coupling via double-gamma HRF parameter fits (confirms the
  double-gamma link is the accepted coupling form): https://www.researchgate.net/publication/377470268
- Empirical simultaneous EEG-fNIRS datasets/studies for realism targets (ERP + decision hemodynamics):
  https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0325017 ;
  lower-limb bimodal fusion: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8227788/

**Standard coupling recipe papers converge on** (adopt this):
1. one shared neural drive per parcel `s_p(t)` (the latent ground truth);
2. EEG branch: EEG dipole moment ∝ `s_p(t)` directly → lead field → sensors + sensor noise;
3. fNIRS branch: neural drive for hemodynamics = a nonneg function of the SAME latent — typically the
   **band-limited power / envelope** of `s_p(t)` (fNIRS responds to slow metabolic demand, not signed
   mV), convolved with the double-gamma HRF, then spread to channels via `A`, plus a **lag** (HRF peak
   ~5-6 s) and modality-specific (autocorrelated/physiological) noise;
4. add physiological confounds to fNIRS (Mayer wave ~0.1 Hz, respiration, cardiac) since real fusion
   must separate those from neural hemodynamics.

So: couple through a **linear/positive shared drive**, keep forwards and noise fully modality-specific,
add the HRF lag. That's the whole engine.

---

## 4. SEREEGA — MNE covers the EEG forward; borrow SEREEGA's *signal-shape* concepts only

SEREEGA (Krol et al., *J. Neurosci. Methods* 2018) is a MATLAB/EEGLAB toolbox for simulating
event-related EEG: define dipolar "brain components" (position+orientation) on a lead field, assign
each a signal, project+sum to scalp. Paper: https://pubmed.ncbi.nlm.nih.gov/30114381/ ;
code: https://github.com/lrkrol/SEREEGA ; open PDF: https://www.biorxiv.org/content/10.1101/326066v1.full

Verdict: **do not port the forward machinery** — `mne.simulation` (§1) already does dipole-on-lead-field
projection, and better integrates with your fsaverage/DK pipeline. SEREEGA's value is its **signal
class library** — the part MNE leaves to you:
- **ERP** class (parameterized peaks: latency, width, amplitude) — for evoked waveforms;
- **oscillation / ersp** class (base frequency, burst/AM modulation, event-related
  synchronization/desynchronization) — for band-power waveforms, which are ALSO exactly what you need
  as the fNIRS neural-drive envelope;
- **noise** classes (brown/pink/white) and **autoregressive** signals;
- systematic **trial-to-trial variability / deviations** (jitter, amplitude slope) — useful realism.
Reimplement these few generators in numpy (they're small, closed-form) to produce the `source_time_series`
fed to `SourceSimulator.add_data` and, via their envelope, the fNIRS drive. That's the only piece worth
taking from SEREEGA.

---

## Concrete minimal architecture

```
                     shared latent: per-parcel neural drive  s_p(t)   [68 DK parcels]
                     (waveforms from numpy'd SEREEGA-style ERP + oscillation generators)
                                   |
             ┌─────────────────────┴───────────────────────┐
             │ EEG BRANCH                                   │ fNIRS BRANCH
             │ waveform = s_p(t)  (Ampere-metres)           │ drive = envelope/band-power of s_p(t)  (>=0)
             │ SourceSimulator(src, tstep)                  │ HbO_p(t) = double_gamma_HRF * drive     (nilearn glover_hrf)
             │   .add_data(label_p, s_p, events)            │ HbR_p(t) = -0.4 * HbO_p  (+ small extra lag)
             │ simulate_raw(info, ss, forward=fwd)          │ Y_channel = A @ [HbO_p ; HbR_p]         (A = parcel×channel sensitivity)
             │ add_noise(raw, ad_hoc_cov, iir_filter=...)   │ + AR/physiological noise (Mayer 0.1Hz, cardiac, resp)
             └─────────────────────┬───────────────────────┘
                                   |
                    paired EEG sensors + fNIRS channels, SHARED known ground truth = s_p(t) on parcels
```

Function checklist:
- EEG: `mne.simulation.SourceSimulator` / `.add_data` / `mne.simulation.simulate_raw` /
  `mne.simulation.add_noise` / `mne.make_ad_hoc_cov`; reuse existing `fwd` (`fwd["src"]`).
- fNIRS temporal: `nilearn.glm.first_level.glover_hrf` (or `mne_nirs.simulation.simulate_nirs_raw` for
  the boxcar/quick case); HbR = scaled/lagged HbO.
- fNIRS spatial `A`: START with the geometric Gaussian-banana approximation (optode midpoint → cortical
  projection → distance kernel to parcel centroid); UPGRADE to `cedalion.dot.ForwardModel` (pmcx /
  NIRFASTerFF) computed once on fsaverage and parcel-reduced, if you need physically-grounded sensitivity.
- Validation: MNE `plot_stc_metrics` spatial-error metrics on the recovered vs planted `stc`.

## Riskiest / most-uncertain piece
**The parcel → fNIRS-channel sensitivity matrix `A`.** Everything else is a maintained API call. There
is no drop-in "DK-parcel → fNIRS-channel" function, and the honest options split into (i) a coarse
geometric kernel — cheap, defensible for a decoder testbed, but not photon-accurate; or (ii) a full
Monte-Carlo/FEM photon model via Cedalion/NIRFASTerFF — accurate but a heavy new dependency and a
voxel→parcel reduction step. Recommendation: ship the geometric `A` first (it's sufficient to prove the
fusion decoder recovers shared ground truth), keep the Cedalion path as an opt-in higher-fidelity mode.
Second-order risk: the exact latent→hemodynamic-drive transform (signed waveform vs its band-power
envelope, and the EEG:fNIRS amplitude/lag coupling constants) — pick band-power + double-gamma lag per
the neurovascular-coupling literature and treat the coupling gain as a documented knob, not a fitted number.
```
