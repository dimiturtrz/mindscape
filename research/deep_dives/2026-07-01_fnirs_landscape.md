# fNIRS & Hybrid EEG-fNIRS Decoding — Landscape for the mindscape BCI Build

**Date:** 2026-07-01
**Purpose:** Fact-finding for a data adapter + job-application portfolio piece. Prioritizes correctness and citations. Unverified items are flagged explicitly.

---

## What's load-bearing for the mindscape build (read this first)

1. **There are TWO separate Shin datasets, do not conflate them.**
   - **Shin 2017** (IEEE TNSRE) = the "Open Access Dataset for EEG+NIRS Single-Trial Classification". **29 subjects.** Two paradigms in ONE recording session set: **Dataset A = left vs. right hand motor imagery (MI)**, **Dataset B = mental arithmetic (MA) vs. rest**. This is the one exposed in **MOABB** as `Shin2017A` / `Shin2017B`.
   - **Shin 2018** (Scientific Data, `sdata2018.3`) = "Simultaneous acquisition of EEG and NIRS during cognitive tasks". **26 subjects.** Three cognitive tasks: **n-back, discrimination/selection response (DSR), word generation (WG)**. **No motor imagery.** Distinct download, distinct license.
   - So if the build needs **MI + MA**, target **Shin 2017**. If it needs **workload/n-back**, target **Shin 2018** (or Tufts fNIRS2MW).

2. **MOABB loads only the EEG half of Shin2017 today. fNIRS is NOT implemented** (hard `NotImplementedError`). A fNIRS adapter must go direct to the `.mat` files from the TU Berlin repository — MOABB will not give you HbO/HbR.

3. **Canonical mne preprocessing** (all functions live in **core `mne.preprocessing.nirs`**, not mne-nirs): `optical_density` → `scalp_coupling_index` (drop SCI<0.5) → `temporal_derivative_distribution_repair` (TDDR, motion) → `beer_lambert_law(ppf=...)` → `filter(0.05, 0.7)` → epoch. **Watch the `ppf` gotcha** (see §2).

4. **Short-separation channels are the credibility axis.** Uncontrolled scalp/Mayer-wave contamination inflates decoding accuracy. Shin 2017/2018 use **only 30 mm long-separation channels — no short channels** → cannot do short-channel regression on them. Mention this limitation explicitly in the writeup; it signals you understand the confound.

5. **Riemannian/pyRiemann on fNIRS covariances is real, recent, and publishable-adjacent** (Näher et al. 2025, Neurophotonics; `HybridBlocks` estimator upstreamed to pyRiemann). Strong differentiator for a portfolio.

---

## 1. Shin datasets — exact contents & how to load

### Shin 2017 — "Open Access Dataset for EEG+NIRS Single-Trial Classification"
Shin, von Lühmann, Blankertz, Kim, Jeong, Hwang, Müller. *IEEE Trans. Neural Syst. Rehabil. Eng.* 25(10):1735–1745, 2017.

| Spec | Value |
|---|---|
| Subjects | **29** (14 M / 15 F) |
| Paradigms | **Dataset A:** left vs. right hand **motor imagery**. **Dataset B:** **mental arithmetic** vs. rest |
| Sessions | 3 per subject |
| Trials | 30 per class (10 × 3 sessions); MOABB reports **5,220 total trials** across subjects |
| Trial length | **10 s** task window (per MOABB) |
| EEG | **30** active electrodes, 10-5 system, linked-mastoid ref, **1000 Hz** (MOABB serves it **downsampled to 200 Hz**) |
| fNIRS | **36 channels**, **14 sources / 16 detectors**, **~12.5 Hz** |
| Aux | 2× EOG; ECG + respiration recorded, not analyzed |
| Format | `.mat` |
| License | **GPL-3.0** (per MOABB metadata) |

**Loading:**
- **EEG only** via MOABB:
  ```python
  from moabb.datasets import Shin2017A   # or Shin2017B
  ds = Shin2017A(accept=True)
  data = ds.get_data(subjects=[1])
  ```
  `Shin2017A` = MI, `Shin2017B` = MA. Paradigm string is `"imagery"`.
- **fNIRS is NOT available through MOABB.** `moabb/datasets/bbci_eeg_fnirs.py` sets `self.fnirs = fnirs  # TODO: actually incorporate fNIRS somehow` and `BaseShin2017.__init__` raises `NotImplementedError('Fnirs not implemented.')` when `fnirs=True`. To get HbO/HbR you must download the raw `.mat` from TU Berlin and parse it yourself (this is exactly where the mindscape adapter adds value).
- Direct download / project page: **http://doc.ml.tu-berlin.de/hBCI/** (BBCI hybrid-BCI portal).

### Shin 2018 — "Simultaneous acquisition of EEG and NIRS during cognitive tasks for an open access dataset"
Shin, von Lühmann, Kim, Mehnert, Hwang, Müller. *Scientific Data* 5:180003 (`sdata2018.3`), 2018.

| Spec | Value |
|---|---|
| Subjects | **26** right-handed (9 M / 17 F, 26.1±3.5 yr) |
| Tasks | **Dataset A: n-back** (0-, 2-, 3-back); **Dataset B: DSR** (discrimination/selection response, 'O' vs 'X'); **Dataset C: word generation (WG)** vs baseline. **No motor imagery.** |
| EEG | **30** electrodes, 10-5 system, ref **TP9** / gnd **TP10**, **1000 Hz** |
| fNIRS | **36 channels**, **16 sources / 16 detectors**, **30 mm** S-D distance (all channels), **10.4 Hz** (⚠ note: MOABB's Shin2017A page cites 12.5 Hz for the 2017 set — the two datasets differ; 2018 = 10.4 Hz) |
| Timing (A/B) | 2 s instruction + **40 s task** + 20 s rest; **180 trials** (=20 × 3 series × 3 sessions) |
| Timing (C) | **10 s task** + 13–15 s rest; **60 trials** (30 WG + 30 baseline) |
| Format | `.mat` (MATLAB, vendor-agnostic) |
| License | **CC-BY-4.0** |
| Repository | http://doc.ml.tu-berlin.de/simultaneous_EEG_NIRS/ and DOI **10.14279/depositonce-5830.2** |

**Neither Shin set is distributed as SNIRF or BIDS** — both are custom `.mat`. No official mne-nirs sample loader for either (mne-nirs ships different sample data). Adapter must map `.mat` → mne `RawArray` manually.

---

## 2. Canonical fNIRS preprocessing with mne / mne-nirs

Reference: MNE "Preprocessing fNIRS data" tutorial (`auto_tutorials/preprocessing/70_fnirs_processing.html`, verified against MNE stable / 1.12.1). **All core functions live in `mne.preprocessing.nirs`** (shipped with core MNE, not only mne-nirs).

Standard chain, in order:

```python
import mne
# 1. raw intensity -> optical density
raw_od = mne.preprocessing.nirs.optical_density(raw_intensity)

# 2. channel-quality: scalp coupling index; drop poorly-coupled channels
sci = mne.preprocessing.nirs.scalp_coupling_index(raw_od)
raw_od.info['bads'] = list(compress(raw_od.ch_names, sci < 0.5))   # SCI < 0.5 = bad

# 3. motion correction: TDDR (parameter-free)
raw_od = mne.preprocessing.nirs.temporal_derivative_distribution_repair(raw_od)

# 4. Beer-Lambert -> HbO/HbR concentrations
raw_haemo = mne.preprocessing.nirs.beer_lambert_law(raw_od, ppf=0.1)

# 5. bandpass (removes drift + heart-rate ~1 Hz; HRF content is < ~0.5 Hz)
raw_haemo.filter(0.05, 0.7, h_trans_bandwidth=0.2, l_trans_bandwidth=0.02)

# 6. epoch around events
```

**Verified function names / parameters:**
- `mne.preprocessing.nirs.optical_density` — exists, current.
- `mne.preprocessing.nirs.scalp_coupling_index` — exists; conventional bad-channel cutoff **SCI < 0.5**.
- `mne.preprocessing.nirs.temporal_derivative_distribution_repair` (**TDDR**) — exists in **current MNE (docs confirmed up to 1.12.1 and mne-nirs 0.7.1)**. Parameter-free (removes baseline shift + spike artifacts on OD data). Note: TDDR is **absent from the specific MNE 70_fnirs tutorial's minimal example**, but it is the standard motion-correction step and a first-class function — add it between OD and Beer-Lambert.
- `mne.preprocessing.nirs.beer_lambert_law(raw_od, ppf=...)` — exists.
  - **⚠ ppf gotcha (load-bearing):** MNE's **default `ppf=0.1`**, whereas Homer/physical differential-pathlength-factor convention is **`ppf≈6`**. MNE's 0.1 default does NOT reproduce Homer-scaled concentrations. This is a documented, frequently-tripped discrepancy. For decoding it usually doesn't matter (linear scaling), but for any cross-tool comparison or absolute-concentration claim, set `ppf` deliberately and state it. Newer MNE has moved toward a physically-meaningful default — **verify the exact default in the pinned MNE version at build time** (flagged: version-dependent).
- Bandpass: tutorial uses **`filter(0.05, 0.7 Hz)`**. Common alternative in the literature is **0.01–0.2 Hz** (tighter, kills Mayer waves at ~0.1 Hz less well but removes more physiology) — both appear; 0.01–0.2 is more aggressive on the low end. Choice interacts with Mayer-wave handling (§4).

**mne-nirs (the add-on package)** provides the higher-level pieces core MNE lacks: **GLM analysis** (`mne_nirs.statistics.run_glm`), **short-channel regression** (`mne_nirs.signal_enhancement.short_channel_regression`), and signal-enhancement utilities. Use mne-nirs when doing GLM/SS-regression; use core `mne.preprocessing.nirs` for the OD/BL/TDDR chain above.

---

## 3. fNIRS decoding accuracy from the literature

All figures are **task-dependent and pipeline-dependent**; treat as ranges, not benchmarks. Deep-learning numbers are frequently optimistic (small subject counts, within-subject CV, possible leakage). Binary **task-vs-rest** classification has a **high ceiling (~90–96%)** because the hemodynamic response vs. flat baseline is an easy contrast; **task-vs-task** (e.g. left vs right MI) is much harder (**~65–77%**).

### Motor imagery (mostly left-vs-right hand)
- LDA on HbO/HbR features: **~66–77%** (one report: HbO 71.32%, HbR 77.01% with feature selection + LDA; other LDA runs 65.96% / 67.00%).
- CNN / LSTM: **~70–78%** (CNN-LSTM hybrid avg **78.44%**; plain CNN/LSTM up to ~70.69%).
- Dilation CapsuleNet (MI + MA): **74.03%–95.01%** across datasets (upper end likely easy contrasts / within-subject).
- **Easy contrasts** (finger tapping vs rest): **96.3%**; tapping vs imagined tapping **80.1%**; treadmill-walk vs rest (LSTM) **78.97%**.

### Mental arithmetic / workload (vs rest or across n-back levels)
- LDA / shrinkage-LDA: **~66–80%** (MA vs rest ~71%; shrinkage-LDA 66.08% in one CNN comparison; two-class workload LDA ~80%).
- CNN: **up to 91.96%** (MA); GASF-image CNN **87.14%**; spectrogram-CNN **82.77%**; MLP two-class workload **96%**.
- Stress vs relaxation (hand-crafted features + CNN): **98.69%** (very easy contrast).

**Cross-subject / generalization** is the honest metric and is much lower — this is the whole point of the **Tufts fNIRS2MW benchmark** (§5), which is explicitly built to test *generalization to a new subject* (reduces calibration). Within-subject numbers above do not transfer.

Key surveys: **"Deep Learning in fNIRS: a review"** (arXiv:2201.13371 / PMC9301871); **"Benchmarking framework for machine learning classification from fNIRS data"** (PMC10790918).

---

## 4. Systemic physiology & short-separation channels

**The problem:** long-separation (~30 mm) channels sample **both** cerebral hemodynamics **and** superficial scalp/systemic signals — heartbeat (~1 Hz), respiration (~0.2–0.3 Hz), and **Mayer waves (~0.1 Hz, range 0.05–0.15 Hz)**. Mayer waves **spectrally overlap** the task hemodynamic response in block designs and are the dominant confound. Documented effect: Mayer-wave amplitude >~1 µM at 0.1 Hz **triples the MSE** of the estimated HRF (Yücel et al., PubMed 27570699).

**Why it inflates decoding accuracy if uncontrolled:** systemic signals are often **time-locked to task blocks** (task-evoked blood-pressure / arousal changes on the scalp), so a classifier can exploit **extra-cerebral** signal and report high accuracy that is **not neural**. This is the #1 fNIRS-decoding validity criticism — naming it is a credibility signal.

**How it's controlled:**
- **Short-separation (SS) channels** (~8 mm S-D): sample scalp only. **Regress** the SS signal out of long channels (short-channel regression, SCR) or add SS as **nuisance regressors in a GLM**.
- Best-practice: include **all** SS channels in the GLM with **orthogonalization** (multiple studies). Advanced: **temporally-embedded Canonical Correlation Analysis (tCCA)** GLM extension (von Lühmann et al., NeuroImage, ScienceDirect S1053811919310638).
- Reported effect: SCR raised HRF **reproducibility 0.64 → 0.81** but in one classification study **did not change accuracy** (85%) — i.e. SS mainly buys *validity/reproducibility*, not always raw accuracy.

**Dataset implication:** **Shin 2017 and Shin 2018 have NO short-separation channels** (all Shin-2018 channels are 30 mm long-separation). So SS-regression is impossible on Shin data — physiology can only be attenuated by band-pass / GLM temporal modeling. **Tufts fNIRS2MW** likewise is a **2-location forehead** rig (no SS). Datasets that *do* include SS channels tend to be the newer BIDS/SNIRF hd-DOT / whole-head sets and lab-specific NIRx recordings; call this out as a limitation of the chosen dataset in the writeup.

---

## 5. Datasets beyond Shin 2018 (2019–2025, open access)

### Tufts fNIRS2MW — mental workload benchmark (recommended for the generalization story)
Huang et al., "The Tufts fNIRS Mental Workload Dataset & Benchmark for BCIs that Generalize," *NeurIPS 2021 Datasets & Benchmarks*.

| Spec | Value |
|---|---|
| Subjects | **68 recommended** (87 collected) |
| Task | **n-back**: 0-, 1-, 2-, 3-back (working-memory load) |
| Sensor | **2 forehead locations** (AB, CD); **8 real-valued features/timestep** = intensity + phase of oxy/deoxy Hb per location (`AB_I_O, AB_PHI_O, AB_I_DO, AB_PHI_DO, CD_I_O, CD_PHI_O, CD_I_DO, CD_PHI_DO`, µmol/L) |
| Sampling | **5.2 Hz** |
| Window | **30 s** default (stride 0.6 s); alt windows 2/5/10/20/40 s provided |
| Volume | ~**21 min** usable data/subject |
| Labels | 4-level n-back (also used as binary low/high load) |
| Format | **CSV** (chunk ID, label, 8 features) |
| License | **CC-BY-4.0** |
| Access | Box.com via https://tufts-hci-lab.github.io/code_and_datasets/fNIRS2MW.html |

Explicitly designed for **cross-subject generalization** (train on many, test on held-out new subject) — the right dataset to demonstrate calibration-free BCI. Note: **no short channels**, only 2 forehead spots — good for ML story, limited for physiology story. A follow-on **NIRS-X / fNIRS2MW audio n-back** set also exists from the same lab.

### Other open fNIRS / hybrid sets
- **BCI Competition fNIRS** data (Berlin) — older, small; superseded by Shin releases.
- **HEFMI-ICH** (Nature *Scientific Data*, 2025, s41597-025-06100-7) — hybrid EEG-fNIRS **motor imagery** in **intracerebral-hemorrhage patients**; clinical population, recent.
- **Multimodal fNIRS-EEG unilateral limb MI** dataset (arXiv 2602.04299) — recent MI set (verify specs before use — **unverified** here).
- **SNIRF / BIDS-fNIRS collections:** SNIRF is the community standard binary format (SfNIRS/Society for fNIRS); **BIDS has an official fNIRS extension**; OpenNeuro hosts a growing number of BIDS-fNIRS datasets (many *with* short channels, unlike Shin). If the build wants SNIRF-native + SS channels, search **OpenNeuro "fNIRS"** — specific dataset IDs **not enumerated here (unverified)**.

---

## 6. MOABB fNIRS support & Riemannian geometry on fNIRS

### MOABB
- **fNIRS is not supported for Shin2017.** `moabb/datasets/bbci_eeg_fnirs.py`: `self.fnirs = fnirs  # TODO: actually incorporate fNIRS somehow`; `BaseShin2017.__init__` raises `NotImplementedError('Fnirs not implemented.')`. Classes: `BaseShin2017` (not for direct use), `Shin2017A` (MI), `Shin2017B` (MA), paradigm `"imagery"`.
- Practical consequence: **MOABB is an EEG-only convenience here.** No general fNIRS paradigm exists in MOABB. Any fNIRS work is DIY off the raw `.mat`/SNIRF.

### Riemannian geometry / pyRiemann on fNIRS — YES, and it's recent
**Näher, Bhat, et al., "Riemannian geometry boosts functional near-infrared spectroscopy-based brain-state classification accuracy," *Neurophotonics* 12(4):045002, 2025** (also bioRxiv 2024.09.06.611347).
- Method: build **covariance/kernel matrices per chromophore**, combine HbO + HbR as **block-diagonal ("super kernel") matrices**, classify in tangent space. Tested correlation, covariance, **Ledoit-Wolf shrinkage**, polynomial, RBF, Laplacian, cosine kernels.
- **pyRiemann used**; their estimator upstreamed as **`HybridBlocks`** (in pyRiemann docs).
- Results (7 subjects, 8 mental-imagery tasks, 96 trials each):
  - **8-class:** Riemannian **65%** vs traditional **42%** (+23 pts).
  - **2-class (28 pairwise):** Riemannian **96%** vs traditional **78%** (+18 pts).
- Related: Riemannian **transfer learning** for calibration reduction is well-established on EEG (arXiv:2111.12071) and the same tangent-space + transfer idea is what makes fNIRS covariances attractive for cross-subject.

**Portfolio angle:** applying pyRiemann `HybridBlocks` (or a hand-rolled block-diagonal HbO/HbR covariance → tangent-space → LDA/SVC) to Shin 2017 fNIRS is a concrete, defensible, current technique that most fNIRS pipelines don't use.

---

## References (URLs)

**Shin datasets**
- Shin 2017 (TNSRE): https://pubmed.ncbi.nlm.nih.gov/27849545/ · IEEE Xplore PDF: http://ieeexplore.ieee.org/iel7/7333/8082143/07742400.pdf
- Shin 2018 (Sci Data): https://www.nature.com/articles/sdata20183 · PMC: https://pmc.ncbi.nlm.nih.gov/articles/PMC5810421/ · repo: http://doc.ml.tu-berlin.de/simultaneous_EEG_NIRS/ · DOI 10.14279/depositonce-5830.2
- BBCI hybrid-BCI portal: http://doc.ml.tu-berlin.de/hBCI/
- MOABB Shin2017A docs: https://moabb.neurotechx.com/docs/generated/moabb.datasets.Shin2017A.html
- MOABB source (fNIRS TODO): https://github.com/NeuroTechX/moabb/blob/develop/moabb/datasets/bbci_eeg_fnirs.py

**Preprocessing**
- MNE fNIRS tutorial: https://mne.tools/stable/auto_tutorials/preprocessing/70_fnirs_processing.html
- TDDR: https://mne.tools/stable/generated/mne.preprocessing.nirs.temporal_derivative_distribution_repair.html
- beer_lambert_law: https://mne.tools/stable/generated/mne.preprocessing.nirs.beer_lambert_law.html
- Homer→MNE migration (ppf discussion): https://mne.tools/mne-nirs/stable/auto_examples/migration/plot_01_homer.html

**Decoding accuracy**
- Deep Learning in fNIRS review: https://pmc.ncbi.nlm.nih.gov/articles/PMC9301871/ · arXiv: https://arxiv.org/pdf/2201.13371
- Benchmarking ML fNIRS: https://pmc.ncbi.nlm.nih.gov/articles/PMC10790918/
- Dilation CapsuleNet (MI+MA): https://www.tandfonline.com/doi/full/10.1080/27706710.2024.2335886

**Systemic physiology / short channels**
- Mayer waves reduce HRF accuracy (Yücel): https://pubmed.ncbi.nlm.nih.gov/27570699/
- Short-channel regression heterogeneity: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7523733/
- tCCA GLM (von Lühmann): https://www.sciencedirect.com/science/article/pii/S1053811919310638
- SCR reproducibility: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8901194/

**Datasets beyond Shin**
- Tufts fNIRS2MW: https://tufts-hci-lab.github.io/code_and_datasets/fNIRS2MW.html · NeurIPS: https://datasets-benchmarks-proceedings.neurips.cc/paper/2021/hash/bd686fd640be98efaae0091fa301e613-Abstract-round2.html
- HEFMI-ICH: https://www.nature.com/articles/s41597-025-06100-7

**Riemannian on fNIRS**
- Näher et al. 2025 (Neurophotonics): https://pmc.ncbi.nlm.nih.gov/articles/PMC12523035/ · bioRxiv: https://www.biorxiv.org/content/10.1101/2024.09.06.611347.full.pdf
- Riemannian transfer learning (EEG): https://arxiv.org/pdf/2111.12071

---

## Open / unverified items to close at build time
- Exact **MNE `beer_lambert_law` default `ppf`** in the pinned version (moved from 0.1 toward physical value in recent releases — pin & check).
- Whether the raw Shin-2017 `.mat` stores **raw intensity, OD, or already-Beer-Lambert HbO/HbR** (determines where the adapter enters the chain) — inspect the `.mat` directly.
- Specific **OpenNeuro BIDS-fNIRS dataset IDs** that include **short-separation channels** (not enumerated here).
- Multimodal fNIRS-EEG unilateral-limb MI set (arXiv 2602.04299) specs — unverified.
