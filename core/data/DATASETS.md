# Data catalog — what's wired, what's known

Operational registry: which datasets have an adapter, where the raw lives, and how to fetch it. This is the
*plumbing* view — for the **science** (specs, SNR, leakage, split rationale) see the research deep-dives,
linked per row. One home per thing: numbers live in research, wiring status lives here.

Raw data lives **outside the repo** at `<data>/raw/<root>/` (`<data>` from gitignored `paths.yaml`;
`core.config.raw_dir()`). MOABB/MNE sets auto-download there on first `get_data`; others are fetched with the
command in the **Get** column (a `download()` in the adapter module, or a documented manual step).

## Wired (adapter exists, in use)

| Dataset | Modality · task | Source | Get | Adapter |
|---|---|---|---|---|
| **BCI IV-2a** | EEG · motor imagery | MOABB `BNCI2014_001` | auto (MOABB) | `eeg/bnci2014_001.py` |
| **Shin 2018 (EEG)** | EEG · n-back workload | TU-Berlin BBCI `.mat` | manual zip → `raw/shin2017_eeg/` | `eeg/shin2017_nback_eeg.py` |
| **Shin 2018 (fNIRS)** | fNIRS · n-back workload | TU-Berlin BBCI `.mat` | manual zip → `raw/shin2017/` | `fnirs/shin2017.py` |

## Known, not wired (cataloged — fetch + adapter pending)

### EEG→image / visual-semantic (Stage-3 target)
→ specs, SNR, labelling, leakage: [`research/deep_dives/2026-07-07_eeg_visual_semantic_decoding_datasets.md`](../../research/deep_dives/2026-07-07_eeg_visual_semantic_decoding_datasets.md)

| Dataset | Modality | Source | Get | Status |
|---|---|---|---|---|
| **THINGS-EEG2** | EEG (10 subj, deep, high-SNR) | OSF `3eayd` (preproc) / `3jk45` (raw) | `eeg/things_eeg2.py:download()` (osfclient) | **fetching — the spine** |
| **THINGS-EEG1** | EEG (50 subj, low-SNR) | OpenNeuro `ds003825` | `openneuro-py download` | cross-dataset probe (caveated) |
| **THINGS-MEG** | MEG (4 subj) | OpenNeuro `ds004212` | `openneuro-py download` | upper-bound / cross-modal target |
| **NOD-EEG** | EEG (19 subj) | OpenNeuro `ds005811` | `openneuro-py download` | image-level cross-modality extension |
| **NOD-MEG / -fMRI** | MEG / fMRI | OpenNeuro `ds005810` / `ds004496` | `openneuro-py download` | same 57k ImageNet files as NOD-EEG |
| **Alljoined-1.6M** | EEG | HuggingFace | `huggingface-hub` | affordable-BCI, 1.6M trials |
| _object/face/food EEG_ | EEG (category-level) | OpenNeuro (dozen — see research) | `openneuro-py download` | more subjects for LOSO only |
| ⚠️ **Spampinato-2017** | EEG | — | — | **block-design leakage trap — do not use** (Li 2020) |

### Other repos, for reference
- **MOABB** — BCI *control* paradigms only (MI / P300 / SSVEP). Auto-downloads via `configure_moabb_download()`.
  **No image/semantic sets** — don't look here for perception. Extra MI sets (PhysioNetMI, Schirrmeister2017)
  are one-file adapters over `MoabbMIAdapter` when a 2nd-dataset generalization check is wanted.
- **OpenNeuro** — primary BIDS hub, GraphQL-searchable (`datasets(modality:"EEG", …)`), `openneuro-py download`.
- **OSF** — THINGS-EEG2, ZuCo. `osf -p <id> clone <dir>`.
- **Zenodo** — scattered (MSS, EEG-ImageNet-2024); `zenodo_get`. Not a primary hub.

Fetch tooling is the optional `data` extra: `uv sync --extra data` (openneuro-py, osfclient) — not needed at runtime.
