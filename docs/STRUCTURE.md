# mindscape — code structure (ported from the siblings)

> The architecture is **not invented here** — it's lifted from the two siblings, which already
> proved the shape. **cardiac-seg (systole)** is the mature reference (2-layer engine/science split,
> split-as-criteria, dataset-adapter registry, calibration-under-shift); **synthscape (mirage)** is
> the simpler single-package version. mindscape takes cardiac-seg's mature shape and retargets it
> from *cardiac MRI segmentation* to *non-invasive neural decoding*.
>
> Status: design doc, no code yet. Locks the structure before the first commit (Stage 0).

---

## The one idea everything hangs off

**The eval harness is the product, not the model.** Every decoder (handcrafted FBCSP, EEGNet,
a transformer) is a `(fit_fn, score_fn)` pair fed through ONE harness. The contribution is the
**honest evaluation regime** — within-subject vs cross-subject vs cross-session — and the
**calibration under that shift**, not any decoder. Same stance as both siblings; the decoder is
commodity, say so.

---

## Two layers (cardiac-seg's maturity move)

```
core/            the reusable engine — dataset-agnostic, decoder-agnostic
<domain>/        the science/contribution layer (calibration, OOD eval, diagnostics, model cards)
baselines/       the quarantine ceiling (the standard reported method), isolated
```

Domain package name TBD (sibling convention: `surfscan`, `cardioseg` → candidates `neuroscan` /
`neurodecode`). Used as the import root below as **`neuroscan/`** — rename freely.

---

## The port table (cardiac-seg → mindscape)

| cardiac-seg | mindscape | what changes |
|---|---|---|
| `core/config.py` (paths.yaml → raw/processed, `CARDIAC_DATA` env) | `core/config.py` (`MINDSCAPE_DATA`) | raw/ = MOABB cache + BIDS downloads; processed/ = **epoched** cache |
| `core/data/store.py` consolidated meta frame (polars) | `core/data/store.py` | one frame, columns `(dataset, subject, session, label, path)` |
| `core/data/splits.py` `make_split(meta, test_datasets, test_vendors, …)` | `core/data/splits.py` `make_split(meta, test_subjects, test_sessions, …)` | **vendor→subject.** Cross-subject = leave-one-subject-out via `test_subjects`. The honest regime IS a criteria filter. |
| `core/data/mri/registry.py` + `base.py DatasetAdapter` Protocol | `core/data/eeg/registry.py` + `base.py DatasetAdapter` | "add a dataset = one file + one line" |
| `core/data/mri/{acdc,mnm2,mnms1}.py` | `core/data/eeg/{bnci2014_001,physionet_mi,things_eeg2}.py` | each remaps to canonical schema (channels montage, label set, sfreq) |
| `core/model.py`, `core/inference.py` | `core/model.py`, `core/inference.py` | Braindecode model wrappers |
| `core/export_onnx.py` | `core/export_onnx.py` | **Stage 2 deploy is first-class engine, not bolted on** |
| `core/preprocessing/` (n4, preprocess) | `core/preprocessing/` (filter, epoch, artifact) | MNE: bandpass, epoch, EOG/line-noise reject |
| `core/hparams.py`, `core/types.py` | same | — |
| `cardioseg/evaluation/calibrate.py` (temp-scaling, ECE across test axes) | `neuroscan/evaluation/calibrate.py` | **verbatim insight**: calibrate within-subject, show ECE fails cross-subject |
| `cardioseg/evaluation/{uncertainty,distribution,validate,results,modelcard,ensemble}.py` | `neuroscan/evaluation/…` | accuracy, κ, ECE/Brier, cross-subject gap, per-subject diagnostics, model cards |
| `cardioseg/analysis/{eda,viz}.py` | `neuroscan/analysis/…` | signal EDA (spectra, channels, epochs) |
| `cardioseg/training/{train,losses,augment,dataset}.py` | `neuroscan/training/…` | EEG augmentations |
| `cardioseg/tracking.py` (MLflow) | `neuroscan/tracking.py` | unchanged |
| `baselines/nnunet/` | `baselines/fbcsp/` (or MOABB-reported) | the standard ceiling to quarantine against |
| `reference.yaml` + `core/reference.py` | same | the published ceiling numbers as config |
| `cardioview/` (browser viewer, in-browser ONNX) | `decode-viewer/` | decoder output on held-out signal, ONNX in-browser |
| `mri-sim/` (acquisition visualizer) | `signal-viz/` | the recording the model consumes (channels, spectra, epochs) |
| `tests/unit/` + `tests/integration/` | same | unit (equivalence-class per module) + integration (pipeline chain) |

---

## The split-as-criteria pattern — why it's the whole honesty story

cardiac-seg: *"a split isn't a named thing — it's the data cloud filtered on criteria … the criteria
live on the config, so a run self-documents what it held out."* `make_split` holds out whole datasets
or whole vendors as test; train/val = the rest.

mindscape's honest number is exactly this, with the axis renamed:

```python
# within-subject ceiling: random val carve, no held-out subject
make_split(meta, test_subjects=(), val_frac=0.2)

# cross-subject (the headline OOD gap): leave-one-subject-out
make_split(meta, test_subjects=["A03"])

# cross-session drift: hold out a session
make_split(meta, test_sessions=["session_2"])
```

Same `(fit_fn, score_fn)` contract; the **regime is just which criteria you filter on**, and the run's
config.json records it. No new harness per regime — the maturity is that the regime is *data*, not code.

---

## What carries 1:1 vs what's genuinely new

**Carries wholesale (don't reinvent):** paths.yaml one-root + data-out-of-repo, the meta-frame +
split-as-criteria, the DatasetAdapter registry, the calibration/ECE-under-shift module, MLflow tracking,
auto-ONNX export + parity gating, model cards, the unit+integration test layout, the README skeleton.

**Genuinely new (the ramp effort goes here):** the EEG/MEG preprocessing (MNE: filter/epoch/artifact),
the Braindecode model wrappers, the canonical neural schema (montage + label set), and the
neuroscience/decoding theory in `learning/`. Everything else is a retarget.

---

## Build order against this structure

1. **Stage 0** — `core/config.py` + `core/data/eeg/bnci2014_001.py` (BCI IV-2a via MOABB) +
   `store.py` + `splits.py` + `neuroscan/evaluation/` (accuracy + ECE + per-subject) +
   `baselines/fbcsp/` + one `experiments/run_eegnet.py`. **First commit = harness running on 2a.**
2. **Stage 0.5** — add `core/data/eeg/physionet_mi.py` (one file + one registry line) → prove the
   harness generalizes to a dataset it wasn't written for.
3. **Stage 1** — `core/data/eeg/things_eeg2.py` (BIDS adapter, same schema) → semantic decoding,
   same evaluation layer untouched.
4. **Stage 2** — `core/export_onnx.py` + quantization/distillation + `decode-viewer/` → the efficient
   on-device decoder, parity-gated.

See [`research/deep_dives/2026-06-30_tasks_datasets_landscape.md`](../research/deep_dives/2026-06-30_tasks_datasets_landscape.md)
for the task/dataset/benchmark landscape this structure serves.
</content>
