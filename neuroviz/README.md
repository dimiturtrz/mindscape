# neuroviz

The motor-imagery EEG viewer — the field-standard 2D views of the signal the decoder reads
(an in-browser viewer of the data the model consumes). 2D topographic maps, not a 3D head: EEG scalp data is
inherently 2D-surface, and topomaps are what the field actually uses.

![neuroviz demo — fNIRS n-back workload: animated HbO hemodynamic response topomap + HbO/HbR waveforms, with the decoder's ground-truth-vs-prediction readout](docs/media/demo.webp)

**Shows, for a BCI IV-2a subject:**
- **Band-power topomaps** (mu 8–12 Hz, beta 13–30 Hz) per class — the motor-imagery signature: imagining
  one hand desynchronizes the *contralateral* motor cortex (the hot spot flips C3↔C4 between left/right hand).
- **CSP spatial patterns** — what the CSP baseline learns (should localize over C3/C4 → physiologically
  real features, not artifacts).
- **Riemannian discriminant** — per-class channel weights of the tangent-space classifier
  (the strongest baseline, 0.706 within-subject). Here the *covariance* is the feature, so the map shows
  which channels' power drives each class — a different lens on the same signal than CSP's spatial filters.
- **Waveforms** — all channels for an example trial, each colored by its contribution to the selected
  map — so the channels that drive the current view light up, no hardcoded highlight.

**fNIRS view (a top-level EEG/fNIRS toggle):** the same viewer renders the hemodynamic modality — animated
**HbO/HbR topomaps** per n-back workload class (watch the response build + peak ~5–8 s over prefrontal
cortex), the optode montage, and the **LDA discriminant** (per-channel amplitude weight the decoder reads).
Same JSON schema, so one web app shows both signals side by side.

**EEG-workload view (same task as fNIRS):** the workload task's EEG side — per-class **θ/α/β band-power
topomaps** (frontal theta rises, parietal alpha suppresses with n-back load), CSP + Riemann patterns, and the
honest LOSO decoder result. Reads channel names + epochs from the **processed store** (one format), maps them
to a standard-10-05 montage; same JSON schema as the motor-imagery EEG view.

**Fusion view (a third toggle):** not a signal animation but the *result* — a per-block **complementarity
map**. Every held-out n-back block (rows = subjects, subject-wise 5-fold GroupKFold) is a cell colored by
which modality decoded it: both-right / EEG-only / fNIRS-only / both-wrong. Blue + orange scattered and
balanced (error corr φ≈0) = the two weak modalities fail on *different* blocks, so an oracle that picked the
right one per block hits **0.69** vs best-single **0.47** — yet naive late fusion sits at the best single,
because confidence doesn't track correctness. The bars show EEG · fNIRS · late · oracle against chance. It
visualizes the honest fusion finding: complementarity is real, no output-space combiner cashes it.

## Run
```bash
# 1) export view data (writes neuroviz/web/data/, gitignored)
uv run python -m neuroviz.export --subject 1           # motor imagery · EEG
uv run python -m neuroviz.export_eeg_workload --subject 1  # workload · EEG (θ/α/β band-power)
uv run python -m neuroviz.export_fnirs --subject 1     # workload · fNIRS
uv run python -m neuroviz.export_fusion                # workload · fusion complementarity map

# 2) serve + open — a two-tier task > modality toggle (Motor imagery | Mental workload -> EEG/fNIRS/Fusion)
python -m http.server 8000 -d neuroviz/web      # then open http://localhost:8000  (deep-link: /#eeg_workload)
```

The web app is dependency-free (vanilla JS + canvas, no build step); the topomap is an inverse-distance
interpolation of the 22 electrodes over the scalp circle.

_The viewer makes the neuroscience visible; the contribution remains the honest cross-subject evaluation,
not the picture._
