# neuroviz

In-browser viewer of the signal the decoder reads — field-standard 2D topographic maps (not a 3D head: EEG
scalp data is inherently 2D-surface, and topomaps are what the field uses). Motor-imagery EEG, mental-workload
EEG/fNIRS, and the **EEG+fNIRS fusion brain-camera**.

![neuroviz demo — EEG+fNIRS fusion brain-camera (Shin n-back): the fused surface-video — raw EEG band-power + the fNIRS CBSI neural map (left), the locality-gated joint firing pattern (right), with the hemodynamic lag derived per subject](docs/media/demo_fusion.webp)

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

![neuroviz — fNIRS n-back workload: animated HbO hemodynamic response topomap + HbO/HbR waveforms, with the decoder's ground-truth-vs-prediction readout](docs/media/demo_fnirs.webp)

**EEG-workload view (same task as fNIRS):** the workload task's EEG side — per-class **θ/α/β band-power
topomaps** (frontal theta rises, parietal alpha suppresses with n-back load), CSP + Riemann patterns, and the
honest LOSO decoder result. Reads channel names + epochs from the **processed store** (one format), maps them
to a standard-10-05 montage; same JSON schema as the motor-imagery EEG view.

**Fusion view (a third toggle) — the EEG+fNIRS *brain-camera*:** the fused surface-video. **LEFT** the two raw
inputs — EEG band-power (fast electrical *strength*) + the fNIRS **CBSI** neural map (slow blood, both
chromophores, systemic-rejected — *origin + spread*). **RIGHT** the **joint firing pattern** = EEG-strength ×
fNIRS-extent, gated to where a co-located EEG↔fNIRS pair exists. The hemodynamic lag is **derived per subject**
(fit from the EEG↔blood coupling, not a fixed 5 s), and the blood is read *forward* to align with the EEG event
that drove it. Honest framing: this is a *visualization*, not a decode win — for graded workload, fusion sits at
a physiological + redundancy ceiling (fNIRS adds ~nothing decodable over EEG; see the top-level README). The
camera shows the physics; the number lives in the extraction path, not the heatmap.

## Run
```bash
# 1) export view data (writes neuroviz/web/data/, gitignored)
uv run python -m neuroviz.export --subject 1           # motor imagery · EEG
uv run python -m neuroviz.export_eeg_workload --subject 1  # workload · EEG (θ/α/β band-power)
uv run python -m neuroviz.export_fnirs --subject 1     # workload · fNIRS
uv run python -m neuroviz.fusion.export --subject 1    # workload · fusion brain-camera (surface-video)

# 2) serve + open — a two-tier task > modality toggle (Motor imagery | Mental workload -> EEG/fNIRS/Fusion)
python -m http.server 8000 -d neuroviz/web      # then open http://localhost:8000  (deep-link: /#eeg_workload)
```

The web app is dependency-free (vanilla JS + canvas, no build step); the topomap is an inverse-distance
interpolation of the 22 electrodes over the scalp circle.

_The viewer makes the neuroscience visible; the contribution remains the honest cross-subject evaluation,
not the picture._
