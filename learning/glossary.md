# Glossary — one line each

Grounded in [`research/deep_dives/2026-06-30_motor-imagery-neuroscience.md`](../research/deep_dives/2026-06-30_motor-imagery-neuroscience.md)
and the lessons.

## Neuroscience / signal
- **Motor imagery (MI)** — imagining a movement without performing it; engages much of the motor network.
- **Motor execution (ME)** — actually performing the movement; stronger, cleaner motor-cortex signal than MI.
- **Functional equivalence** — the hypothesis that MI and ME share motor mechanisms (M1, PMC, SMA, etc.).
- **M1 (primary motor cortex)** — issues movement commands; engaged but weaker (and output-inhibited) in MI.
- **SMA (supplementary motor area)** — planning; during MI it *suppresses* M1 so no muscle contraction occurs.
- **Contralateral** — the hemisphere opposite the body part; right-hand → left cortex (C3), left-hand → C4.
- **C3 / Cz / C4** — electrodes over left-hand-area / midline (feet) / right-hand-area motor cortex.
- **Sensorimotor rhythm (SMR)** — idle oscillation over motor cortex in mu + beta bands.
- **mu** — 8–12 Hz sensorimotor rhythm ("central alpha"); its power drops during (imagined) movement.
- **beta** — 13–30 Hz sensorimotor rhythm; ERD during planning/execution, ERS rebound after.
- **ERD (event-related desynchronization)** — a localized DROP in mu/beta power during a sensorimotor event;
  the signal we decode. Reflects increased cortical excitability.
- **ERS (event-related synchronization)** — a power REBOUND (above baseline) after the event ends.
- **Mental chronometry** — imagined and real movements take ~equal time; evidence of shared planning.
- **Volume conduction** — the field spreads through tissue, so each electrode sees a blurred MIX of sources.
- **EEG epoch / trial** — one cue-locked window (here 22 ch × 1125 samples = 4.5 s @ 250 Hz).
- **Montage** — the set + scalp positions of the electrodes (here 22-channel 10-20).

## Methods / ML
- **CSP (Common Spatial Patterns)** — spatial filters from a generalized eigendecomposition of class
  covariances; maximizes the band-power (variance) ratio between two classes.
- **Spatial filter** — a weighted sum of channels that isolates one cortical source (undoes volume conduction).
- **LDA** — Linear Discriminant Analysis; a linear classifier on the CSP features.
- **EEGNet** — compact CNN (~3.7K params): temporal conv (frequency) + depthwise spatial conv (CSP-like).
- **ATCNet** — attention + temporal-convolutional net (~114K params) with internal sliding-window augmentation.
- **Band power** — energy in a frequency band; for band-limited signals, ≈ variance.
- **kappa (Cohen's κ)** — chance-corrected agreement; κ=0 = guessing. Reported because 4-class accuracy flatters.
- **ECE (Expected Calibration Error)** — gap between predicted confidence and actual accuracy.
- **Temperature scaling** — divide logits by a learned scalar T to recalibrate confidence post-hoc.

## Evaluation regimes
- **within-subject** — train + test the same person (the ceiling; flatters the decoder).
- **cross-subject (LOSO)** — leave-one-subject-out; the honest "works on a new person" number.
- **cross-session** — same person, different day; tests drift.
- **generalization gap** — the accuracy drop from within → cross-subject (ours: 0.598 → 0.382).

## Stack
- **BCI** — brain–computer interface; here, decode intent from EEG to drive a device.
- **MOABB** — Mother of All BCI Benchmarks; standardized datasets + pipelines (our data source).
- **MNE-Python** — the EEG/MEG processing library.
- **Braindecode** — the PyTorch EEG-decoding library (EEGNet/ATCNet live here).
- **BCI IV-2a** — the 4-class motor-imagery benchmark we use (9 subjects).
