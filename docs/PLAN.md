# mindscape — build plan

> **Personal learning project — non-invasive neural decoding: read intent/meaning from brain signals, evaluate honestly, deploy efficiently on the edge.**

Working name `mindscape` (the terrain of the mind). Folder/repo may differ; rename freely.

Sibling projects — same build philosophy, different domain: **[systole](https://github.com/dimiturtrz/cardiac-seg)** (cardiac MRI → ejection fraction → honest cross-scanner eval) and **[mirage](https://github.com/dimiturtrz/synthscape)** (3D surface-defect anomaly → honest sim-to-real eval). mindscape is the same *spine-first, honest-eval* pattern applied to neural signal. The structural and honesty conventions below are lifted from them deliberately.

> **Status: initialization (2026-06-30).** Thesis + shape set; the spine (first task, eval harness, dataset) lands next. No results yet — when there are, they're reported as the *measured contrast + where it fails*, not a leaderboard number.

---

## The center (what this actually is)
**Not** a "BCI stack." It's **honest, efficient non-invasive neural decoding** — the through-line carried from prior acoustic-detection work, retargeted to brain signal:

**signal → preprocessing → decode → honest evaluation (calibration / out-of-distribution) → efficient on-device deployment.**

- **Decoding + evaluation rigor = the leverage** (signature strength). The recurring contribution across the siblings: separate genuine generalization from overfitting on real, uncontrolled signal.
- **Efficient / on-device decoding = the second through-line** (carried edge-inference: quantization / distillation) — the deployable angle the field underweights.
- **Neuroscience + the decoding methods = the new domain** (the genuinely new skill; treat as a ramp, the same move systole/mirage made).

## What we're building (one line)
A non-invasive neural decoder, the honest measurement of how it holds up on signal it never trained on, and an efficient edge-deployable version of it.

---

## The spine: one honest, bounded capability
systole's power is one through-line — *segment the heart → measure EF → show where it fails* — with the **honest-hard-part as the contribution**, not a leaderboard number. mirage mirrors it (synthetic defects → detect → quantify the sim-to-real gap). mindscape:

**decode a non-invasive neural signal → measure it honestly (calibration, out-of-distribution) → show where it fails → run it efficiently on the edge.**

### Headline numbers — decided up front (the siblings' lesson)
Concrete numbers tied to a bar, committed early:
- **Decoding accuracy** on a public benchmark vs the reported/reproduced ceiling (the standard result is the baseline to quarantine against — the role nnU-Net played in systole, the SOTA memory-bank played in mirage).
- **Calibration / robustness under shift** (ECE / reliability across subjects/sessions) — the rare-class + calibration signature, carried.
- **Efficiency** — latency / model size / (where measurable) power for the on-device decoder, vs the full-precision model. The deployable number.
Per-condition diagnostics stratify the failure (by subject, session, signal quality), the way systole stratified EF error by pathology/vendor.

---

## The ramp (real spine first, then the angle)
1. **Stage 0 — Warm-up, real data.** Stand up a standard non-invasive decoding task (e.g. motor-imagery on a public benchmark) through a *verified* eval harness (accuracy + calibration + per-subject/session diagnostics). Prove the spine before anything novel. **First commit = dataset + eval harness.** = systole's Gate 1 (presentable) = **the public-flip trigger.**
2. **Stage 1 — Toward communication / semantic decoding.** Move to the harder, more meaningful target — decoding intended communication from non-invasive signal. This is where the real-world / cross-subject / cross-session evaluation traps live (a decoder that scores well on its own recordings can mean little out of distribution). Reproduce ONE known method and *measure* it honestly — don't chase the leaderboard. Calibration + out-of-distribution gap = the headline honest number.
3. **Stage 2 — Efficient on-device decoder (the differentiator).** Quantization / distillation toward a real-time decoder on commodity edge hardware; ONNX → runtime, latency/size/(power) benchmark. Target minimal accuracy loss vs full precision. The edge-inference discipline carried from prior work, applied to neural decoding — what the field underweights.
4. **After the spine — expansion axes** (modality breadth · decoding depth). Optional, sequenced *after* the public flip — mapped so the ladder isn't lost, not pulled forward.

## Comparisons (the honest triad IS the contribution)
Like systole's nnU-Net baseline and mirage's SOTA memory-bank, report honestly:
1. Reproduced ceiling (standard method on the standard split).
2. Cross-subject / cross-session (the out-of-distribution gap).
3. Efficient/quantized version vs full-precision (the deployment cost).

That triad + calibration + per-condition diagnostics is the defensible result — not a leaderboard number.

---

## Structure — three pieces, general → specific (the siblings' shape)
The siblings' memorability is the visualizers that make them demonstrable (systole's `mri-sim` + `cardioview`; mirage's synth-gen viz + point-cloud viewer). mindscape clones the three-piece shape. **A piece is added only when it's real — no empty speculative folders.**

| systole / mirage | mindscape | role |
|---|---|---|
| `mri-sim` / synth-gen viz | **signal viz** — the neural recording the model consumes (channels, spectra, epochs) | *understand the data the model consumes* |
| `cardioview` / point-cloud viewer | **decode viewer** — the decoder's output on held-out signal, in-browser (ONNX) | *see the model work* |
| `cardioseg` / pipeline | **pipeline** — data → preprocess → decode → eval harness | *the science layer — the eval harness is the contribution* |

### learning/ track (the siblings' "how I ramp into the field")
Mirror the `learning/<date>_<topic>.md` + glossary + on-demand self-quizzes, and the circuit: **research (grounds it → `research/`) → theory writeup (`learning/`) → quiz → sharpen the plan.** Neuroscience + neural-decoding theory is where the ramp effort goes. Build log = git history; theory artifacts = `learning/`.

### Reuse from the siblings + the acoustic engine wholesale (don't reinvent)
- **`paths.yaml` one-root config** + data-out-of-repo + per-source adapters → a common schema.
- **Calibration + eval harness** — accuracy, Brier/ECE, KS-test domain-gap, distribution-shift detection — carried, retargeted to neural decoding.
- **Model cards + auto-ONNX-export at train-end** (per-run, parity-gated quantization).
- **Cross-implementation parity tests** — the train-vs-deploy preprocessing-skew discipline applies directly to ONNX/edge export parity.
- **README skeleton:** one-line → the honest question → three pieces → the number that matters first → stratified where-it-fails → honest limits → data → quickstart → tests → how-it's-built → license.
- **Test layout:** unit (equivalence-class) + integration (module-pairs).

---

## Tooling / stack landscape (⚠️ = verify before it lands in a result claim; a dedicated research pass is owed)
- **Processing:** MNE-Python (EEG/MEG) ⚠️; signal-processing (filtering, spectral, epoching) carries from acoustic.
- **Decoding libs:** Braindecode (PyTorch EEG decoding) ⚠️, MOABB (Mother of All BCI Benchmarks — standardized datasets + pipelines) ⚠️.
- **Models:** PyTorch — CNN/temporal/transformer decoders; the contribution is the harness + honest comparison, not any one model.
- **Datasets (public, outside the repo):** motor-imagery benchmarks (e.g. BCI Competition IV) for the warm-up ⚠️; toward communication/semantic decoding (e.g. Things-EEG/MEG, non-invasive speech-decoding sets such as MEG-MASC) ⚠️.
- **Deploy:** PyTorch → ONNX → edge runtime; quantization/distillation carried from prior edge work.
- **Eval:** custom harness — accuracy, calibration-under-shift, cross-subject/session gap, per-condition diagnostics ← **the contribution.**

## SOTA stance / honesty rules
1. **Don't chase decoding-accuracy SOTA.** Use the standard libs/benchmarks, reproduce a known method, **contribute the eval rigor + the efficient deployable** the field underweights.
2. **Differentiator = the eval + the efficient on-device decoder, not the model.** The decoder is commodity; say so.
3. **Cite prior art** (the benchmark papers, the decoding methods, the libs) — informed, not naive.
4. **Verify every ⚠️** against a primary source before it lands in the README.
5. Thesis framing: *"honest, efficient non-invasive neural decoding,"* not *"I built a BCI."*
6. **Honest-limits section, measured not assumed** (the siblings' standard): claim edge deploy (real), never clinical/production-grade; neuroscience is a ramp — say so; state the gaps as numbers.
7. Build private → **flip public at Stage 0** (warm-up + eval harness presentable, prior art cited); steady real commits after.

---

## Start here
**A standard non-invasive decoding task + the eval harness, real data.** First commit. Stage 1 (harder target) and Stage 2 (efficient deploy) build on top. That commit is the public-flip trigger.
