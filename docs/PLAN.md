# mindscape — build plan

> **Personal learning project — non-invasive neural decoding: read intent/meaning from brain signals, evaluate honestly, deploy efficiently on the edge.**

Working name `mindscape` (the terrain of the mind). Folder/repo may differ; rename freely.

mindscape applies a *spine-first, honest-eval* pattern — the same one carried from prior ML projects of mine (honest cross-distribution evaluation as the contribution, not a leaderboard number) — to neural signal. The structural and honesty conventions below are lifted from that prior work deliberately.

> **Status: initialization (2026-06-30).** Thesis + shape set; the spine (first task, eval harness, dataset) lands next. No results yet — when there are, they're reported as the *measured contrast + where it fails*, not a leaderboard number.

---

## The center (what this actually is)
**Not** a "BCI stack." It's **honest, efficient non-invasive neural decoding** — the through-line carried from prior acoustic-detection work, retargeted to brain signal:

**signal → preprocessing → decode → honest evaluation (calibration / out-of-distribution) → efficient on-device deployment.**

- **Decoding + evaluation rigor = the leverage** (signature strength). The recurring contribution across the prior projects: separate genuine generalization from overfitting on real, uncontrolled signal.
- **Efficient / on-device decoding = the second through-line** (carried edge-inference: quantization / distillation) — the deployable angle the field underweights.
- **Neuroscience + the decoding methods = the new domain** (the genuinely new skill; treat as a ramp, the same move prior projects made).

## What we're building (one line)
A non-invasive neural decoder, the honest measurement of how it holds up on signal it never trained on, and an efficient edge-deployable version of it.

---

## The spine: one honest, bounded capability
The pattern: one through-line — *take a signal → produce a measured result → show where it fails* — with the **honest-hard-part as the contribution**, not a leaderboard number. mindscape:

**decode a non-invasive neural signal → measure it honestly (calibration, out-of-distribution) → show where it fails → run it efficiently on the edge.**

### Headline numbers — decided up front
Concrete numbers tied to a bar, committed early:
- **Decoding accuracy** on a public benchmark vs the reported/reproduced ceiling (the standard result is the baseline to quarantine against — here the published ATCNet / FBCSP ceiling).
- **Calibration / robustness under shift** (ECE / reliability across subjects/sessions) — the rare-class + calibration signature, carried.
- **Efficiency** — latency / model size / (where measurable) power for the on-device decoder, vs the full-precision model. The deployable number.
Per-condition diagnostics stratify the failure (by subject, session, signal quality) — the axis that hides the mean's lie.

---

## The ramp (real spine first, then breadth)
1. **Stage 1 — EEG decode + honest eval + cross-subject transfer. ✅ DONE.** Standard motor-imagery benchmark (BCI IV-2a) through a *verified* harness (accuracy + κ + calibration + per-subject/session diagnostics); reproduce known methods (CSP+LDA, Riemann, the nets); *measure* the out-of-distribution gap honestly (within 0.706 → cross 0.357) — then **close it with the field's fix** (Riemannian re-centering → 0.496). The cross-subject gap + its principled, unsupervised, deployable fix is the headline. This was the public-flip trigger.
2. **Stage 2 — Second modality: fNIRS (parallel track).** Add the hemodynamic axis on **Shin 2017** (mental arithmetic — fNIRS's home task, prefrontal). Same honest harness + Riemann/recentering spine (validated on fNIRS covariances, Näher 2025). The modality-specific failure mode — **systemic-physiology leakage** (scalp/heartbeat/Mayer waves correlated with task) — is the new rigor angle. Parallel to Stage 1, not strictly after it. Sequence: fNIRS-alone → EEG-alone on the same subjects → sets up Stage 3.
3. **Stage 3 — Fusion + harder tasks.** Hybrid EEG+fNIRS **fusion** (the complementarity grid: EEG strong on motor imagery, fNIRS strong on arithmetic — each covers the other's blind spot, fusion beats either alone); and the harder **semantic / communication-decoding** target (THINGS-EEG2). The payoff, built on two *proven* unimodal pipelines — not attempted before the baselines exist.
- **Cross-cutting (optional) — efficient deploy.** ONNX export + parity gate + quantization, applied per method *where it actually helps* (measured: our nets are already edge-tiny, ~26 KB / sub-ms — INT8 *adds* overhead here). Method-dependent optimization, **not** a mandatory stage.

## Comparisons (the honest reporting IS the contribution)
Against the published ceiling (reproduced through the same harness), report honestly:
1. Reproduced ceiling (standard method on the standard split).
2. Cross-subject / cross-session — the out-of-distribution gap **and the transfer fix that closes it**.
3. The modality-specific way the number lies — EEG's cross-subject location shift, fNIRS's systemic-physiology leakage.
4. *(optional, when the method warrants edge deploy)* efficient/quantized vs full-precision — the deployment cost.

That + calibration + per-condition diagnostics is the defensible result — not a leaderboard number.

---

## Structure — three pieces, general → specific
Memorability comes from a visualizer that makes the work demonstrable. mindscape's three pieces, general → specific. **A piece is added only when it's real — no empty speculative folders.**

| piece | role |
|---|---|
| **signal viz** (`neuroviz/`) — the neural recording the model consumes (topomaps, spectra, epochs) | *understand the data the model consumes* |
| **decode viewer** — the decoder's output on held-out signal, in-browser (ONNX) | *see the model work* |
| **pipeline** (`core/` + `neuroscan/`) — data → preprocess → decode → eval harness | *the science layer — the eval harness is the contribution* |

### learning/ track — how I ramp into the field
Mirror the `learning/<date>_<topic>.md` + glossary + on-demand self-quizzes, and the circuit: **research (grounds it → `research/`) → theory writeup (`learning/`) → quiz → sharpen the plan.** Neuroscience + neural-decoding theory is where the ramp effort goes. Build log = git history; theory artifacts = `learning/`.

### Reuse the engine conventions + carry the acoustic-work discipline (don't reinvent)
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
2. **Differentiator = the honest eval + cross-modality rigor, not the model.** The decoder is commodity; say so. (Efficient on-device deploy is an *optional* bonus, method-dependent — not the headline.)
3. **Cite prior art** (the benchmark papers, the decoding methods, the libs) — informed, not naive.
4. **Verify every ⚠️** against a primary source before it lands in the README.
5. Thesis framing: *"honest, efficient non-invasive neural decoding,"* not *"I built a BCI."*
6. **Honest-limits section, measured not assumed** (the prior projects' standard): claim edge deploy (real), never clinical/production-grade; neuroscience is a ramp — say so; state the gaps as numbers.
7. Build private → **flip public at Stage 0** (warm-up + eval harness presentable, prior art cited); steady real commits after.

---

## Where we are
**Stage 1 done** — EEG motor-imagery decode + honest eval + cross-subject transfer (recentering closes 0.357→0.496); public, tested. **Stage 2 next** — fNIRS second modality (Shin 2017, mental arithmetic), parallel. **Stage 3** — hybrid fusion + harder semantic decoding, on top of the proven unimodal pipelines. Efficient edge deploy = optional, applied per method where it helps.
