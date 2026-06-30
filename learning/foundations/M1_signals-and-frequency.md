# M1 · Signals & frequency (DSP) — the bedrock tool

Working fluency, tied to where it bites in EEG. Reference spine: **Steven Smith, *The Scientist and
Engineer's Guide to DSP*** (free, dspguide.com); **Mike X Cohen** (EEG/time-frequency); **3Blue1Brown**
(*But what is the Fourier Transform?*, intuition).

## The core ideas
- **Two domains.** A signal is a quantity over time (EEG = voltage(t), sampled at 250 Hz → 1125 numbers /
  4.5 s trial). The **Fourier transform** re-expresses it as a sum of sine waves — the **frequency domain**:
  *which rhythms compose it, and with what magnitude + phase.* For N real samples, N/2+1 independent
  frequency bins (conjugate symmetry), spaced Fs/N apart.
- **Sampling / Nyquist.** Faithful representation up to **Fs/2 = 125 Hz**. Components above Nyquist, if not
  filtered before sampling, **alias** — fold back to phantom lower frequencies (140 Hz → 110 Hz).
- **Band power.** Power = |FFT|²; band power = Σ|FFT(f)|² over the band's bins. **Band power ≈ variance**
  *after band-pass filtering* (Parseval: same energy in both domains; variance = average power; filter to
  one band → all remaining variance is that band's power). **This is the fact CSP runs on** — it maximizes
  a variance *ratio*, which post-filter is a band-power (ERD) ratio.
- **Filtering.** Band-pass EEG ~4–40 Hz: below 4 = slow drift (DC, sweat, movement) that dominates scale;
  above 40 = EMG / line noise / non-motor gamma, often bigger than the brain signal. Aggressive (too narrow
  / too sharp) filters kill real signal + cause ringing / phase distortion / edge artifacts.
- **Time-frequency tradeoff.** One FFT over the trial says mu power dropped, not *when*. Window the signal to
  localize in time → the **Gabor/uncertainty limit**: Δt·Δf ≥ const (sharp time ⇒ blurry frequency, and vv).
- **Why 250 Hz suffices** (vs 44.1 kHz audio): scalp EEG measures slow **summed population** postsynaptic
  potentials (smeared by volume conduction), content <~100 Hz — not fast (~1 ms) action potentials, which
  EEG doesn't capture. So Nyquist needs only ~200 Hz.
- **The arousal axis:** low/synchronized = a region idling; high/desynchronized = a region working. ERD *is*
  this: idle motor cortex synchronizes (mu grows); engaged (imagined movement) desynchronizes (mu drops).

---

## Quiz log

### 2026-06-30 — quiz M1 (self-assessed with teacher)
**Score ~4.5/6 — fluent in DSP; gaps: FFT bin count/symmetry, filter specifics, band-power≈variance.**

1. *FFT output / count* — ✓ concept (magnitude + phase, decomposition); missed N/2+1 independent bins
   (conjugate symmetry), Fs/N spacing.
2. *Nyquist / aliasing* — ✓ correct (125 Hz, aliasing; 140→110 fold-back is the precise version).
3. *Band power* — ✓ partial: power = magnitude **squared** (had summed magnitude); **band power≈variance**
   was the flagged gap → taught (Parseval + variance=power + filter-to-band).
4. *Filtering* — ✓ partial: knew the cuts (4 / 40), not what's at each end (drift / EMG+line) or the
   aggressive-filter cost (ringing/phase/edge).
5. *Time-frequency tradeoff* — ✓ correct, stated the uncertainty principle in own words.
6. *250 Hz vs 44.1 kHz* — ✓ good instinct ("clouds of neurons, not individuals" = the real reason); slip:
   action potentials are fast (~1 kHz), EEG just doesn't measure them.

**Follow-up Qs (understood after):** variance = the band's oscillation strength (the classification feature);
CSP = Common Spatial Patterns (learned spatial filters maximizing between-class power difference); the
low/high axis is really synchronization(idle)↔desynchronization(active) = the ERD mechanism.
