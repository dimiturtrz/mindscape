# 01 · Motor imagery & ERD — the signal we decode

Grounded in [`research/deep_dives/2026-06-30_motor-imagery-neuroscience.md`](../research/deep_dives/2026-06-30_motor-imagery-neuroscience.md)
(cited sources). The neuroscience slice mindscape stands on. Connects to the data the pipeline consumes
(`core/data/eeg/`) and the signal the decoder reads. Grounding: BCI IV-2a, 4-class, our CSP+LDA hits
**0.598 within-subject / 0.382 cross-subject**.

Lesson plan:
1. The task — what we're predicting, and why *imagined* movement
2. The motor system — where the signal comes from
3. Imagery vs execution — same engine, throttled
4. ERD — what the decoder actually grabs
5. Why it's subject-specific (the root of our cross-subject gap)

---

## Lesson 1 — the task

**One sentence:** from ~4 s of scalp EEG, predict which of 4 movements a person is *imagining*
(left hand, right hand, both feet, tongue). 4-class classification, chance 25%.

A **cue** (arrow) tells the subject which to imagine; they imagine it ~4 s without moving; 22 electrodes
record voltage at 250 Hz. One imagined movement = one **trial** = a `[22 channels × 1125 samples]` array.

**Why imagery, not real movement?** The goal is a **BCI for people who can't move** (ALS, spinal injury,
locked-in). Imagined movement drives motor cortex almost like real movement — so a decoder reading imagery
can move a cursor / prosthetic / speller *by thought alone*. Real muscles not required.

*In our pipeline:* `core/data/eeg/bnci2014_001.py` pulls exactly this — 9 subjects, 2 sessions, 288
trials/session, 4 balanced classes. One row of the meta cloud = one trial.

**Takeaway.** A balanced 4-class time-series classification problem, where the labels are *intentions*.

## Lesson 2 — the motor system

Movement is planned and issued by a strip of cortex along the central sulcus:
- **Primary motor cortex (M1)** — issues the movement command; somatotopically organized (the "homunculus":
  hand, foot, face each have their patch).
- **Premotor cortex + supplementary motor area (SMA)** — planning, sequencing, preparation.
- Plus parietal cortex (where/how), cerebellum + basal ganglia (timing/coordination).

Key fact for us: motor cortex is **contralateral** — the *left* hemisphere controls the *right* body, and
vice-versa. So right-hand activity shows up over **left** motor cortex (electrode **C3**), left-hand over
**right** (**C4**), feet near the midline (**Cz**). That spatial layout is *the* thing CSP and the topomaps
exploit.

**Takeaway.** Hand/foot/tongue map to different, partly-lateralized motor-cortex locations — the signal has
a *spatial* signature, not just a temporal one.

## Lesson 3 — imagery vs execution: same engine, throttled

Imagining a movement runs **much of the same machinery** as doing it (the "functional equivalence"
hypothesis): M1, premotor, SMA, parietal all engage. Evidence: **mental chronometry** — imagined movements
take about the same *time* as real ones.

The differences:
1. **M1 fires but weaker** in imagery than execution.
2. **The motor output is actively inhibited** downstream (corticospinal gating) — so you imagine without
   twitching.
3. **No sensory feedback** — real movement floods somatosensory cortex with proprioception; imagery doesn't.

Consequence for the signal: **imagery's signature is the same shape but fainter and noisier** than
execution's. That weaker-signal tax is a real chunk of why our accuracy is ~60%, not ~95% — if we decoded
*actual* finger movements the numbers would be far higher.

**Takeaway.** Same network, lower intensity, output unplugged — and that throttling is exactly the
difficulty we fight.

## Lesson 4 — ERD: what the decoder actually grabs

When a patch of sensorimotor cortex is **idle**, its neurons fire in lazy synchrony → a strong rhythmic
oscillation in the **mu (8–12 Hz)** and **beta (13–30 Hz)** bands (high spectral power). This is the
**sensorimotor rhythm (SMR)**.

When that patch goes to **work** (move or imagine moving), the neurons stop the unison and do differentiated
work → the rhythm breaks down → **8–30 Hz power drops**. That localized power drop is **event-related
desynchronization (ERD)**.

> Idle cortex = a crowd chanting in unison (loud, rhythmic). Start a task = everyone talks differently — the
> chant dies. The dying chant = ERD.

So a decoder distinguishing left vs right hand is, at bottom, asking: *did the mu/beta chant go quiet over
C3 (→ right hand) or over C4 (→ left hand)?* The complement, **ERS** (synchronization), is a power *rebound*
after the movement.

*In our pipeline:* `neuroviz` renders exactly this — mu/beta power topomaps that animate the ERD over the
trial; the hot spot flips C3↔C4 between left and right hand. The decoders quantify what neuroviz shows.

**Takeaway.** ERD = the suppression of mu/beta over the active (contralateral) motor cortex. *That power
drop, localized, is the decodable signal.*

## Lesson 5 — why it's subject-specific (our cross-subject gap, neurally)

The ERD signature **varies per person**: the exact frequency (one person's mu is 9 Hz, another's 11),
the precise scalp location (cortical folding differs), the strength (SNR), even whether beta or mu carries
more. Skull thickness and electrode placement add more between-subject variance.

So a decoder that learned subject A's exact spatial-spectral pattern is **mis-tuned for subject B**. That's
the *neural* reason our cross-subject accuracy collapses **0.598 → 0.382**: not a code bug, a real biological
fact the field underreports. It's also why per-subject calibration exists in real BCIs.

**Takeaway.** The signal is real but **idiosyncratic** — the generalization gap is biology, not just modeling.

---

## Quiz — 01 (answer cold, then check)
1. Why does the task use *imagined* rather than real movement? Who is it for?
2. Right-hand imagery desynchronizes which hemisphere, and which electrode is the key one?
3. Name two ways imagery differs from execution in the brain, and the consequence for the EEG signal.
4. Define ERD in one sentence. Which frequency bands, and does power go up or down?
5. Give the *neural* reason (not a coding reason) our cross-subject accuracy drops from 0.598 to 0.382.

<!-- quiz log appended below on demand -->
