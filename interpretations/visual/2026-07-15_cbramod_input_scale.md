# CBraMod input scale: pretraining-faithful /100 vs z-score — the frozen probe favours z-score (bd 7mi4)

**Date:** 2026-07-15 · **Task:** Stage-3 frozen-CBraMod head search (train[1-4] → test5, S=1/200 Hz)

## Question

The jwzl audit found CBraMod's pretraining scaled input by `/100` (microvolts/100 — `pretrain_trainer.py:64`
does `x/100`), **not** the per-channel z-score our pipeline applied. The deep-dive (and the foundation
docstring) claimed "z-score", from the paper abstract; the *code* is ground truth. Hypothesis (pfad): feeding
CBraMod the amplitude-preserving scale it was pretrained on — instead of a z-score that flattens per-channel
amplitude — should lift the frozen features off chance and reframe "frozen-probe-loses" as partly an
input-mismatch artifact.

## Method

Via the new `core.normalization` chain (bd 4aoz): CBraMod fed `Scale(1e4)` (our volts ×1e6 → µV, /100 = the
pretraining range; derived + amplitude-verified, p99 ≈ 0.83) vs the per-channel `ZScore`. Same frozen
head-search protocol, same head zoo, `--normalize scale` vs the documented z-score baseline (identical
protocol, multi-seed).

## Result — REFUTED (at the frozen probe)

| head       | z-score (documented) | **Scale /100** |   Δ    |
|------------|:--------------------:|:--------------:|:------:|
| mean_lin   |        0.60          |     0.58       | ≈ chance |
| flat_mlp   |        1.21          |     1.27       | ≈      |
| pos_attn   |        1.59          |     1.23       | **−0.36** |
| topo_cnn   |        1.75          |     1.21       | **−0.54** |
| gcn        |        1.07          |     0.99       | ≈      |

Single-trial top1 %. Chance 0.5%.

## Interpretation

The pretraining-faithful scale does **not** help the frozen CBraMod probe — it **specifically degrades the
geometry heads** (topo 1.75→1.21, pos_attn 1.59→1.23), the ones that read scalp spatial structure. The
geometry-blind `flat` head is unaffected (1.21↔1.27), and `mean_lin` stays at chance either way.

Mechanism: the z-score equalizes per-channel variance, so the frozen features expose the spatial structure the
geometry heads exploit; the amplitude-preserving scale lets high-variance channels dominate the frozen
representation, washing out the relative spatial signal. Under z-score, geometry beats `flat` (1.75 > 1.21);
under scale, geometry collapses to `flat` (1.21 ≈ 1.27) — spatial information is gone.

So the "z-score is wrong for CBraMod" premise is **refuted for the frozen probe**, and the evidenced default
reverts to z-score (also what every existing CBraMod fine-tune/LoRA number used).

## Caveat — the frozen probe is not the fine-tune

A physically-reasonable input (the pretraining scale) regressing on a *finite* test means the test lacks the
axis, not that the input is wrong: the frozen probe **cannot adapt the backbone** to exploit correct-scale
input — only its conv filters "know" the /100 scale, and they are frozen, so the head is left compensating for
whatever the frozen features look like (z-score gives it more usable ones). Whether the amplitude scale helps
**fine-tuning** — where the backbone adapts — is the open question, filed as a follow-up. `Scale` stays a named
override (`--normalize scale`) for that test.

## What stands

The `core.normalization` chain (bd 4aoz) is the real deliverable and unaffected by the verdict: NICE gets MVNN
(the b40j win), CBraMod/EEGPT get z-score, all behind one composable `Normalizer` interface; the `/100` scale
is a first-class link kept for the fine-tune test. The frozen-probe finding is an honest negative for pfad's
premise, not a failure of the chain.
