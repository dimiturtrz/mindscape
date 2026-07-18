# Contrastive trainer seam contract (bd 3cy)

**Status:** documented, **not extracted.** There is exactly one contrastive/retrieval trainer today
(`TrainNice`). Abstracting a shared trainer at n=1 guesses the seam — this doc records *where* the seam is and
*what* triggers the refactor, so that when a real second method lands the extraction is mechanical and
parity-checkable rather than speculative. This is the honest middle: name the contract now, build it later.

## The shared spine (what any contrastive retrieval method has in common)

`TrainNice.train_encoder` (neuroscan/tasks/visual/train_nice.py) is the reference loop:

```
data (eeg, targets, concept, subject)
  -> leak-free val split (hold out TRAIN concepts, not test)         _val_split
  -> per-epoch batch construction                                    _epoch_steps / Sampling
  -> encoder forward -> embedding                                    EncoderRegistry
  -> contrastive loss vs the target embedding                        Nice.clip_infonce
  -> optimizer step (+ optional aux terms: geo prior, adversary)     _run_epoch / _StepCtx
  -> early-stop on val retrieval top-1                               evaluate
  -> best-val checkpoint + retrieval eval (top-k, rank, continuous)  evaluate / Retrieval
```

The spine is stable — every step above is method-agnostic *except* the four hook points below.

## The hook points (what a 2nd method would vary)

A shared trainer would expose exactly these — derived from the loop, not invented:

1. **Loss** — `clip_infonce` (symmetric InfoNCE vs CLIP target). A 2nd method varies *this*: a different
   contrastive objective (SigLIP/pairwise-sigmoid, multi-positive, a generative/diffusion reconstruction
   loss). Signature: `(embedding, target, *aux) -> scalar`.
2. **Batch contract** — `_epoch_steps` yields `(eeg, target, subject)` batches; the target is a *dense CLIP
   vector per trial*. A method whose supervision is not a per-trial dense target (e.g. class-index positives,
   a text tower, hard-negative triplets) changes the batch record shape.
3. **Eval head** — `evaluate` scores retrieval against a per-concept prototype bank by cosine. A method with a
   different retrieval geometry (a learned metric, a cross-encoder re-ranker) swaps this.
4. **Encoder I/O** — already abstracted (`EncoderRegistry` + `EncoderSpec.embed_dim` auto-sizing). *Not* a
   seam that needs work — swapping NICE/CBraMod/EEGPT/LoRA already rides the current loop. **This is why a new
   backbone does NOT trigger 3cy.**

## The trigger (do not extract before this)

Extract the shared trainer **only** when a method appears that varies hook 1, 2, or **3** — a genuinely
different *shape*, not another backbone through hooks that already exist. Concretely:

- ✅ triggers: a non-InfoNCE objective; a text-tower / dual-encoder retrieval; a generative decoder trained
  jointly; supervision that isn't a per-trial dense CLIP target.
- ❌ does not trigger: another braindecode/foundation backbone; LoRA vs full-ft; a new CLIP target dim; a new
  sampler (those are already config on the current loop, see bd 7tl).

## The refactor contract (when triggered)

1. Factor the spine into a shared trainer taking the four hooks as injected strategy objects (one home; the
   op-namespace pattern the rest of the tree uses).
2. Re-express `TrainNice` as a config over the shared trainer. **Acceptance = byte-identical:** same seed →
   identical cross-subject single-trial top-1 / MRR / median-rank to the pre-refactor `train()`. A retrieval
   regression means the seam is wrong.
3. Land method #2 *through* the shared trainer, not bolted alongside. If it doesn't host cleanly, the hook
   set is wrong — fix the seam, don't special-case the caller.

**Non-goal:** a common trainer with a single caller. That is the guessed interface this bead exists to avoid.
Until the trigger fires, `TrainNice` stays the whole story and this doc is the standing plan.
