# EEG→image reconstruction: the recon objective + unCLIP decoder (ooi / 71n)

**Date:** 2026-07-17 · **Task:** Stage-3 perception (EEG→image), cross-subject · **Beads:** ooi (recon
objective), 71n (decoder) · **Endpoint:** the retrieval number was always the proxy — this is the payoff:
turn a predicted CLIP embedding into an image, and report the honest cross-subject gap.

## What was built

- **Recon objective (ooi):** a CLIP target zoo (`clip_targets._TARGETS`) — the EEG encoder auto-sizes to the
  target dim (`EncoderSpec(embed_dim=targets.shape[1])`), so switching to a richer 768-d target needs zero
  encoder rewiring — plus an **MSE-to-CLIP** loss term (`mse_weight`) alongside InfoNCE to hit the actual
  embedding (a usable decoder-conditioning latent), not only its direction.
- **Decoder (71n):** `reconstruct.py` decodes a predicted CLIP ViT-L/14 embedding via
  **StableUnCLIPImg2Img** (`diffusers/stable-diffusion-2-1-unclip-i2i-l`, `image_embeds` injection). Output is
  a per-concept triptych `[stimulus | decode(true CLIP) | decode(EEG-predicted)]` so **decoder quality and EEG
  quality read separately**.

## The CLIP-space trap (caught before it cost a run)

The unCLIP decoder was trained on **OpenAI** CLIP ViT-L/14. The first recon arm targeted **LAION** ViT-L/14 —
a *different embedding space*. Feeding LAION embeddings to an OpenAI-trained decoder decodes to **silent
garbage** (no error, wrong image). Fix: retargeted `vitl14` → OpenAI CLIP. (The researcher's cited model id
`stabilityai/stable-diffusion-2-1-unclip-small` was also dead/404 — the live diffusers-org conversion is
`diffusers/stable-diffusion-2-1-unclip-i2i-l`, verified image_encoder `projection_dim=768`, patch 14.)

## The recon-arm numbers (cbramod_ft + OpenAI ViT-L/14 + mse_weight 1.0, train1-4 / test5, 75ep, best-val ep61)

| metric | recon arm (vitl14+mse) | vitb32-ft baseline | Δ |
|--------|------------------------|--------------------|---|
| single-trial top1 | 1.82 | 2.38 | **−0.56** |
| single-trial top5 | 7.60 | 9.58 | −1.98 |
| concept-avg top1  | 2.50 | 4.50 | −2.00 |
| concept-avg top5  | 13.50 | 20.0 | −6.50 |
| emb_mse | 0.00253 | — | — |

**The recon objective COSTS retrieval** (all four metrics down vs the plain vitb32 fine-tune). This was
forecast (u9sv): richer target + MSE aim at recon fidelity, orthogonal to — and here costing — top-k.

**emb_mse 0.00253 is not the win it looks like.** On 768-d L2-unit vectors, MSE `= ‖a−b‖²/768`, so 0.00253 ⟺
`‖a−b‖² = 1.94` ⟺ **cos ≈ 0.03** — the predictions sit **near-orthogonal** to the true concept embedding.
That is exactly the established cross-subject faint-tilt floor ([perception-continuous-metrics]:
EEG embeddings point ~86–90° off the true concept; retrieval lives in a tiny positive *margin*, not a
semantic hit). The MSE term did **not** overcome the single-trial SNR floor; it re-found it. Confound noted:
this arm changes target *and* adds MSE — a vitl14-no-MSE control would separate the two (follow-up), but both
land on the same near-orthogonal floor, so the attribution barely matters for the endpoint.

## The unCLIP scale bug (the decode-garbage trap, and its fix)

First render: **even decode(true CLIP) was garbage** (colored text/noise), not the stimulus — so the fault was
in how embeddings reach the decoder, not the EEG. Diagnosis (cheap, image_encoder only): our open_clip target
and the pipe's HF-transformers CLIP are the **same space** (cosine 0.97), but our embeddings are
**L2-normalized (norm 1)** while raw OpenAI ViT-L/14 image embeds have norm **≈19** (measured mean over the 200
test images, std 1.0). unCLIP's `image_normalizer` was fit on the raw ~19-magnitude distribution, so a norm-1
vector lands far outside it → garbage. **Fix:** rescale embeddings to `_CLIP_SCALE = 19.0` before decoding
(`reconstruct.decode`). This is the deep-dive's flagged open question answered empirically: the decoder does
**not** assume L2-normalized input — it wants raw-magnitude embeddings.

(Also fixed en route: `_predict` lacked `torch.no_grad()` → the encoder retained the autograd graph across the
16k-trial test set, leaking ~30 GB of activations; and the unCLIP pipe was co-resident with the encoder on GPU.
Now: `no_grad` predict → free encoder → `enable_model_cpu_offload` pipe → decode, peak ~6.6 GB.)

## The reconstruction (triptych) — test5, 8 concepts

![EEG→image reconstruction triptych: stimulus | decode(true CLIP) | decode(EEG)](2026-07-17_eeg_image_reconstruction.png)

With the scale fix, the triptych `[stimulus | decode(true CLIP) | decode(EEG)]` is the honest result:

- **decode(true CLIP) = the decoder ceiling: faithful.** Warship→warship, antelope→antelope, wooden
  spoon→spoons, bench→wooden beam, bananas→yellow fruit on a plate, basil→potted green plant,
  basketball→basketball. The pipeline reconstructs the *stimulus's semantic content* from its true CLIP
  embedding — end-to-end validated.
- **decode(EEG-predicted) = faint / generic.** Overwhelmingly wrong: text placards, generic rooms, a person
  holding a card; at most one weak semantic hit (basil → a plant nursery). A cos-0.03 embedding points at a
  near-random direction, so unCLIP returns a plausible-but-unrelated image.

**This contrast is the deliverable, not a failure.** Cross-subject single-trial EEG→image on THINGS-EEG2 does
not carry enough per-trial semantic signal to reconstruct the viewed stimulus. The decoder ceiling proves the
pipeline is correct; the gap between columns 2 and 3 is the honest measure of what single-trial cross-subject
EEG actually encodes. The field's headline reconstructions are within-subject and/or trial-averaged over many
repetitions — the calibration the qoa epic exists to report.

## Verdict

- ooi recon objective: **KEEP the infrastructure** (target zoo + MSE term + emb_mse metric + decodable
  checkpoint), opt-in; **not a retrieval lever** (costs top-k, mse doesn't beat the SNR floor). z-score/vitb32
  stay the retrieval default.
- 71n decoder: **built + validated** (decoder-ceiling column). The EEG→image endpoint is reachable
  mechanically; the *quality* is floored by the cross-subject single-trial SNR wall, not the decoder.
- The portfolio deliverable is the **honest gap**: a working pipeline whose decoder ceiling is sharp and whose
  EEG reconstructions are faint — the calibrated truth about single-trial cross-subject visual decoding.
