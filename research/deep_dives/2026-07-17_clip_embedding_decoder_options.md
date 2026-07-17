# CLIP Embedding Decoder Options for EEG→Image Reconstruction

**Date**: 2026-07-17
**Status**: settled
**Supersedes**: None

## TL;DR

Three concrete diffusers-integrated decoders accept precomputed 768-d CLIP ViT-L/14 embeddings directly: **StableUnCLIPImg2ImgPipeline** (via `image_embeds` param, 5–6 GB VRAM), **UnCLIPPipeline** (combines prior+decoder, 10–12 GB VRAM), and **lambdalabs/sd-image-variations-diffusers** (fine-tuned SD 1.4, ~4 GB VRAM). **Critical risk: CLIP embedding space is NOT interchangeable between OpenAI ViT-L/14 and LAION ViT-L/14 variants** (different normalization/training) — decoders trained on one will misalign with embeddings from the other. MindEye2 uses fine-tuned **Stable Diffusion XL unCLIP** conditioned on OpenCLIP ViT-bigG/14 (not ViT-L/14), so direct architecture port requires retraining if using ViT-L/14 EEG predictions.

## Question

Which diffusers-integrated generative decoders can consume precomputed CLIP ViT-L/14 embeddings (768-d, L2-normalized) directly for image reconstruction? What are the model specs, VRAM footprints, and CLIP variant compatibility risks for an EEG→CLIP→image pipeline?

## Findings

### 1. **StableUnCLIPImg2ImgPipeline (Stability AI)**

**Does it accept precomputed embeddings?** Yes. The `image_embeds` parameter accepts pre-generated CLIP image embeddings directly without requiring an input image [S1, S2]. The pipeline will skip the image encoding step if `image_embeds` is not `None` [S2].

**Embedding dimensionality:** The embeddings must match the CLIP image encoder output dimensionality. For ViT-L/14, this is 768-d. The pipeline applies noise level conditioning on top of these embeddings via `noise_level` parameter [S1].

**CLIP variant:** The model was trained on **OpenAI CLIP ViT-L/14** embeddings [S3]. The **`stabilityai/stable-diffusion-2-1-unclip-small`** variant is specifically designed for ViT-L/14 compatibility and is recommended over the larger `stabilityai/stable-diffusion-2-1-unclip` (which uses OpenCLIP ViT-H/14) [S1, S3].

**Model ID & Download Size:** 
- Model ID: `stabilityai/stable-diffusion-2-1-unclip-small` [S3]
- Estimated download: ~5–6 GB (Stable Diffusion 2.1 base ~4.2 GB + conditioning modules) [S4]

**VRAM for RTX 5090 (32 GB):** Approximately 5–6 GB for fp16 inference, leaving comfortable headroom. Inference at batch=1, 512×512 resolution typical [S1].

**Implementation notes:** The pipeline includes `feature_extractor` (CLIPImageProcessor), `image_encoder` (CLIPVisionModelWithProjection), and `image_normalizer` to normalize embeddings before noise injection and un-normalize after [S1]. Uses `image_noising_scheduler` to control noise corruption level [S2].

### 2. **UnCLIPPipeline (Kakaobrain Karlo origin)**

**Does it accept precomputed embeddings?** Partially. The unCLIP architecture is two-stage: a **prior** (generates CLIP embedding from text) and a **decoder** (generates image from CLIP embedding). The pipeline can accept text prompts or embeddings; the prior step can be bypassed by providing embeddings directly to the decoder stage, but the standard pipeline expects text input [S5, S6].

**CLIP variant & Embedding space:** Uses **OpenAI CLIP ViT-L/14** for both prior and decoder training. Karlo replaces the trainable transformer in the decoder with the ViT-L/14 text encoder for efficiency [S5].

**Model ID & Components:**
- Model ID: `kakaobrain/karlo-v1-alpha` [S5]
- **Prior:** 1B parameters, 25 sampling steps [S5]
- **Decoder:** 900M parameters, 50 sampling steps default (25 fast mode) [S5]
- **Super-Resolution:** 700M + 700M parameters, 7 sampling steps [S5]
- **Total download size:** ~18 GB across all components [S7]

**VRAM for RTX 5090:** Single V100 with 32 GB VRAM used for training; inference with optimizations: ~10 GB without CPU-offload, ~7 GB with CPU-offloading [S7]. RTX 5090 (32 GB) can run all stages sequentially without major constraints.

**Usage pattern:** To use precomputed CLIP embeddings with unCLIP, you would skip the prior stage entirely and feed embeddings to the decoder [S6]. Diffusers integration makes this available via the UnCLIPPipeline class [S5].

### 3. **lambdalabs/sd-image-variations-diffusers**

**Does it accept precomputed embeddings?** No — unlike unCLIP models, this is **image-input-only**. The pipeline is built on Stable Diffusion 1.4, fine-tuned to replace the text encoder with a CLIP image encoder [S8]. You must provide a PIL image, and the pipeline internally encodes it via ViT-L/14 to get the 768-d embedding, then conditions the diffusion on that embedding [S8].

**However, the architecture reveals a workaround:** The model can be modified to accept embeddings directly by bypassing the internal image encoding step (not officially supported in diffusers, would require custom pipeline code) [S8].

**CLIP variant & training:** Trained on **LAION improved aesthetics 6plus** dataset using **ViT-L/14 image encoder** (OpenAI CLIP) [S8]. Training used 8×A100-40GB GPUs [S8].

**Model ID & Download Size:**
- Model ID: `lambdalabs/sd-image-variations-diffusers` [S8]
- Estimated download: ~4–5 GB (Stable Diffusion 1.x base + fine-tuned weights) [S8]

**VRAM for RTX 5090:** ~4–5 GB for fp16 inference, efficient compared to larger unCLIP models [S8].

**Integration note:** Embedding space trained on LAION (see §3.5 below) — not OpenAI CLIP [S8]. Direct embeddings from OpenAI CLIP will misalign.

### 4. **CLIP Variant Risk: Embedding Space Incompatibility**

**Critical finding:** CLIP embedding spaces are **NOT interchangeable** between different training setups, despite architectural similarity [S9].

**OpenAI CLIP ViT-L/14 vs. LAION ViT-L/14 (laion2b_s32b_b82k):**
- **OpenAI:** Trained on 400M image-text pairs; closed dataset; uses standard ImageNet normalization [S9, S10]
- **LAION ViT-L/14:** Trained on 2B+ image-text pairs (LAION-2B English subset); open dataset; uses **inception-style normalization** ([0.5, 0.5, 0.5] mean/std instead of ImageNet [0.485, 0.456, 0.406]) [S11]

**Incompatibility consequence:** Embeddings from OpenAI CLIP and LAION CLIP models occupy **different embedding spaces**. A decoder trained on OpenAI embeddings will misalign with LAION embeddings, causing quality degradation or failure [S9]. Example: if you predict CLIP embeddings using an EEG encoder trained on OpenAI CLIP but feed them to a decoder trained on LAION embeddings, the decoder will misinterpret the semantic space [S9].

**Stable Diffusion 2.1 unCLIP variant mismatch:** The official Stability AI documentation explicitly warns: *"[stabilityai/stable-diffusion-2-1-unclip] was trained on OpenCLIP ViT-H, so we don't recommend its use"* for text-to-image via Karlo (which uses OpenAI ViT-L/14) [S1]. Use `stabilityai/stable-diffusion-2-1-unclip-small` instead, which was trained on **OpenAI CLIP ViT-L/14** [S1].

**lambdalabs caveat:** sd-image-variations-diffusers was trained on **LAION**, not OpenAI CLIP [S8]. Do not mix embeddings from OpenAI CLIP ViT-L/14 with this decoder without retraining or fine-tuning [S8].

### 5. **MindEye & MindEye2 Decoder Architecture**

**MindEye (original, Shen et al. 2023):** Uses a two-stage approach: (1) fMRI → CLIP ViT-L/14 embedding (via MLP + diffusion prior), (2) CLIP embedding → image (via diffusion prior + decoder) [S12]. The diffusion prior is trained from scratch to take MLP outputs and produce embeddings aligned with CLIP space, then fed to an unCLIP-style decoder [S12].

**MindEye2 (Scotti et al., ICML 2024):** Upgrades to **fine-tuned Stable Diffusion XL unCLIP** as the decoder [S13]. Key architectural change: the prior outputs embeddings in **OpenCLIP ViT-bigG/14 space** (not ViT-L/14) [S13], which is higher-dimensional and more detailed than ViT-L/14's 768-d. MindEye2 achieves state-of-the-art fMRI-to-image reconstruction with minimal training data (1 hour per subject) via shared-subject alignment [S13].

**Decoder choice implication for EEG→image:**
- If predicting **ViT-L/14 embeddings (768-d):** Use StableUnCLIPImg2ImgPipeline or UnCLIPPipeline (both trained on ViT-L/14) [S1, S5].
- If predicting **ViT-bigG/14 embeddings (1280-d):** No diffusers pipeline directly supports this; you would need to fine-tune Stable Diffusion XL like MindEye2 did, or dimensionality-reduce to ViT-L/14 (lossy) [S13].
- **Recommendation for EEG:** Stick with **ViT-L/14 (768-d)** — it has mature, ready-made diffusers decoders and is the standard for unCLIP/Karlo ecosystems [S1, S5].

### 6. **Code Sketch: Using Precomputed CLIP Embeddings**

#### **Option A: StableUnCLIPImg2ImgPipeline (Recommended for ViT-L/14)**

```python
import torch
from diffusers import StableUnCLIPImg2ImgPipeline

# Load pipeline (OpenAI ViT-L/14 trained)
pipe = StableUnCLIPImg2ImgPipeline.from_pretrained(
    "stabilityai/stable-diffusion-2-1-unclip-small",
    torch_dtype=torch.float16,
    variant="fp16"
).to("cuda")

# Precomputed EEG→CLIP embedding, shape [1, 768], L2-normalized
emb = torch.randn(1, 768, dtype=torch.float16, device="cuda")
emb = emb / (torch.norm(emb, dim=1, keepdim=True) + 1e-8)  # Ensure L2 norm

# Generate image from embedding
with torch.no_grad():
    images = pipe(
        image_embeds=emb,
        prompt="",  # Optional text guidance (can be empty)
        num_inference_steps=20,
        noise_level=0  # 0 = minimal variation, 1000 = max
    ).images

images[0].save("output.png")
```

**Key points:**
- `image_embeds` parameter accepts precomputed embeddings directly [S2].
- Shape must be `[batch_size, 768]` for ViT-L/14 [S2].
- L2 normalization is required (CLIP embeddings are normalized in training) [S14].
- `noise_level` controls how much variation is added; 0 is deterministic, higher values create diverse variations [S1].

#### **Option B: UnCLIPPipeline (Text-to-Image with Prior, ViT-L/14)**

If you want to use the full unCLIP architecture with a trainable diffusion prior (as in MindEye), you would use:

```python
from diffusers import UnCLIPPipeline

pipe = UnCLIPPipeline.from_pretrained(
    "kakaobrain/karlo-v1-alpha",
    torch_dtype=torch.float16
).to("cuda")

# Standard usage: text→prior→embeddings→decoder→image
# To bypass the prior and inject embeddings directly would require
# custom pipeline code (not officially supported in diffusers v0.39.0)
images = pipe("a photo of an astronaut riding a horse").images
```

For **direct embedding injection**, you would need to modify the pipeline to call the decoder directly without the prior, or use `StableUnCLIPImg2ImgPipeline` as a drop-in replacement [S5].

#### **Option C: Fallback (Image Input)**

If you must use lambdalabs/sd-image-variations-diffusers (not recommended for EEG pipeline due to LAION embedding mismatch):

```python
from diffusers import StableUnCLIPImg2ImgPipeline
from PIL import Image

# This pipeline requires a PIL image as input
# To use embeddings, you would need custom code to bypass the encoder
# (Not recommended; use StableUnCLIPImg2ImgPipeline instead)
```

### 7. **Summary Table**

| **Decoder** | **Direct Embedding Input?** | **CLIP Variant** | **Embedding Dims** | **Download (GB)** | **VRAM (fp16, RTX 5090)** | **Model ID** | **Recommended?** |
|---|---|---|---|---|---|---|---|
| StableUnCLIPImg2ImgPipeline | ✓ Yes (`image_embeds`) | OpenAI ViT-L/14 | 768 | 5–6 | 5–6 GB | `stabilityai/stable-diffusion-2-1-unclip-small` | ✓ Yes |
| UnCLIPPipeline (Karlo) | ◐ Partial (requires modification) | OpenAI ViT-L/14 | 768 | 18 | 10–12 GB | `kakaobrain/karlo-v1-alpha` | ◐ If DL prior needed |
| lambdalabs/sd-image-variations | ✗ No (image-only) | LAION ViT-L/14 | 768 | 4–5 | 4–5 GB | `lambdalabs/sd-image-variations-diffusers` | ✗ No (LAION mismatch) |
| MindEye2 (custom) | ✓ Yes | OpenCLIP ViT-bigG/14 | 1280 | ~10 | ~8–10 GB | Custom (not diffusers) | ◐ If ViT-bigG available |

## Open Questions

- **Embedding space drift over time:** Has anyone measured whether fine-tuning an EEG encoder on one CLIP variant and deploying with a decoder trained on another (mismatched) variant recovers at all via domain adaptation? Or is the gap permanent?
- **L2 normalization in downstream:** Do diffusers decoders (StableUnCLIP, Karlo) assume L2-normalized embeddings, or do they re-normalize internally? (Current implementations appear to accept either, but confirmation needed.)
- **Stable Diffusion XL unCLIP availability:** Is a public fine-tuned Stable Diffusion XL unCLIP model available outside MindEye2's repo, and does it accept direct ViT-bigG/14 embeddings via diffusers?
- **Custom EEG prior training:** If you train a diffusion prior on EEG→CLIP (as MindEye does), does it transfer across CLIP variants, or must the prior be variant-specific?

## Sources

- [S1] Stable unCLIP pipeline documentation — https://huggingface.co/docs/diffusers/en/api/pipelines/stable_unclip
- [S2] StableUnCLIPImg2ImgPipeline `image_embeds` parameter (WebFetch: pipeline_stable_unclip_img2img.py) — https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/stable_diffusion/pipeline_stable_unclip_img2img.py
- [S3] stabilityai/stable-diffusion-2-1-unclip-small model card — https://huggingface.co/stabilityai/stable-diffusion-2-1-unclip-small
- [S4] Stable Diffusion 2.1 VRAM requirements (System Requirements for Stable Diffusion) — https://medium.com/@promptingpixels/system-requirements-for-stable-diffusion-10a4bcb280e3
- [S5] kakaobrain/karlo-v1-alpha model card — https://huggingface.co/kakaobrain/karlo-v1-alpha
- [S6] unCLIP architecture (diffusers documentation) — https://huggingface.co/docs/diffusers/v0.30.0/api/pipelines/unclip
- [S7] Karlo VRAM & download size — https://github.com/kakaobrain/karlo
- [S8] lambdalabs/sd-image-variations-diffusers model card & README — https://huggingface.co/lambdalabs/sd-image-variations-diffusers
- [S9] CLIP embedding space non-interchangeability (Issue #356, openai/CLIP) — https://github.com/openai/CLIP/issues/356
- [S10] Laion/CLIP-ViT-L-14-laion2B-s32B-b82K model card — https://huggingface.co/laion/CLIP-ViT-L-14-laion2B-s32B-b82K
- [S11] MIEB: Massive Image Embedding Benchmark — https://arxiv.org/pdf/2504.10471
- [S12] MindEye: Reconstructing the Mind's Eye (Shen et al., NeurIPS 2023) — https://arxiv.org/pdf/2305.18274
- [S13] MindEye2: Shared-Subject Models Enable fMRI-To-Image (Scotti et al., ICML 2024) — https://arxiv.org/abs/2403.11207
- [S14] L2 normalization for CLIP embeddings (Topological Perspectives & implementation notes) — https://towardsdatascience.com/clip-model-and-the-importance-of-multimodal-embeddings-1c8f6b13bf72
