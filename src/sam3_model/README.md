# SAM3 Change Detection

This document describes the SAM3 architecture in detail (with tensor shapes), explains the `SAM3ChangeDetector` modification, its failure modes, and where the project goes from here.

---

## Table of Contents

1. [Original SAM3 Architecture](#1-original-sam3-architecture)
   - [Overview](#overview)
   - [Stage 1: Visual Backbone](#stage-1--visual-backbone-pe-vit)
   - [Stage 2: Text Backbone](#stage-2--text-backbone-pe-text-encoder)
   - [Stage 3: Prompt Assembly](#stage-3--prompt-assembly)
   - [Stage 4: Fusion Encoder](#stage-4--fusion-encoder)
   - [Stage 5: DETR Decoder](#stage-5--detr-decoder)
   - [Stage 6: Segmentation Head](#stage-6--segmentation-head)
   - [Complete Tensor Flow](#complete-tensor-flow)
2. [Our Modification: SAM3ChangeDetector](#2-our-modification-sam3changedetector)
3. [Why It Failed](#3-why-it-failed)
4. [What We Tried](#4-what-we-tried)
5. [Path Forward](#5-path-forward)
6. [Dataset & Training Reference](#6-dataset--training-reference)

---

## 1. Original SAM3 Architecture

### Overview

SAM3 is a grounded segmentation model. Given an image and a text prompt (or geometric prompt), it finds and segments all instances of the described concept.

Total parameters: ~850M
- ~750M in the PE backbone
- ~100M in the detector and decoder

The pipeline has six stages: visual backbone, text backbone, prompt assembly, fusion encoder, DETR decoder, and segmentation head.

---

### Stage 1 — Visual Backbone: PE ViT

**Module**: `Sam3DualViTDetNeck` + ViT  
**Code**: `SAM3VLBackbone.forward_image()` in `vl_combiner.py`

- Input image: `[B, 3, 1008, 1008]`
- Patch size: 14, but the neck downsamples by 2x, giving an effective stride of 7
- Output: `backbone_fpn` — a list of multi-scale feature maps

The top-level feature map used downstream:

```
backbone_fpn[-1]:  [B, 256, 72, 72]   <- image tokens (72x72 = 5184 tokens)
vision_features:   same as backbone_fpn[-1]
vision_pos_enc:    positional encodings matching each FPN level
```

---

### Stage 2 — Text Backbone: PE Text Encoder

**Module**: `VETextEncoder`  
**Code**: `SAM3VLBackbone.forward_text()` in `vl_combiner.py`

- Input: list of caption strings
- Tokenizer: SimpleTokenizer (BPE), max 32 tokens
- Outputs:

```
language_features:  [N, B, 256]   where N <= 32 (seq-first, batch-second)
language_mask:      [B, N]        True = padding token
language_embeds:    [1, B, 256]   CLS-like embedding before encoder
```

---

### Stage 3 — Prompt Assembly

**Code**: `Sam3Image._encode_prompt()` in `sam3_image.py:167`  
**Module**: `SequenceGeometryEncoder`

No attention occurs here — this is pure tensor assembly.

```python
txt_feats = language_features[:, txt_ids]        # [N, B, 256]  (index by query)
geo_feats = geometry_encoder(bbox, img_feats)    # [K, B, 256]  (ROI-align + proj)
prompt    = cat([txt_feats, geo_feats], dim=0)   # [N+K, B, 256]
```

- The geometry encoder uses ROI-align to pool from the edited image's own backbone features.
- `K` = number of geometric tokens (0 if no box or point prompt is provided).

---

### Stage 4 — Fusion Encoder

**Module**: `TransformerEncoderFusion` (6 layers)  
**Code**: `TransformerEncoderFusion.forward()` in `encoder.py:464`

This is where image features are conditioned on the prompt.

Image tokens from `backbone_fpn[-1]` are flattened to `[B, 5184, 256]`.

Each of the 6 `TransformerEncoderLayer` layers applies:

```
# 1. Image self-attention (image tokens communicate with each other)
h += SDPA(h, h, h)       Q=K=V = image tokens

# 2. Image -> prompt cross-attention (image reads prompt)
h += SDPA(h, p, p)       Q = image tokens, K=V = prompt tokens

# 3. FFN
h = FFN(h)
```

**Critical detail**: prompt tokens are keys and values only — they are never updated in the encoder. Only the image tokens `h` are updated.

Output:

```
memory:  [5184, B, 256]   (seq-first)
```

Image tokens are now conditioned on the prompt.

Note: `add_pooled_text_to_img_feat=False` in the actual model builder — the pooling code path is disabled.

---

### Stage 5 — DETR Decoder

**Module**: `TransformerDecoder` (6 layers)  
**Code**: `TransformerDecoder` + `TransformerDecoderLayer` in `decoder.py`

Fresh learnable object queries drive this stage — not image features.

```
Q = query_embed.weight    [200, B, 256]   (200 learnable queries)

for each of 6 TransformerDecoderLayer layers:
  # 1. Query self-attention (queries suppress duplicates and compete)
  Q += SDPA(Q, Q, Q)

  # 2. Query -> prompt cross-attention (query asks "what am I looking for?")
  Q += SDPA(Q, prompt, prompt)      # ca_text, use_text_cross_attention=True

  # 3. Query -> memory cross-attention (query asks "where is it in the image?")
  Q += SDPA(Q, memory, memory)      # deformable attention

  # 4. FFN
  Q = FFN(Q)

-> hs  [L=6, B, 200, 256]    (intermediate outputs from all 6 layers)
```

**Presence token**: a special extra token is prepended to `Q` in the decoder. It attends to everything `Q` attends to. Its final hidden state produces a scalar `presence_logit` — "is the described concept present in this image?"

**Output heads**:

```
pred_logits = dot_product_scoring(hs, prompt)            [B, 200, 1]   (each query dot vs prompt)
pred_boxes  = bbox_embed_MLP(hs) + reference_boxes_offset  [B, 200, 4]  (cx, cy, w, h)
```

---

### Stage 6 — Segmentation Head

**Module**: `UniversalSegmentationHead`, `PixelDecoder`, `MaskPredictor`  
**Code**: `maskformer_segmentation.py`

#### A. PixelDecoder — FPN-style upsample

```
Input: backbone_fpn (multi-scale)
       BUT backbone_fpn[-1] is REPLACED with encoder memory reshaped:
       [5184, B, 256] -> [B, 256, 72, 72]

FPN upsample (3 stages, nearest-neighbor + conv + GroupNorm + ReLU):
  72x72 -> 144x144 -> 288x288

-> pixel_embed  [B, 256, 288, 288]
```

Why replace `backbone_fpn[-1]` with the encoder output? The encoder output is already conditioned on the prompt. Coarser FPN levels add back fine-grained spatial detail via skip connections.

#### B. Extra prompt cross-attention (enabled in actual model)

```python
encoder_hidden_states += cross_attn(encoder_hidden_states, prompt, prompt)
```

#### C. Mask prediction via dot product

```
instance_embeds = Conv2d(pixel_embed)                  [B, 256, 288, 288]
mask_vec        = mask_embed_MLP(hs[-1])               [B, 200, 256]   (3-layer MLP on decoder output)
pred_masks      = einsum("bqc,bchw->bqhw",
                          mask_vec, instance_embeds)   [B, 200, 288, 288]
```

Each of the 200 queries produces one mask. At inference the highest-scoring query is selected.

---

### Complete Tensor Flow

```
Image [B, 3, 1008, 1008]
  -> PE ViT backbone
  -> backbone_fpn[-1]:  [B, 256, 72, 72]   (5184 tokens)

Text (captions)
  -> PE text encoder
  -> language_features: [N<=32, B, 256]

_encode_prompt:
  prompt = cat(language_features, geo_feats)   [N+K, B, 256]

TransformerEncoderFusion (6 layers):
  h [B, 5184, 256]:  self-attn + cross-attn-to-prompt + FFN
  -> memory [5184, B, 256]

TransformerDecoder (6 layers):
  Q [200, B, 256] (learnable):  self-attn + cross-attn-to-prompt + cross-attn-to-memory + FFN
  -> hs         [6, B, 200, 256]
  -> pred_boxes [B, 200, 4]
  -> pred_logits [B, 200, 1]

PixelDecoder:
  backbone_fpn (with memory substituted at top level)
  -> pixel_embed [B, 256, 288, 288]

MaskPredictor:
  einsum(mask_embed_MLP(hs[-1]), pixel_embed)
  -> pred_masks [B, 200, 288, 288]
```

---

## 2. Our Modification: SAM3ChangeDetector

### Core Idea

Replace the text prompt with the original image's spatial features. The original image acts as a semantic reference: "find what changed between this reference and the edited image."

The model receives an (original, edited) image pair. The edited image is processed normally through the PE backbone. The original image's backbone features are injected into the prompt slot, so the fusion encoder and DETR decoder learn to cross-attend to spatial image tokens instead of text tokens.

### What Changes

The modification is a single override of `_encode_prompt`. Everything else is untouched.

```
language_features: [N, B, 256]     <- replaced with ->
orig_tokens:       [5184, B, 256]
```

- Original image features come from `backbone_fpn[-1][img_ids + 1]` (odd indices in the batch).
- A small `orig_proj: Linear(256, 256)`, identity-initialized, adapts them to the prompt space.
- **2D positional encoding** from `vision_pos_enc[-1]` is pre-added to the original image tokens before they enter the prompt slot. This is critical because the fusion encoder never applies positional encoding to the prompt/memory side (by design — text tokens don't need spatial positions). Without this, the model would have no way to know where each original image token came from spatially during cross-attention.
- Batching convention: `img_batch = [edited_0, orig_0, edited_1, orig_1, ...]`
  - Edited images at even indices, original images at odd indices.

### What Does NOT Change

| Component | Status |
|---|---|
| PE backbone (~750M params) | Frozen, untouched |
| `TransformerEncoderFusion` (6 layers) | Unchanged |
| `TransformerDecoder` (6 layers, 200 queries) | Unchanged |
| `PixelDecoder` + `MaskPredictor` | Unchanged |
| Loss functions (focal + dice + presence BCE) | Same as original SAM3 |
| SAM3 source code | Zero edits — pure subclass override |

### Implementation

`SAM3ChangeDetector` subclasses `Sam3Image` and overrides `_encode_prompt()`. It is "upgraded" onto an existing `Sam3Image` instance via `from_sam3_image_model()` to avoid duplicating the complex `__init__` parameter list.

---

## 3. Why It Failed

### The shortcut: saliency detection instead of change detection

After ~14 training runs across multiple hyperparameter settings, learning rate schedules, and architectural variants, all runs converge to the same failure mode: **the model segments the most salient object in the edited image, completely ignoring the original image.**

Visual inspection of predicted masks across all runs shows the model producing clean segmentations of prominent objects (crocodiles, people, buildings, picnic tables) rather than masks of *what changed* between the image pair. The loss converges — sometimes to seemingly reasonable values — but the model is solving the wrong problem.

### Why SAM3 can't learn to compare

The fundamental issue is **modality laziness** (a well-documented failure mode in multimodal learning). When one input modality is sufficient to achieve low loss, the model learns to ignore the other.

In standard SAM3, the text prompt is **load-bearing**: without it, the model has no idea *what* to segment. It must attend to the prompt or it produces garbage. The gradient pressure to use the prompt is absolute.

In our modification, the edited image alone contains everything the model needs to produce a "reasonable" mask. The 95M trainable parameters inherited strong pretrained weights for saliency-based segmentation. The model can achieve moderate loss by:

1. Ignoring the original image prompt entirely (cross-attention weights → 0)
2. Falling back on its pretrained ability to segment salient objects
3. Getting partial overlap with GT masks (salient objects often overlap with changed regions)

The Hungarian matcher finds *some* assignment between predicted and GT masks, the dice/focal losses produce moderate (not catastrophic) gradients, and the model settles into this local minimum. The loss going from ~280 → ~85 over 4 epochs is the model recovering its pretrained saliency capabilities, not learning change detection.

### The residual connection makes ignoring the prompt free

The fusion encoder architecture makes this shortcut structurally trivial:

```
h = h + cross_attn(Q=h, K=prompt, V=prompt)
```

If cross-attention weights go to zero, this reduces to `h = h + 0 = h`. The residual connection means **ignoring the prompt has zero cost** — the edited image features pass through unchanged. There is no mechanism that forces the model to read the prompt.

This is fundamentally different from the text prompt case, where the model *must* attend to text to distinguish "segment the dog" from "segment the table."

### Why a 400K-param U-Net succeeded where 95M params failed

A small U-Net trained from scratch on 20-50 image pairs converged correctly on this task. This seems paradoxical — more capacity and more pretraining should help, not hurt.

The explanation: the U-Net had **no pretrained shortcut available.** It processed both images as concatenated channel input from pixel level. The only way to minimize loss was to actually learn to compare the two images. The architecture couldn't "ignore" one image because they were fused at the very first layer.

SAM3's pretrained weights create an enormous basin of attraction toward saliency detection. Fine-tuning 95M parameters can't escape this basin because the shortcut loss is good enough that gradients don't push the model toward the harder (but correct) comparison solution.

---

## 4. What We Tried

### Experiment 1: Cross-attention prompt injection (original approach)

Inject original image backbone features (`[5184, B, 256]`) into the prompt slot where text tokens normally go.

**Result**: Model produces saliency masks of the edited image. Loss converges (~280 → ~45 over 20 epochs). Predicted masks show clean segmentations of salient objects (crocodiles, people, buildings) unrelated to actual changes.

**Runs**: 9 runs from 2026-04-06 to 2026-04-10 (see W&B project). Best val_loss ~3.2 (bs=2, lr=1e-5, normalized for two-mask objective). All show the same saliency-detection behavior.

### Experiment 2: Cross-attention ablation (prompt zeroed)

Set `orig_proj(orig_tokens) * 0` — zero out the projected original image tokens before adding positional encoding.

**Result**: Identical masks. Same loss convergence. The model produces the exact same saliency-based segmentations with or without original image content in the prompt.

**Conclusion**: The cross-attention mechanism provides zero meaningful signal. The model ignores the prompt entirely, confirming the modality laziness hypothesis.

### Experiment 3: Latent feature difference

Replace the edited image's backbone features with `PE(edited) - PE(original)` before the fusion encoder. Remove the cross-attention prompt entirely.

**Result**: Complete collapse. ~50% higher loss than baseline. Model outputs blank/empty masks. Presence head learns "nothing here."

**Explanation**: The dataset filters for image pairs with DINO similarity > 0.94. In the PE feature space, near-identical images produce near-identical features. The difference `PE(edited) - PE(original) ≈ 0` everywhere. The pretrained decoder receives near-zero input and correctly concludes nothing is present.

### Experiment 4: Alpha-gating variant (`sam3_alpha_wrapper.py`)

Learned per-channel gating between edited and original features at each FPN level.

**Result**: Same saliency detection behavior. Loss converges but masks show no change-detection signal.

---

## 5. Path Forward

### Why E2E SAM3 change detection is structurally blocked

The frozen PE backbone cannot learn to compare two images — it processes each independently and produces features optimized for single-image understanding. The trainable decoder has a cheaper solution available (saliency detection via pretrained weights) and no gradient pressure strong enough to overcome it. Unfreezing the full 850M-param backbone risks destroying SAM3's segmentation ability and requires far more data than we have.

**The E2E SAM3 approach is abandoned.**

### Recommended architecture: Change detector + SAM3 prompt pipeline

Separate the two sub-tasks:

1. **Change detection** (what changed, roughly where): A lightweight model that takes two images and produces bounding boxes or point prompts for changed regions. The U-Net proof-of-concept showed this is learnable from scratch with very little data.

2. **Precise segmentation** (pixel-perfect masks from prompts): Stock SAM3, used exactly as designed — given a box or point prompt, produce a high-quality segmentation mask. No fine-tuning needed.

This plays to SAM3's actual strength (precise prompted segmentation) without asking it to perform comparison, which it cannot do with a frozen backbone.

### Alternative: VLM routing (current working pipeline)

The existing Gemini-based pipeline already works: VLM analyzes the edit prompt and image pair, determines what to segment and from which image, then SAM3 segments based on text/geometric prompts. This is accurate but adds VLM latency and API cost. The change-detector approach above would replace the VLM with a fast, local model.

---

## 6. Dataset & Training Reference

### Dataset Requirements

Each training sample needs:

```
clean_data/{i}/
    base.jpeg              — original image when meta["base"] == "original", edited otherwise
    other.jpeg             — the counterpart
    subtraction_mask.png   — mask of regions removed from original (may be all-black if N/A)
    union_mask.png         — mask of regions added in the edit (may be all-black if N/A)
    meta.json              — see below
```

`meta.json` determines which masks are valid:

```json
{
    "prompt": "Replace the film roll with a light meter...",
    "base": "original",
    "subtraction": {
        "success": ["green cylindrical Kodak 120 film roll"],
        "failed": []
    },
    "union": {
        "success": ["small vintage silver-colored rectangular photographic light meter"],
        "failed": []
    },
    "similarity_score": 0.9965
}
```

- `len(subtraction.success) > 0` → subtraction mask is real (thing removed from original)
- `len(union.success) > 0` → union mask is real (thing added in the edit)
- Both empty → sample is skipped (no valid change to learn from)

This yields three training cases:

| Edit type | subtraction | union | Example |
|---|---|---|---|
| Addition | empty | valid | "Add a duck to the table" |
| Removal | valid | empty | "Remove the dog" |
| Replacement | valid | valid | "Replace the film roll with a light meter" |

SAM3's Hungarian matcher and DETR loss handle all cases natively:
- 0 objects → presence token trains to 0, per-query losses zeroed
- 1 object → 1 of 200 queries matched, 199 unmatched
- 2 objects → 2 of 200 queries matched, 198 unmatched

The existing `src/data_collection/` pipeline produces these — see `clean_data/` outputs.

### Training Config

Training is configured via `src/sam3_model/config.json`:

```json
{
    "max_epochs": 20,
    "batch_size": 2,
    "lr": 1e-5,
    "min_sim": 0.94,
    "val_fraction": 0.1,
    "num_images_logged": 8,
    "resolution": 1008,
    "data_subdir": "clean_data"
}
```

Training runs on Modal via `train_modal.py`:

```bash
# Full training
modal run src/sam3_model/train_modal.py

# Overfit sanity check
modal run src/sam3_model/train_modal.py --overfit-batches 1 --max-epochs 100
```

Upload data to Modal volume first:

```bash
modal volume put clean-data C:/path/to/clean_data /clean_data
```

### Wandb Logging

Training and validation log 6 mask images per sample:
- `sub_mask_gt` / `union_mask_gt` — ground-truth subtraction and union masks
- `sub_mask_pred` / `union_mask_pred` — top-2 predicted masks by confidence

Note: predicted masks are the top-2 by logit score, not by Hungarian assignment. They may not correspond 1:1 to sub vs union, but show whether the model learns to produce two distinct masks.
