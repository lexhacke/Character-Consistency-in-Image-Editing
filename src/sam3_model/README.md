# SAM3 Change Detection

This document describes the SAM3 architecture in detail (with tensor shapes), explains the `SAM3ChangeDetector` modification, and outlines the training plan.

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
3. [Next Steps](#3-next-steps)

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

### Phase 0 Verification

Shape checks confirmed:

```
backbone_fpn[-1]:              [1, 256, 72, 72]    OK
language_features (swapped):   [5184, 1, 256]      OK
pred_masks:                    [1, 200, 288, 288]   OK
backbone frozen:               705/705 params       OK
orig_proj:                     identity init        OK
```

---

## 3. Next Steps

### Phase 1 — Fine-tune detector only (current plan)

Freeze the PE backbone entirely (~750M params). Train only:

| Module | Approx. params |
|---|---|
| `TransformerEncoderFusion` (6-layer fusion encoder) | ~20M |
| `TransformerDecoder` (6-layer DETR decoder + bbox/presence heads) | ~60M |
| `UniversalSegmentationHead` (pixel decoder + mask predictor) | ~15M |
| `orig_proj` (256x256 linear) | ~65K |
| **Total trainable** | **~95M** |

**Rationale**: The PE backbone already produces good 256-dim spatial features. The detector was trained to cross-attend to text tokens — it needs to learn to do the same with image tokens. The mask decoder uses those detector outputs directly, so it also needs updating.

### Things to try if the model fails

Firstly, we might want to straight up concatenate the image editing prompt into the prompt encoding. SAM3 does this natively anyway as if you have an examplar prompt AND a text prompt, both encodings get concatenated together and passed through the fusion encoder as one. So we could just concatenate the PE text encoding of the original edit instruction onto the prompt encoding.

If the model fails to learn meaningful change masks with the backbone fully frozen, the likely issue is that the original image's features are not well-adapted to the prompt role — they were trained as spatial descriptors, not concept descriptors.

Solution: add LoRA adapters (~2-5M params each) to the PE backbone's attention layers for the original-image path, leaving the edited-image path unchanged. This avoids cloning 750M params while allowing the backbone to specialize for the reference role.

Do not attempt this until Phase 1 training shows clear evidence of failure (e.g. loss plateaus above random baseline, or attention maps show no meaningful localization of changed regions).

### Dataset Requirements

Each training sample needs:

```python
{
    "image_original": PIL.Image,   # reference image
    "image_edited":   PIL.Image,   # same scene with edits applied
    "mask_gt":        Tensor,      # bool [H, W] — True where pixels changed
    "presence_gt":    bool,        # True always (edited region is always present)
}
```

The existing `src/data_collection/` pipeline produces these — see `data_sample/success/` outputs.

### Training Config

Adapt `sam3/sam3/train/configs/odinw13/odinw_text_only_train.yaml`. Key changes:

```yaml
dataset:           src/sam3_model/sam3_dataset.py
freeze_image_tower: FullFreeze
freeze_text_tower:  True
enable_segmentation: True
# Lower LR: backbone is frozen, detector trains fast, risk of overfitting is real
```
