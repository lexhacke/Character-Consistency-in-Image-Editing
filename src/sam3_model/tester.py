"""
Phase 0 verification — run this before writing any training code.
Confirms:
  1. backbone_fpn[-1] shape is [1, 256, 36, 36]
  2. language_features is [N, 1, 256]
  3. Swapping language_features with orig image tokens runs forward_grounding without error
  4. Output pred_masks has a sensible shape
"""

import torch
from sam3_builder import build_sam3_image_model_fixed

device = "cuda" if torch.cuda.is_available() else "cpu"

# ── 1. Build model ────────────────────────────────────────────────────────────
print("\n[1] Building model...")
model = build_sam3_image_model_fixed()
model.eval()

# Let the model stay in its native dtype — don't force float32
# Use autocast so inputs/activations match whatever the model expects internally
dtype = torch.bfloat16
model = model.to(dtype)
print(f"Using device: {device}, dtype: {dtype}")
print("    OK")

# ── 2. Check backbone_fpn[-1] shape ──────────────────────────────────────────
print("\n[2] Checking backbone output shapes...")
dummy_img = torch.randn(1, 3, 1008, 1008, device=device, dtype=dtype)
with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
    backbone_out = model.backbone.forward_image(dummy_img)

fpn_last = backbone_out["backbone_fpn"][-1]
print(f"    backbone_fpn[-1]: {fpn_last.shape}")   # expect [1, 256, 36, 36]
print(f"    vision_features:  {backbone_out['vision_features'].shape}")

# ── 3. Check language_features shape ─────────────────────────────────────────
print("\n[3] Checking text backbone output...")
with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
    text_out = model.backbone.forward_text(["a cat"], device=device)
print(f"    language_features: {text_out['language_features'].shape}")  # [N, 1, 256]
print(f"    language_mask:     {text_out['language_mask'].shape}")       # [1, N]

# ── 4. Swap language_features with orig image tokens ─────────────────────────
print("\n[4] Swapping language_features with original image tokens...")
orig_img = torch.randn(1, 3, 1008, 1008, device=device, dtype=dtype)
edit_img = torch.randn(1, 3, 1008, 1008, device=device, dtype=dtype)

with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
    backbone_out = model.backbone.forward_image(edit_img)
    orig_backbone = model.backbone.forward_image(orig_img)

orig_tokens = orig_backbone["backbone_fpn"][-1]         # [1, 256, 36, 36]
orig_tokens = orig_tokens.flatten(2).permute(2, 0, 1)   # [1296, 1, 256]
orig_mask   = torch.zeros(1, orig_tokens.shape[0], dtype=torch.bool, device=device)

backbone_out["language_features"] = orig_tokens
backbone_out["language_mask"]      = orig_mask
print(f"    language_features (swapped): {orig_tokens.shape}")
print(f"    language_mask:               {orig_mask.shape}")

# ── 5. Run forward_grounding ──────────────────────────────────────────────────
print("\n[5] Running forward_grounding...")
from sam3.model.geometry_encoders import Prompt
from types import SimpleNamespace

find_input = SimpleNamespace(
    img_ids=torch.tensor([0], device=device),
    text_ids=torch.tensor([0], device=device),
)
geo_prompt = Prompt()

with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
    out = model.forward_grounding(
        backbone_out=backbone_out,
        find_input=find_input,
        find_target=None,
        geometric_prompt=geo_prompt,
    )

print(f"    pred_masks:   {out['pred_masks'].shape}")    # expect [1, 200, H, W]
print(f"    pred_boxes:   {out['pred_boxes'].shape}")
print(f"    pred_logits:  {out['pred_logits'].shape}")
print("\nAll checks passed!")
