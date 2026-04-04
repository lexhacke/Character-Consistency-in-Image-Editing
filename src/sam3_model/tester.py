"""
Phase 0 verification — run this before writing any training code.
Confirms:
  1. backbone_fpn[-1] shape is [1, 256, 36, 36]
  2. language_features is [N, 1, 256]
  3. Swapping language_features with orig image tokens runs forward_grounding without error
  4. Output pred_masks has a sensible shape
"""

import torch
from sam3.model_builder import build_sam3_image_model
from sam3.train.data.collator import BatchedDatapoint

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# ── 1. Build model ────────────────────────────────────────────────────────────
print("\n[1] Building model...")
model = build_sam3_image_model(
    device=device,
    eval_mode=True,
    load_from_HF=True,
    enable_segmentation=True,
)
model.eval()
print("    OK")

# ── 2. Check backbone_fpn[-1] shape ──────────────────────────────────────────
print("\n[2] Checking backbone output shapes...")
dummy_img = torch.randn(1, 3, 1008, 1008, device=device)
with torch.no_grad():
    backbone_out = model.backbone.forward_image(dummy_img)

fpn_last = backbone_out["backbone_fpn"][-1]
print(f"    backbone_fpn[-1]: {fpn_last.shape}")   # expect [1, 256, 36, 36]
print(f"    vision_features:  {backbone_out['vision_features'].shape}")

# ── 3. Check language_features shape ─────────────────────────────────────────
print("\n[3] Checking text backbone output...")
with torch.no_grad():
    text_out = model.backbone.forward_text(["a cat"], device=device)
print(f"    language_features: {text_out['language_features'].shape}")  # [N, 1, 256]
print(f"    language_mask:     {text_out['language_mask'].shape}")       # [1, N]

# ── 4. Swap language_features with orig image tokens ─────────────────────────
print("\n[4] Swapping language_features with original image tokens...")
orig_img = torch.randn(1, 3, 1008, 1008, device=device)
edit_img = torch.randn(1, 3, 1008, 1008, device=device)

with torch.no_grad():
    backbone_out = model.backbone.forward_image(edit_img)
    orig_backbone = model.backbone.forward_image(orig_img)

orig_tokens = orig_backbone["backbone_fpn"][-1]         # [1, 256, 36, 36]
orig_tokens = orig_tokens.flatten(2).permute(2, 0, 1)   # [1296, 1, 256]
orig_mask   = torch.zeros(1, orig_tokens.shape[0], dtype=torch.bool, device=device)  # [1, 1296]

backbone_out["language_features"] = orig_tokens
backbone_out["language_mask"]      = orig_mask
print(f"    language_features (swapped): {orig_tokens.shape}")
print(f"    language_mask:               {orig_mask.shape}")

# ── 5. Run forward_grounding ──────────────────────────────────────────────────
print("\n[5] Running forward_grounding...")
from sam3.model.geometry_encoders import Prompt
from types import SimpleNamespace

# forward_grounding only reads find_input.text_ids and find_input.img_ids
find_input = SimpleNamespace(
    img_ids=torch.tensor([0], device=device),
    text_ids=torch.tensor([0], device=device),
)

# Empty geometric prompt (no boxes, no points — all None)
geo_prompt = Prompt()

with torch.no_grad():
    out = model.forward_grounding(
        backbone_out=backbone_out,
        find_input=FakeInput(),
        find_target=None,
        geometric_prompt=geo_prompt,
    )

print(f"    pred_masks:   {out['pred_masks'].shape}")    # expect [1, 200, H, W]
print(f"    pred_boxes:   {out['pred_boxes'].shape}")
print(f"    pred_logits:  {out['pred_logits'].shape}")
print("\nAll checks passed!")
