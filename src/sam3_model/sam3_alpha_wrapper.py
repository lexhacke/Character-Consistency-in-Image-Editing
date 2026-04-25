"""
SAM3AlphaGating — alpha-gating wrapper around Sam3Image for two-image change detection.

Architecture (from PI's proposed diagram):
  e = PE(edited),  o = PE(original),  d = PE(|edited - original|)
  e_new = e + α·o + β·d          (learnable scalars, init 0 → warm start as vanilla SAM3)

  Text prompt goes through the native SAM3 text encoding path (unlike SAM3ChangeDetector
  which replaced text tokens entirely).  The fusion encoder cross-attends e_new against
  the text prompt, then DETR decoder + mask head proceed unchanged.

Batching convention (same as SAM3ChangeDetector / SAM3ChangeDetectionDataset):
  img_batch = [edited_0, orig_0, edited_1, orig_1, ...]   shape [2*B, 3, H, W]
  find_input.img_ids = [0, 2, 4, ...]   (edited indices)
  orig indices        = [1, 3, 5, ...]   (img_ids + 1)

Distance image:
  Computed on-the-fly in pixel space: |edited - original| (per-channel absolute diff).
  This is run through the *same* frozen backbone to produce d.
"""

from typing import Dict

import torch
import torch.nn as nn
from sam3.model.data_misc import BatchedDatapoint, FindStage
from sam3.model.geometry_encoders import Prompt
from sam3.model.model_misc import SAM3Output
from sam3.model.sam3_image import Sam3Image


class SAM3AlphaGating(Sam3Image):
    """
    Subclass of Sam3Image that fuses original-image and distance-image backbone
    features into the edited-image features via learnable scalars α and β.

    Trainable parameters:
      - alpha, beta (2 scalars)
      - transformer (fusion encoder + DETR decoder)
      - segmentation_head
      - presence-related heads

    Frozen parameters (~750M):
      - backbone (PE ViT + text encoder + neck)

    Construction:
        model = SAM3AlphaGating.from_sam3_image_model(
            build_sam3_image_model(eval_mode=False), freeze_backbone=True
        )
    """

    @classmethod
    def from_sam3_image_model(
        cls, base_model: Sam3Image, freeze_backbone: bool = True
    ) -> "SAM3AlphaGating":
        """Upgrade an existing Sam3Image to SAM3AlphaGating in-place."""
        device = next(base_model.parameters()).device
        base_model.__class__ = cls

        if freeze_backbone:
            for p in base_model.backbone.parameters():
                p.requires_grad_(False)
            base_model.backbone.eval()
            base_model._freeze_backbone = True
        else:
            base_model._freeze_backbone = False

        # Learnable gating scalars — init to 0 so training starts as vanilla SAM3.
        base_model.alpha = nn.Parameter(torch.zeros(1, device=device))
        base_model.beta = nn.Parameter(torch.zeros(1, device=device))

        return base_model

    def train(self, mode: bool = True):
        super().train(mode)
        if getattr(self, "_freeze_backbone", False):
            self.backbone.eval()
        return self

    # ------------------------------------------------------------------
    # Training forward — alpha-gates backbone features, uses native text path
    # ------------------------------------------------------------------

    def forward(self, input: BatchedDatapoint) -> SAM3Output:
        """
        input.img_batch: [2*B, 3, H, W]  (edited and original interleaved)
        input.find_inputs[0].img_ids: [0, 2, 4, ...]  (edited indices)
        """
        device = self.device

        # --- 1. Run backbone on all images (edited + original) ---
        with torch.no_grad():
            all_backbone_out = self.backbone.forward_image(input.img_batch)

        assert len(input.find_inputs) == 1
        find_input = input.find_inputs[0]
        find_target = input.find_targets[0]

        edited_ids = find_input.img_ids          # [B] e.g. [0, 2, 4, ...]
        orig_ids = find_input.img_ids + 1        # [B] e.g. [1, 3, 5, ...]

        # --- 2. Compute distance images and run backbone on them ---
        edited_imgs = input.img_batch[edited_ids]   # [B, 3, H, W]
        orig_imgs = input.img_batch[orig_ids]        # [B, 3, H, W]
        dist_imgs = (edited_imgs - orig_imgs).abs()  # [B, 3, H, W]

        with torch.no_grad():
            dist_backbone_out = self.backbone.forward_image(dist_imgs)

        # --- 3. Alpha-gate: e_new = e + α·o + β·d at every FPN level ---
        gated_fpn = []
        for level_idx, fpn_feats in enumerate(all_backbone_out["backbone_fpn"]):
            # fpn_feats: [2*B, C, H_l, W_l]
            e = fpn_feats[edited_ids]                                    # [B, C, H_l, W_l]
            o = fpn_feats[orig_ids]                                      # [B, C, H_l, W_l]
            d = dist_backbone_out["backbone_fpn"][level_idx]             # [B, C, H_l, W_l]
            e_new = e + self.alpha * o + self.beta * d                   # [B, C, H_l, W_l]
            gated_fpn.append(e_new)

        # Also gate the pos encodings? No — positional encodings are spatial and
        # don't depend on image content.  We just index them for the edited images.
        gated_pos = [
            pos[edited_ids]
            for pos in all_backbone_out["vision_pos_enc"]
        ]

        # --- 4. Build a new backbone_out with only B images (the gated edited ones) ---
        # The standard _get_img_feats indexes backbone_fpn[img_ids], so we need
        # img_ids to be [0, 1, 2, ..., B-1] (one per sample in the gated batch).
        B = edited_ids.shape[0]
        new_img_ids = torch.arange(B, device=device)

        backbone_out = {
            "img_batch_all_stages": input.img_batch,
            "backbone_fpn": gated_fpn,
            "vision_pos_enc": gated_pos,
        }

        # --- 5. Text encoding (native SAM3 path) ---
        text_outputs = self.backbone.forward_text(
            input.find_text_batch, device=device
        )
        backbone_out.update(text_outputs)

        # --- 6. Remap find_input to use new 0-indexed img_ids ---
        # We need to temporarily replace img_ids so _get_img_feats indexes correctly
        # into our B-sized gated_fpn (not the 2*B original).
        original_img_ids = find_input.img_ids
        find_input.img_ids = new_img_ids

        geometric_prompt = Prompt(
            box_embeddings=find_input.input_boxes,
            box_mask=find_input.input_boxes_mask,
            box_labels=find_input.input_boxes_label,
        )

        out = self.forward_grounding(
            backbone_out=backbone_out,
            find_input=find_input,
            find_target=find_target,
            geometric_prompt=geometric_prompt.clone(),
        )

        # Restore original img_ids
        find_input.img_ids = original_img_ids

        # Ensure matching indices are present for validation loss computation
        if find_target is not None and "indices" not in out:
            self._compute_matching(out, self.back_convert(find_target))

        previous_stages_out = SAM3Output(
            iter_mode=SAM3Output.IterMode.LAST_STEP_PER_STAGE
        )
        previous_stages_out.append([out])
        return previous_stages_out
