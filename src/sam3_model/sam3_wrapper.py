"""
SAM3ChangeDetector — fine-tunable wrapper around Sam3Image for two-image change detection.

Architecture insight:
  Sam3Image's fusion encoder cross-attends image tokens (queries) against prompt tokens
  (keys/values). Normally the prompt tokens come from backbone.forward_text(). Here we
  substitute the original image's backbone tokens as the prompt, exploiting the fact that
  PE's contrastive pretraining puts image and text tokens in the same 256-dim space.

Batching convention (must match SAM3ChangeDetectionDataset):
  Each Datapoint has two images: images[0] = edited, images[1] = original.
  After collation, img_batch is [edited_0, orig_0, edited_1, orig_1, ...].
  Input shape is gonna be [2*B, 3, H, W], and find_input.img_ids = [0, 2, 4, ...] (edited indices).
  find_input.img_ids = [0, 2, 4, ...] (edited indices).
  find_input.img_ids + 1 = [1, 3, 5, ...] (original indices).

Phase 0 inference helper:
  Use inject_original_as_prompt() + Sam3Processor._forward_grounding() to validate
  the dimension swap works before any training.
"""

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
from sam3.model.data_misc import FindStage
from sam3.model.geometry_encoders import Prompt
from sam3.model.model_misc import SAM3Output
from sam3.model.sam3_image import Sam3Image
from sam3.model.data_misc import BatchedDatapoint


class SAM3ChangeDetector(Sam3Image):
    """
    Subclass of Sam3Image that uses the original image's backbone tokens as the
    "language" prompt instead of text tokens.

    Trainable parameters (~100M):
      - transformer (fusion encoder + DETR decoder)
      - segmentation_head
      - orig_proj (small 256→256 linear adapter, initialised as identity)
      - presence-related heads

    Frozen parameters (~750M):
      - backbone (PE ViT + text encoder + neck)

    Construction:
        Don't call __init__ directly — use the classmethod:
            model = SAM3ChangeDetector.from_sam3_image_model(
                build_sam3_image_model(eval_mode=False), freeze_backbone=True
            )
    """

    @classmethod
    def from_sam3_image_model(
        cls, base_model: Sam3Image, freeze_backbone: bool = True
    ) -> "SAM3ChangeDetector":
        """
        Upgrade an existing Sam3Image to SAM3ChangeDetector in-place.

        This avoids duplicating all of Sam3Image.__init__'s complex parameter list.
        The base_model's class is changed to cls; orig_proj is added as a new sub-module.
        """
        device = next(base_model.parameters()).device
        base_model.__class__ = cls

        if freeze_backbone:
            for p in base_model.backbone.parameters():
                p.requires_grad_(False)

        # Small projection to adapt image tokens → prompt-token space.
        # Initialised as identity so training starts from the pretrained representation.
        proj = nn.Linear(base_model.hidden_dim, base_model.hidden_dim)
        nn.init.eye_(proj.weight)
        nn.init.zeros_(proj.bias)
        base_model.add_module("orig_proj", proj.to(device))

        return base_model

    def __init__(self, freeze_backbone: bool = True, **kwargs):
        # Only used if constructing from scratch (not the recommended path).
        super().__init__(**kwargs)
        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad_(False)
        self.orig_proj = nn.Linear(self.hidden_dim, self.hidden_dim)
        nn.init.eye_(self.orig_proj.weight)
        nn.init.zeros_(self.orig_proj.bias)

    # ------------------------------------------------------------------
    # Core override: swap language features for original-image tokens
    # ------------------------------------------------------------------

    def _encode_prompt(
        self,
        backbone_out: Dict,
        find_input: FindStage,
        geometric_prompt: Prompt,
        visual_prompt_embed=None,
        visual_prompt_mask=None,
        encode_text: bool = True,    # ignored — we always use image tokens
        prev_mask_pred=None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """
        Replace language_features with per-sample original-image tokens.

        find_input.img_ids contains indices of the *edited* images in backbone_fpn.
        The original image for query i is always at img_ids[i] + 1 (see batching
        convention in module docstring).
        """
        feat_tuple = self._get_img_feats(backbone_out, find_input.img_ids)
        backbone_out, img_feats, img_pos_embeds, vis_feat_sizes = feat_tuple

        if prev_mask_pred is not None:
            img_feats = [img_feats[-1] + prev_mask_pred]

        # --- original image tokens as prompt ---
        orig_img_ids = find_input.img_ids + 1          # [B]
        # backbone_fpn[-1]: [total_imgs, 256, H, W]
        orig_feats = backbone_out["backbone_fpn"][-1][orig_img_ids]   # [B, 256, H, W]
        orig_tokens = orig_feats.flatten(2).permute(2, 0, 1)          # [H*W, B, 256]
        orig_tokens = self.orig_proj(orig_tokens)                      # [H*W, B, 256]
        # Add 2D positional encoding so the fusion encoder knows spatial layout.
        # The encoder doesn't pass pos enc for prompt/memory tokens (designed for
        # text which has no spatial structure), so we bake it in here.
        orig_pos = backbone_out["vision_pos_enc"][-1]                  # [total_imgs, 256, H, W]
        orig_pos = orig_pos[orig_img_ids].flatten(2).permute(2, 0, 1)  # [H*W, B, 256]
        orig_tokens = orig_tokens + orig_pos
        # mask: all False (no padding in a dense spatial grid)
        orig_mask = torch.zeros(
            orig_tokens.shape[1], orig_tokens.shape[0],
            dtype=torch.bool, device=orig_tokens.device,
        )                                                              # [B, H*W]

        # --- geometric tokens (boxes/points from find_input) ---
        geo_feats, geo_masks = self.geometry_encoder(
            geo_prompt=geometric_prompt,
            img_feats=img_feats,
            img_sizes=vis_feat_sizes,
            img_pos_embeds=img_pos_embeds,
        )

        # --- optional visual prompt passthrough (keep compatible with base class) ---
        if visual_prompt_embed is None:
            visual_prompt_embed = torch.zeros(
                (0, *geo_feats.shape[1:]), device=geo_feats.device
            )
            visual_prompt_mask = torch.zeros(
                (*geo_masks.shape[:-1], 0),
                device=geo_masks.device,
                dtype=geo_masks.dtype,
            )

        prompt = torch.cat([orig_tokens, geo_feats, visual_prompt_embed], dim=0)
        prompt_mask = torch.cat([orig_mask, geo_masks, visual_prompt_mask], dim=1)
        return prompt, prompt_mask, backbone_out

    # ------------------------------------------------------------------
    # Training forward — skips text encoding, handles 2-img-per-sample batch
    # ------------------------------------------------------------------

    def forward(self, input: BatchedDatapoint) -> SAM3Output:
        """
        input.img_batch: [2*B, 3, H, W]  (edited and original images interleaved)
        input.find_inputs[0].img_ids: [0, 2, 4, ..., 2*(B-1)]  (edited)
        """
        device = self.device

        # Encode ALL images (edited + original) in one backbone pass.
        # Backbone is frozen; torch.no_grad() avoids storing intermediate activations.
        backbone_out: Dict = {"img_batch_all_stages": input.img_batch}
        with torch.no_grad():
            backbone_out.update(self.backbone.forward_image(input.img_batch))

        # NOTE: no backbone.forward_text() call — original image tokens are the prompt.

        assert len(input.find_inputs) == 1, (
            "SAM3ChangeDetector expects exactly one find stage per forward pass."
        )
        find_input = input.find_inputs[0]
        find_target = input.find_targets[0]

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

        # The stock Sam3Image only computes matcher indices while training.
        # Our validation loop still uses Sam3LossWrapper, which expects those
        # indices whenever a target is present.
        if find_target is not None and "indices" not in out:
            self._compute_matching(out, self.back_convert(find_target))

        previous_stages_out = SAM3Output(
            iter_mode=SAM3Output.IterMode.LAST_STEP_PER_STAGE
        )
        previous_stages_out.append([out])
        return previous_stages_out


# ---------------------------------------------------------------------------
# Phase-0 inference helper (no training required)
# ---------------------------------------------------------------------------

def inject_original_as_prompt(
    processor,
    original_state: Dict,
    edited_state: Dict,
) -> Dict:
    """
    Inject original image tokens into edited_state as language_features so that
    Sam3Processor._forward_grounding() can be used unchanged for inference.

    Usage:
        model = build_sam3_image_model()
        processor = Sam3Processor(model)

        orig_state  = processor.set_image(original_pil)
        edit_state  = processor.set_image(edited_pil)

        out_state = inject_original_as_prompt(processor, orig_state, edit_state)
        masks, boxes, scores = out_state["masks"], out_state["boxes"], out_state["scores"]
    """
    orig_fpn = original_state["backbone_out"]["backbone_fpn"][-1]  # [1, 256, H, W]
    orig_tokens = orig_fpn.flatten(2).permute(2, 0, 1)             # [H*W, 1, 256]
    orig_mask = torch.zeros(
        1, orig_tokens.shape[0], dtype=torch.bool, device=orig_tokens.device
    )  # [1, H*W]

    edited_state["backbone_out"]["language_features"] = orig_tokens
    edited_state["backbone_out"]["language_mask"] = orig_mask
    edited_state["backbone_out"]["language_embeds"] = torch.zeros(
        1, 1, orig_tokens.shape[-1], device=orig_tokens.device
    )

    if "geometric_prompt" not in edited_state:
        edited_state["geometric_prompt"] = processor.model._get_dummy_prompt()

    return processor._forward_grounding(edited_state)
