"""
SAM3ChangeDetectionDataset — bridges clean_data/ directory structure to SAM3's Datapoint format.

Directory layout expected (from the data collection pipeline):
    clean_data/{i}/
        base.jpeg              — original image when meta["base"] == "original", edited otherwise
        other.jpeg             — the counterpart
        subtraction_mask.png   — mask of regions removed from original (may be empty)
        union_mask.png         — mask of regions added in the edit (may be empty)
        meta.json              — {"base", "subtraction": {"success": [...]}, "union": {"success": [...]}, ...}

    Which masks are valid is determined by meta.json:
        len(meta["subtraction"]["success"]) > 0  →  subtraction_mask.png is a real mask
        len(meta["union"]["success"]) > 0         →  union_mask.png is a real mask

    This yields 0, 1, or 2 Object entries per sample. SAM3's Hungarian matcher
    and DETR loss handle all three cases natively:
        0 objects  →  presence_logit trains to 0, per-query losses zeroed
        1 object   →  1 query matched, 199 unmatched
        2 objects  →  2 queries matched, 198 unmatched

Datapoint structure produced:
    images[0]  edited image tensor  [3, R, R],  objects=[Object(...), ...]  (0-2 objects)
    images[1]  original image tensor [3, R, R], objects=[]
    find_queries[0]  image_id=0, object_ids_output=[0], [0,1], or []

Batching convention (must match SAM3ChangeDetector._encode_prompt):
    After collation img_batch = [edited_0, orig_0, edited_1, orig_1, ...].
    find_input.img_ids = [0, 2, 4, ...] so orig = img_ids + 1.
"""

import json
import random
from pathlib import Path
from typing import List, Optional

import torch
import torch.utils.data
import torchvision.transforms.v2 as T
import torchvision.transforms.v2.functional as F
from PIL import Image as PILImage
from sam3.train.data.sam3_image_dataset import (
    Datapoint,
    FindQueryLoaded,
    Image,
    InferenceMetadata,
    Object,
)


class SAM3ChangeDetectionDataset(torch.utils.data.Dataset):
    """
    Args:
        data_root:   path to clean_data/ (or any directory with {i}/ sub-dirs)
        resolution:  target square resolution fed to the backbone (default 1008)
        min_sim:     drop samples below this DINO similarity score (default 0.94)
        split:       "train" or "val"
        val_fraction: fraction of samples held out for validation (default 0.05)
        seed:        random seed for train/val split (default 42)
    """

    # Same normalisation as Sam3Processor and SAM3's NormalizeAPI
    _NORM_MEAN = (0.5, 0.5, 0.5)
    _NORM_STD = (0.5, 0.5, 0.5)

    def __init__(
        self,
        data_root: str,
        resolution: int = 1008,
        min_sim: float = 0.94,
        split: str = "train",
        val_fraction: float = 0.05,
        seed: int = 42,
    ):
        self.resolution = resolution
        self.split = split

        self._img_transform = T.Compose([
            T.ToImage(),
            T.Resize((resolution, resolution), interpolation=T.InterpolationMode.BILINEAR, antialias=True),
            T.ToDtype(torch.float32, scale=True),
            T.Normalize(mean=list(self._NORM_MEAN), std=list(self._NORM_STD)),
        ])

        all_dirs = sorted(
            [p for p in Path(data_root).iterdir() if p.is_dir()],
            key=lambda p: p.name,
        )

        valid: List[Path] = []
        skipped_no_masks = 0
        for d in all_dirs:
            meta_path = d / "meta.json"
            if not (meta_path.exists() and (d / "base.jpeg").exists()
                    and (d / "other.jpeg").exists()):
                continue
            with open(meta_path) as f:
                meta = json.load(f)
            if meta.get("similarity_score", 1.0) < min_sim:
                continue
            # Determine which masks are valid from meta.json
            has_sub = len(meta.get("subtraction", {}).get("success", [])) > 0
            has_union = len(meta.get("union", {}).get("success", [])) > 0
            if not has_sub and not has_union:
                skipped_no_masks += 1
                continue
            # Verify the mask files actually exist
            if has_sub and not (d / "subtraction_mask.png").exists():
                continue
            if has_union and not (d / "union_mask.png").exists():
                continue
            valid.append(d)

        rng = random.Random(seed)
        indices = list(range(len(valid)))
        rng.shuffle(indices)
        n_val = max(1, int(len(valid) * val_fraction))
        val_idx = set(indices[:n_val])
        if split == "val":
            self.samples = [valid[i] for i in indices[:n_val]]
        else:
            self.samples = [valid[i] for i in indices[n_val:]]

        if skipped_no_masks:
            print(f"[SAM3ChangeDetectionDataset] Skipped {skipped_no_masks} samples "
                  f"with no valid masks (both subtraction and union empty)")
        print(f"[SAM3ChangeDetectionDataset] {split}: {len(self.samples)} samples "
              f"(from {len(valid)} valid, min_sim={min_sim})")

    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Datapoint:
        sample_dir = self.samples[idx]
        with open(sample_dir / "meta.json") as f:
            meta = json.load(f)

        base_pil = PILImage.open(sample_dir / "base.jpeg").convert("RGB")
        other_pil = PILImage.open(sample_dir / "other.jpeg").convert("RGB")

        # Determine which image is the original (unedited) vs edited
        if meta["base"] == "original":
            original_pil, edited_pil = base_pil, other_pil
        else:
            original_pil, edited_pil = other_pil, base_pil

        # Preprocess images to [3, R, R] float32 in [-1, 1]
        edited_tensor = self._img_transform(edited_pil)    # [3, R, R]
        original_tensor = self._img_transform(original_pil)  # [3, R, R]

        # Build object list from whichever masks are valid
        has_sub = len(meta.get("subtraction", {}).get("success", [])) > 0
        has_union = len(meta.get("union", {}).get("success", [])) > 0

        objects = []
        object_ids = []
        obj_id = 0

        if has_sub:
            sub_pil = PILImage.open(sample_dir / "subtraction_mask.png").convert("L")
            sub_mask = self._preprocess_mask(sub_pil)
            sub_bbox = self._mask_to_bbox_cxcywh(sub_mask)
            objects.append(Object(
                bbox=sub_bbox,
                area=float(sub_mask.sum().item()),
                object_id=obj_id,
                frame_index=0,
                segment=sub_mask,
                is_crowd=False,
            ))
            object_ids.append(obj_id)
            obj_id += 1

        if has_union:
            union_pil = PILImage.open(sample_dir / "union_mask.png").convert("L")
            union_mask = self._preprocess_mask(union_pil)
            union_bbox = self._mask_to_bbox_cxcywh(union_mask)
            objects.append(Object(
                bbox=union_bbox,
                area=float(union_mask.sum().item()),
                object_id=obj_id,
                frame_index=0,
                segment=union_mask,
                is_crowd=False,
            ))
            object_ids.append(obj_id)
            obj_id += 1

        original_h, original_w = edited_pil.size[1], edited_pil.size[0]

        return Datapoint(
            images=[
                Image(
                    data=edited_tensor,
                    objects=objects,
                    size=(self.resolution, self.resolution),
                ),
                Image(
                    data=original_tensor,
                    objects=[],
                    size=(self.resolution, self.resolution),
                ),
            ],
            find_queries=[
                FindQueryLoaded(
                    query_text=meta.get("prompt", ""),
                    image_id=0,
                    object_ids_output=object_ids,
                    is_exhaustive=True,
                    query_processing_order=0,
                    inference_metadata=InferenceMetadata(
                        coco_image_id=idx,
                        original_image_id=idx,
                        original_size=(original_h, original_w),
                        object_id=0,
                        frame_index=0,
                        original_category_id=0,
                    ),
                )
            ],
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _preprocess_mask(self, mask_pil: PILImage.Image) -> torch.Tensor:
        """Resize mask to [R, R] and binarise."""
        mask_resized = mask_pil.resize(
            (self.resolution, self.resolution), PILImage.NEAREST
        )
        mask_arr = torch.from_numpy(
            __import__("numpy").array(mask_resized, dtype="uint8")
        )  # [R, R] uint8
        return mask_arr > 127  # [R, R] bool

    @staticmethod
    def _mask_to_bbox_cxcywh(mask: torch.Tensor) -> torch.Tensor:
        """
        Convert a boolean mask [H, W] → normalised CxCyWH [4].
        Returns a zero-size centred box if the mask is empty.
        """
        H, W = mask.shape
        nonzero = mask.nonzero(as_tuple=False)  # [N, 2] (row, col)
        if nonzero.numel() == 0:
            return torch.tensor([0.5, 0.5, 0.0, 0.0], dtype=torch.float32)
        y_min = nonzero[:, 0].min().float()
        y_max = nonzero[:, 0].max().float()
        x_min = nonzero[:, 1].min().float()
        x_max = nonzero[:, 1].max().float()
        cx = (x_min + x_max) / 2.0 / W
        cy = (y_min + y_max) / 2.0 / H
        w = (x_max - x_min + 1.0) / W
        h = (y_max - y_min + 1.0) / H
        return torch.stack([cx, cy, w, h])
