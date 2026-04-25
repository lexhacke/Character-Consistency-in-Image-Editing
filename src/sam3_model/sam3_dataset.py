"""
SAM3ChangeDetectionDataset — bridges clean_data/ directory structure to SAM3's Datapoint format.

Directory layout expected (from the data collection pipeline):
    clean_data/{i}/
        base.jpeg              — original image when meta["base"] == "original", edited otherwise
        other.jpeg             — the counterpart
        sub_0.png, sub_1.png, ...       — individual SAM3 masks per subtraction object
        union_0.png, union_1.png, ...   — individual SAM3 masks per union object
        subtraction_mask.png   — merged OR of all subtraction masks
        union_mask.png         — merged OR of all union masks
        meta.json              — see below

    meta.json format (new):
        "subtraction": {"success": [{"prompt": "...", "masks": ["sub_0.png", ...]}, ...]}
        "union":       {"success": [{"prompt": "...", "masks": ["union_0.png", ...]}, ...]}

    Each object-prompt entry becomes a separate Object for the DETR decoder.
    Its ground-truth mask is the OR of all individual SAM3 masks for that entry.
    This yields N objects per sample (N = #subtraction_objects + #union_objects).

    Falls back to old format (success = flat list of strings) by loading the
    merged subtraction_mask.png / union_mask.png as single objects.

Datapoint structure produced:
    images[0]  edited image tensor  [3, R, R],  objects=[Object(...), ...]  (0-N objects)
    images[1]  original image tensor [3, R, R], objects=[]
    find_queries[0]  image_id=0, object_ids_output=[0, 1, ..., N-1] or []

Batching convention (must match SAM3ChangeDetector._encode_prompt):
    After collation img_batch = [edited_0, orig_0, edited_1, orig_1, ...].
    find_input.img_ids = [0, 2, 4, ...] so orig = img_ids + 1.
"""

import json
import random
from pathlib import Path
from typing import List, Optional

import numpy as np
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
        merge_masks: bool = True,
    ):
        self.resolution = resolution
        self.split = split
        self.merge_masks = merge_masks

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
            sub_entries = meta.get("subtraction", {}).get("success", [])
            union_entries = meta.get("union", {}).get("success", [])
            if not sub_entries and not union_entries:
                skipped_no_masks += 1
                continue
            # Verify mask files exist (new format: individual files; old format: merged files)
            has_loadable_mask = False
            for entry in sub_entries:
                if isinstance(entry, dict) and "masks" in entry:
                    if any((d / mf).exists() for mf in entry["masks"]):
                        has_loadable_mask = True
                        break
                elif (d / "subtraction_mask.png").exists():
                    has_loadable_mask = True
                    break
            if not has_loadable_mask:
                for entry in union_entries:
                    if isinstance(entry, dict) and "masks" in entry:
                        if any((d / mf).exists() for mf in entry["masks"]):
                            has_loadable_mask = True
                            break
                    elif (d / "union_mask.png").exists():
                        has_loadable_mask = True
                        break
            if not has_loadable_mask:
                skipped_no_masks += 1
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
        return self._load_sample(idx)

    def _load_sample(self, idx: int) -> Datapoint:
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

        # Build object list from meta.json mask entries
        sub_entries = meta.get("subtraction", {}).get("success", [])
        union_entries = meta.get("union", {}).get("success", [])

        objects = []
        object_ids = []
        obj_id = 0

        if self.merge_masks:
            # Merge mode: OR all subtraction into one Object, all union into one Object (max 2)
            sub_mask = self._load_and_merge_entries(sample_dir, sub_entries, "subtraction_mask.png")
            if sub_mask is not None:
                objects.append(Object(
                    bbox=self._mask_to_bbox_cxcywh(sub_mask),
                    area=float(sub_mask.sum().item()),
                    object_id=obj_id, frame_index=0, segment=sub_mask, is_crowd=False,
                ))
                object_ids.append(obj_id)
                obj_id += 1

            union_mask = self._load_and_merge_entries(sample_dir, union_entries, "union_mask.png")
            if union_mask is not None:
                objects.append(Object(
                    bbox=self._mask_to_bbox_cxcywh(union_mask),
                    area=float(union_mask.sum().item()),
                    object_id=obj_id, frame_index=0, segment=union_mask, is_crowd=False,
                ))
                object_ids.append(obj_id)
                obj_id += 1
        else:
            # Per-object mode: one Object per object-prompt entry
            for entry in sub_entries:
                mask = self._load_object_mask(sample_dir, entry, fallback="subtraction_mask.png")
                if mask is None:
                    continue
                objects.append(Object(
                    bbox=self._mask_to_bbox_cxcywh(mask),
                    area=float(mask.sum().item()),
                    object_id=obj_id, frame_index=0, segment=mask, is_crowd=False,
                ))
                object_ids.append(obj_id)
                obj_id += 1

            for entry in union_entries:
                mask = self._load_object_mask(sample_dir, entry, fallback="union_mask.png")
                if mask is None:
                    continue
                objects.append(Object(
                    bbox=self._mask_to_bbox_cxcywh(mask),
                    area=float(mask.sum().item()),
                    object_id=obj_id, frame_index=0, segment=mask, is_crowd=False,
                ))
                object_ids.append(obj_id)
                obj_id += 1

        # Safety: if all mask loads failed, try a different sample
        if not objects:
            alt = (idx + 1) % len(self.samples)
            if alt != idx:
                return self._load_sample(alt)

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

    def _load_and_merge_entries(self, sample_dir: Path, entries: list, fallback: str) -> Optional[torch.Tensor]:
        """OR all entries' masks into a single mask. Returns None if nothing loaded."""
        combined = None
        for entry in entries:
            m = self._load_object_mask(sample_dir, entry, fallback)
            if m is None:
                continue
            combined = m if combined is None else (combined | m)
        return combined

    def _load_object_mask(self, sample_dir: Path, entry, fallback: str) -> Optional[torch.Tensor]:
        """
        Load mask for one object-prompt entry.

        New format: entry is {"prompt": str, "masks": ["sub_0.png", ...]}
            → OR all individual mask files together.
        Old format: entry is a plain string (prompt description)
            → fall back to merged mask file (subtraction_mask.png / union_mask.png).
        """
        if isinstance(entry, dict) and "masks" in entry:
            # New format: OR individual mask files
            combined = None
            for mask_file in entry["masks"]:
                mask_path = sample_dir / mask_file
                if not mask_path.exists():
                    continue
                m = self._preprocess_mask(PILImage.open(mask_path).convert("L"))
                combined = m if combined is None else (combined | m)
            return combined
        else:
            # Old format: load merged mask
            fallback_path = sample_dir / fallback
            if not fallback_path.exists():
                return None
            return self._preprocess_mask(PILImage.open(fallback_path).convert("L"))

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
