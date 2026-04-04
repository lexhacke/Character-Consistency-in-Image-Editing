"""
SAM3ChangeDetectionDataset — bridges clean_data/ directory structure to SAM3's Datapoint format.

Directory layout expected (from the data collection pipeline):
    clean_data/{i}/
        base.jpeg       — original image when meta["base"] == "original", edited otherwise
        other.jpeg      — the counterpart
        mask.png        — binary ground-truth change mask
        meta.json       — {"base": "original"|"edited", "similarity_score": float, ...}

Datapoint structure produced:
    images[0]  edited image tensor  [3, R, R],  objects=[Object(bbox, segment, ...)]
    images[1]  original image tensor [3, R, R], objects=[]
    find_queries[0]  image_id=0, object_ids_output=[0]

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
        for d in all_dirs:
            meta_path = d / "meta.json"
            if not (meta_path.exists() and (d / "base.jpeg").exists()
                    and (d / "other.jpeg").exists() and (d / "mask.png").exists()):
                continue
            with open(meta_path) as f:
                meta = json.load(f)
            if meta.get("similarity_score", 1.0) < min_sim:
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
        mask_pil = PILImage.open(sample_dir / "mask.png").convert("L")

        # Determine which image is the original (unedited) vs edited
        if meta["base"] == "original":
            original_pil, edited_pil = base_pil, other_pil
        else:
            original_pil, edited_pil = other_pil, base_pil

        # Preprocess images to [3, R, R] float32 in [-1, 1]
        edited_tensor = self._img_transform(edited_pil)    # [3, R, R]
        original_tensor = self._img_transform(original_pil)  # [3, R, R]

        # Preprocess mask to [R, R] binary bool
        mask_tensor = self._preprocess_mask(mask_pil)  # [R, R] bool

        # Compute bounding box in normalised CxCyWH (what the model expects after
        # NormalizeAPI; we bypass that transform, so we provide it directly)
        bbox = self._mask_to_bbox_cxcywh(mask_tensor)  # [4]

        area = float(mask_tensor.sum().item())

        original_h, original_w = edited_pil.size[1], edited_pil.size[0]

        return Datapoint(
            images=[
                Image(
                    data=edited_tensor,
                    objects=[
                        Object(
                            bbox=bbox,
                            area=area,
                            object_id=0,
                            frame_index=0,
                            segment=mask_tensor,
                            is_crowd=False,
                        )
                    ],
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
                    query_text="",          # unused — wrapper uses image tokens
                    image_id=0,             # edited image is image 0
                    object_ids_output=[0],  # the changed region
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
