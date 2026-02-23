import json
import os
import random
from typing import Dict

import dotenv
import numpy as np
import PIL.Image
import torch
from einops import rearrange
from torch import nn
from torch.utils.data import Dataset
from torchvision.transforms import InterpolationMode
import torchvision.transforms.functional as TF
from transformers import AutoImageProcessor

dotenv.load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))


class Perturbations:
    """Tensor-level augmentations shared with the UNet data pipeline."""

    MASK_KEYS = frozenset(("mask", "sub_mask", "union_mask"))

    def __init__(
        self,
        crop_scale=(0.85, 1),
        flip_prob=0.5,
        rotation_deg=12.0,
        color_jitter=(0.1, 0.1, 0.05),
        noise_std=0.02,
    ):
        self.crop_scale = crop_scale
        self.flip_prob = flip_prob
        self.rotation_deg = rotation_deg
        self.brightness_delta, self.contrast_delta, self.saturation_delta = color_jitter
        self.noise_std = noise_std

    def augment(self, sample: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        sample = self._random_crop(sample)
        sample = self._horizontal_flip(sample)
        sample = self._rotate(sample)
        sample = self._color_jitter(sample)
        sample = self._noise(sample)
        return sample

    def _random_crop(self, sample):
        ref = sample["original"]
        _, H, W = ref.shape
        scale = random.uniform(*self.crop_scale)
        if scale >= 0.999:
            return sample
        new_h = max(1, int(H * scale))
        new_w = max(1, int(W * scale))
        top = random.randint(0, H - new_h)
        left = random.randint(0, W - new_w)
        for key in sample:
            crop = sample[key][:, top : top + new_h, left : left + new_w]
            mode = "nearest" if key in self.MASK_KEYS else "bilinear"
            kw = {"align_corners": False} if mode == "bilinear" else {}
            sample[key] = nn.functional.interpolate(
                crop.unsqueeze(0), size=(H, W), mode=mode, **kw
            ).squeeze(0)
        return sample

    def _horizontal_flip(self, sample):
        if random.random() >= self.flip_prob:
            return sample
        for key in sample:
            sample[key] = sample[key].flip(-1)
        return sample

    def _rotate(self, sample):
        if self.rotation_deg <= 0:
            return sample

        angle = random.uniform(-self.rotation_deg, self.rotation_deg)
        if abs(angle) < 1e-2:
            return sample

        import math

        rad = math.radians(abs(angle))
        safe_scale = (1.0 / (math.cos(rad) + math.sin(rad))) * 0.95

        for key in sample:
            fill = [0.0] if key in self.MASK_KEYS else [-1.0]
            interp = (
                InterpolationMode.NEAREST
                if key in self.MASK_KEYS
                else InterpolationMode.BILINEAR
            )
            img = TF.rotate(sample[key], angle, interpolation=interp, fill=fill)
            _, H, W = img.shape
            new_h, new_w = int(H * safe_scale), int(W * safe_scale)
            img = TF.center_crop(img, [new_h, new_w])
            mode = "nearest" if key in self.MASK_KEYS else "bilinear"
            kw = {"align_corners": False} if mode == "bilinear" else {}
            sample[key] = nn.functional.interpolate(
                img.unsqueeze(0), size=(H, W), mode=mode, **kw
            ).squeeze(0)
        return sample

    def _color_jitter(self, sample):
        for key in ("original", "edited"):
            img = sample[key]
            if self.brightness_delta:
                factor = random.uniform(
                    1 - self.brightness_delta, 1 + self.brightness_delta
                )
                img = img * factor + (factor - 1)
            if self.contrast_delta:
                factor = random.uniform(
                    1 - self.contrast_delta, 1 + self.contrast_delta
                )
                mean = img.mean()
                img = mean + (img - mean) * factor
            if self.saturation_delta:
                factor = random.uniform(
                    1 - self.saturation_delta, 1 + self.saturation_delta
                )
                gray = img[0:1] * 0.299 + img[1:2] * 0.587 + img[2:3] * 0.114
                img = gray + (img - gray) * factor
            sample[key] = img.clamp(-1, 1)
        return sample

    def _noise(self, sample):
        if self.noise_std <= 0:
            return sample
        for key in ("original", "edited"):
            sample[key] = (
                sample[key] + torch.randn_like(sample[key]) * self.noise_std
            ).clamp(-1, 1)
        return sample


def _pil_to_tensor(img: PIL.Image.Image, is_mask: bool) -> torch.Tensor:
    arr = np.array(img, dtype=np.float32)
    if arr.ndim == 2:
        arr = arr[..., None]
    tensor = torch.from_numpy(arr)
    tensor = rearrange(tensor, "H W C -> C H W")
    if is_mask:
        return tensor[:1] / 255.0
    return tensor / 127.5 - 1.0


def _tensor_to_pil(tensor: torch.Tensor, is_mask: bool) -> PIL.Image.Image:
    tensor = tensor.detach().cpu()
    if is_mask:
        array = (tensor[0] * 255.0).clamp(0, 255).byte().numpy()
        return PIL.Image.fromarray(array, mode="L")
    array = ((tensor + 1.0) * 127.5).clamp(0, 255).byte()
    array = rearrange(array, "C H W -> H W C").numpy()
    return PIL.Image.fromarray(array)


class DinoDataset(Dataset):
    """Dataset that mirrors UNet preprocessing without producing delta maps."""

    def __init__(
        self,
        hw=256,
        device="cpu",
        path="/content/",
        n=None,
        skip_zero_edit=True,
        processor_name: str = "facebook/dinov3-vits16-pretrain-lvd1689m",
    ):
        self.processor = AutoImageProcessor.from_pretrained(processor_name)
        self.perturb = Perturbations()
        self.hw = hw
        self.device = device
        folder = "data_sample/success/"
        self.data = []
        scans = 0
        fails = 0
        for file in os.listdir(path + folder):
            scans += 1
            if n is not None and n == len(self.data):
                break
            try:
                meta = json.load(open(path + folder + file + "/meta.json"))
                if meta["similarity_score"] < 0.94:
                    continue
                if skip_zero_edit and len(meta["subtraction"]["success"]) == 0 and len(
                    meta["union"]["success"]
                ) == 0:
                    continue
                self.data.append(
                    {
                        "original": path + folder + file + "/base.jpeg"
                        if meta["base"] == "original"
                        else path + folder + file + "/other.jpeg",
                        "edited": path + folder + file + "/base.jpeg"
                        if meta["base"] == "edited"
                        else path + folder + file + "/other.jpeg",
                        "mask": path + folder + file + "/mask.png",
                        "sub_mask": path + folder + file + "/subtraction_mask.png",
                        "union_mask": path + folder + file + "/union_mask.png",
                    }
                )
            except Exception:
                fails += 1
                continue
        print(f"Tried {scans}, Failed {fails}")

    def __len__(self):
        return len(self.data)

    def _encode_inputs(self, tensor: torch.Tensor) -> Dict[str, torch.Tensor]:
        pil_img = _tensor_to_pil(tensor, is_mask=False)
        encoded = self.processor(images=pil_img, return_tensors="pt")
        return {k: v.squeeze(0) for k, v in encoded.items()}

    def __getitem__(self, idx):
        paths = self.data[idx]
        raw_images = {key: PIL.Image.open(paths[key]) for key in paths}

        tensors = {}
        for key, img in raw_images.items():
            tensors[key] = _pil_to_tensor(img, key in Perturbations.MASK_KEYS)

        if self.perturb is not None:
            tensors = self.perturb.augment(tensors)

        original_inputs = self._encode_inputs(tensors["original"])
        edited_inputs = self._encode_inputs(tensors["edited"])

        return {
            "original": tensors["original"],
            "edited": tensors["edited"],
            "mask": tensors["mask"],
            "sub_mask": tensors["sub_mask"],
            "union_mask": tensors["union_mask"],
            "original_inputs": original_inputs,
            "edited_inputs": edited_inputs,
        }
