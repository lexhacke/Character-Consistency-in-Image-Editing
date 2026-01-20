import cv2
import numpy as np
import PIL
import numpy as np
from torch.nn import functional as F
import torch
import matplotlib.pyplot as plt
from einops import rearrange

def expand_mask(mask, hw=25):
    mask = mask[:, :, :1]
    kernel = torch.ones((1, 1, hw, hw))
    mask = rearrange(mask[None, :, :, :], "B H W C -> B C H W")
    mask = F.conv2d(torch.from_numpy(mask).float(), kernel.float(), padding=hw//2, stride=1).numpy()[0]
    mask = rearrange(mask, "C H W -> H W C")
    mask = mask.clip(0, 1)
    return np.tile(mask, (1, 1, 3))

def blend(mask, base, other, mode=None):
    if mode is None:
        inv_mask = 1 - mask
        overlay = mask * other
        underlay = inv_mask * base
        composite = overlay + underlay
    elif mode == "laplacian":
        levels = 4
        # Ensure float32 for precision
        base = base.astype(np.float32)
        other = other.astype(np.float32)
        mask = mask.astype(np.float32)

        # 1. Generate Gaussian Pyramids
        gauss_pyramid_base = [base]
        gauss_pyramid_other = [other]
        gauss_pyramid_mask = [mask]

        for i in range(levels):
            gauss_pyramid_base.append(cv2.pyrDown(gauss_pyramid_base[-1]))
            gauss_pyramid_other.append(cv2.pyrDown(gauss_pyramid_other[-1]))
            gauss_pyramid_mask.append(cv2.pyrDown(gauss_pyramid_mask[-1]))

        # 2. Reconstruct using Laplacian Blending logic
        # Start with the coarsest level (top of Gaussian pyramid)
        
        # Fix dimensions for mask if lost during pyrDown
        mask_top = gauss_pyramid_mask[-1]
        if len(mask_top.shape) == 2:
            mask_top = mask_top[..., np.newaxis]

        # Blend the top-most images directly
        composite = gauss_pyramid_other[-1] * mask_top + \
                    gauss_pyramid_base[-1] * (1.0 - mask_top)

        # Reconstruct going up the pyramid
        for i in range(levels, 0, -1):
            # Upsample current composite to match the next level down
            composite = cv2.pyrUp(composite)

            # Resize to match exact dimensions of the next level (handle odd sizes)
            target_h, target_w = gauss_pyramid_base[i-1].shape[:2]
            composite = cv2.resize(composite, (target_w, target_h))

            # Calculate Laplacians for this level
            # Base Laplacian
            tmp_base = cv2.pyrUp(gauss_pyramid_base[i])
            tmp_base = cv2.resize(tmp_base, (target_w, target_h))
            L_base = cv2.subtract(gauss_pyramid_base[i-1], tmp_base)

            # Other Laplacian
            tmp_other = cv2.pyrUp(gauss_pyramid_other[i])
            tmp_other = cv2.resize(tmp_other, (target_w, target_h))
            L_other = cv2.subtract(gauss_pyramid_other[i-1], tmp_other)

            # Blend Laplacians using the mask at this level
            mask_level = gauss_pyramid_mask[i-1]
            if len(mask_level.shape) == 2:
                mask_level = mask_level[..., np.newaxis]
            
            L_composite = L_other * mask_level + L_base * (1.0 - mask_level)

            # Add blended detail to the upsampled composite
            composite = cv2.add(composite, L_composite)

        composite = np.clip(composite, 0, 1)
    else:
        raise NotImplementedError(f"Unrecognised blending mode {mode}")
    return composite