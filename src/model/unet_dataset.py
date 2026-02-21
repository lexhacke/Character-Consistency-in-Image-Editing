import os, dotenv, json, torch, random
import numpy as np
import PIL.Image
from torch import nn
from einops import rearrange
from torch.utils.data import DataLoader, Dataset
from attention import DinoMap
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode

dotenv.load_dotenv(os.path.join(os.path.dirname(__file__), '..', '..', '.env'))

class Perturbations:
    """
    Tensor-based augmentations on preprocessed C H W tensors.
    Geometric transforms applied consistently to all keys.
    Photometric transforms applied only to original/edited.
    Images are in [-1,1], mask is in [0,1].
    """
    MASK_KEYS = frozenset(('mask',))

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

    def augment(self, sample):
        sample = self._random_crop(sample)
        sample = self._horizontal_flip(sample)
        sample = self._rotate(sample)
        sample = self._color_jitter(sample)
        sample = self._noise(sample)
        return sample

    def _random_crop(self, sample):
        ref = sample['original']
        _, H, W = ref.shape
        scale = random.uniform(*self.crop_scale)
        if scale >= 0.999:
            return sample
        new_h = max(1, int(H * scale))
        new_w = max(1, int(W * scale))
        top = random.randint(0, H - new_h)
        left = random.randint(0, W - new_w)
        for key in sample:
            crop = sample[key][:, top:top+new_h, left:left+new_w]
            mode = 'nearest' if key in self.MASK_KEYS else 'bilinear'
            kw = {'align_corners': False} if mode == 'bilinear' else {}
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
        
        # 1. Base Math: s = 1 / (cos|a| + sin|a|)
        # 2. Add Buffer: Multiply by 0.98 to zoom in an extra 2%
        safe_scale = (1.0 / (math.cos(rad) + math.sin(rad))) * 0.95

        for key in sample:
            # Get original fill values based on your normalization
            fill = [0.0] if key in self.MASK_KEYS else [-1.0]
            interp = InterpolationMode.NEAREST if key in self.MASK_KEYS else InterpolationMode.BILINEAR
            
            # Apply Rotation
            img = TF.rotate(sample[key], angle, interpolation=interp, fill=fill)
            
            # Center Crop using the buffered safe_scale
            _, H, W = img.shape
            new_h, new_w = int(H * safe_scale), int(W * safe_scale)
            img = TF.center_crop(img, [new_h, new_w])
            
            # Resize back to original H, W
            mode = 'nearest' if key in self.MASK_KEYS else 'bilinear'
            kw = {'align_corners': False} if mode == 'bilinear' else {}
            sample[key] = nn.functional.interpolate(
                img.unsqueeze(0), size=(H, W), mode=mode, **kw
            ).squeeze(0)
            
        return sample
    
    def _color_jitter(self, sample):
        for key in ('original', 'edited'):
            img = sample[key]
            if self.brightness_delta:
                factor = random.uniform(1 - self.brightness_delta, 1 + self.brightness_delta)
                img = img * factor + (factor - 1)
            if self.contrast_delta:
                factor = random.uniform(1 - self.contrast_delta, 1 + self.contrast_delta)
                mean = img.mean()
                img = mean + (img - mean) * factor
            if self.saturation_delta:
                factor = random.uniform(1 - self.saturation_delta, 1 + self.saturation_delta)
                gray = img[0:1] * 0.299 + img[1:2] * 0.587 + img[2:3] * 0.114
                img = gray + (img - gray) * factor
            sample[key] = img.clamp(-1, 1)
        return sample

    def _noise(self, sample):
        if self.noise_std <= 0:
            return sample
        for key in ('original', 'edited'):
            sample[key] = (sample[key] + torch.randn_like(sample[key]) * self.noise_std).clamp(-1, 1)
        return sample
    
class unet_processor:
    def __init__(self, hw, device):
        self.hw = hw
        self.device = device
        self.dino_map = DinoMap(device)

    def dino_delta(self, original, edited):
        """
        Expects 2 PIL.Image
        """
        return self.preprocess_img(self.dino_map(original, edited).unsqueeze(0), False)

    def preprocess_img(self, img, is_mask):
        """Resize + normalize. No padding (call pad_to_square separately)."""
        if isinstance(img, PIL.Image.Image):
            img = np.array(img, dtype=np.float32)
            img = torch.from_numpy(img)
            if img.ndim == 2:
                img = img.unsqueeze(-1)  # H W -> H W 1
            img = rearrange(img, 'H W C -> C H W')

        C, H, W = img.shape
        ratio = self.hw / max(H, W)
        img = nn.functional.interpolate(img.unsqueeze(0), size=(int(H*ratio+0.5), int(W*ratio+0.5)), mode='bilinear', align_corners=False)[0]

        if is_mask:
            return img[:1] / 255
        else:
            return img / 127.5 - 1

    def pad_to_square(self, img, is_mask):
        """Pad to self.hw x self.hw with appropriate fill value."""
        C, H, W = img.shape
        pad_val = 0.0 if is_mask else -1.0
        if H < self.hw:
            img = torch.cat([img, torch.full((C, self.hw - H, W), pad_val)], dim=1)
        elif W < self.hw:
            img = torch.cat([img, torch.full((C, H, self.hw - W), pad_val)], dim=2)
        return img

class UNetDataset(Dataset):
    def __init__(self, hw=512, device='cpu', path="/content/", mode="dino", n=None, skip_zero_edit=True):
        if mode == 'L1':
            self.delta_mode = mode
        elif mode == 'dino':
            self.delta_mode = mode
        else:
            raise NotImplementedError
        self.processor = unet_processor(hw, device=device)
        self.perturb = Perturbations()
        self.hw = hw
        folder = "data_sample/success/"
        self.data = []
        scans = 0
        fails = 0
        for file in os.listdir(path+folder):
            scans += 1
            if n is not None:
                if n == (scans - fails):
                    break
            try:
                meta = json.load(open(path+folder+file+'/meta.json'))
                if meta['similarity_score'] < 0.94 or (skip_zero_edit and len(meta['subtraction']['success']) == 0 and len(meta['union']['success'] == 0)):
                    continue
                self.data.append({
                    'original':path+folder+file+'/base.jpeg' if meta['base'] == "original" else path+folder+file+'/other.jpeg',
                    'edited':path+folder+file+'/base.jpeg' if meta['base'] == "edited" else path+folder+file+'/other.jpeg',
                    'mask':path+folder+file+'/mask.png',
                    'sub_mask':path+folder+file+'/subtraction_mask.png',
                    'union_mask':path+folder+file+'/union_mask.png'
                })

            except Exception as e:
              print(e)
              fails += 1
              continue
        print(f"Tried {scans}, Failed {fails}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        paths = self.data[idx]
        datapoint = {key: PIL.Image.open(paths[key]) for key in ('original', 'edited', 'mask', 'sub_mask', 'union_mask')}
        # Dino delta needs PIL images — compute before tensor conversion
        if self.delta_mode == "dino":
            datapoint['delta'] = self.processor.dino_delta(datapoint['edited'], datapoint['original'])
        # Resize + normalize (no padding yet)
        for key in ('original', 'edited', 'mask', 'sub_mask', 'union_mask'):
            datapoint[key] = self.processor.preprocess_img(datapoint[key], key == 'mask')
        if self.delta_mode == 'L1':
            datapoint['delta'] = torch.abs(datapoint['original'] - datapoint['edited'])
        # Augment on unpadded tensors
        if self.perturb is not None:
            datapoint = self.perturb.augment(datapoint)
        # Pad to square AFTER augmentation
        for key in datapoint:
            datapoint[key] = self.processor.pad_to_square(datapoint[key], key == 'mask')
        return datapoint

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    dataset = UNetDataset(hw=512, path=r"C:\Users\lex\Documents\Ai2\data_sample_backup" + os.sep)
    print(f"Dataset size: {len(dataset)}")
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    for batch in dataloader:
        print(batch['original'].shape)
        for img in batch['original']:
            plt.imshow(rearrange(img, "C H W -> H W C"))
            plt.show()

        print(batch['edited'].shape)
        for img in batch['edited']:
            plt.imshow(rearrange(img, "C H W -> H W C"))
            plt.show()

        print(batch['mask'].shape)
        for img in batch['mask']:
            plt.imshow(rearrange(img, "C H W -> H W C"))
            plt.show()
        break