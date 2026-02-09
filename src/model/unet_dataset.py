import os, dotenv, json, torch
import numpy as np
import PIL.Image
from torch import nn
from einops import rearrange
from torch.utils.data import DataLoader, Dataset
from attention import DinoMap

dotenv.load_dotenv(os.path.join(os.path.dirname(__file__), '..', '..', '.env'))

assert os.environ.get('SAVE_PATH') is not None, "Please set the SAVE_PATH in the .env file"

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
        """
        Expects img of PIL.Image, reshape via padding with zeros on the minor axis and normalises to [-1,1]
        """
        device = self.device
        if isinstance(img, PIL.Image.Image):
            device = 'cpu'
            img = np.array(img, dtype=np.float32)
            img = torch.from_numpy(img)
            # Handle grayscale images (add channel dimension)
            if img.ndim == 2:
                img = img.unsqueeze(-1)  # H W -> H W 1
            img = rearrange(img, 'H W C -> C H W')

        C, H, W = img.shape
        ratio = self.hw / max(H, W)
        img = nn.functional.interpolate(img.unsqueeze(0), size=(int(H*ratio+0.5), int(W*ratio+0.5)), mode='bilinear', align_corners=False)[0]

        C, H, W = img.shape
        if H < self.hw:
            img = torch.cat([img, torch.zeros((C, self.hw-H, W), device=device)], axis=1)
        else:
            img = torch.cat([img, torch.zeros((C, H, self.hw-W), device=device)], axis=2)
        if is_mask:
            return img[:1] / 255
        else:
            return img / 127.5 - 1

class UNetDataset(Dataset):
    def __init__(self, hw=512, path="/content/", mode="dino", n=None):
        if mode == 'L1':
            self.delta_mode = mode
        elif mode == 'dino':
            self.delta_mode = mode
        else:
            raise NotImplementedError
        self.processor = unet_processor(hw, 'cuda')
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
                if meta['similarity_score'] < 0.94:
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
        """
        Does not handle reshape but permutes dims from H W C to C H W and normalises to [-1, 1]
        """
        paths = self.data[idx]
        datapoint = {key: PIL.Image.open(paths[key]) for key in {'original', 'edited', 'mask', 'sub_mask', 'union_mask'}}
        if self.delta_mode == "dino":
            datapoint['delta'] = self.processor.dino_delta(datapoint['edited'], datapoint['original'])
        datapoint = {key: self.processor.preprocess_img(datapoint[key], key == 'mask') for key in datapoint}
        if self.delta_mode == 'L1':
            datapoint['delta'] = torch.abs(datapoint['original'] - datapoint['edited'])

        return datapoint

if __name__ == "__main__":
    import shutil
    shutil.unpack_archive(r'C:\Users\lex\Downloads\data.zip', os.environ['SAVE_PATH']+"/data_sample")
    dataset = UNetDataset(hw=512, path=os.environ['SAVE_PATH']+"/")
    print(f"Dataset size: {len(dataset)}")
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    for batch in dataloader:
        print(batch['original'].shape)
        print(batch['edited'].shape)
        print(batch['mask'].shape)
        break
    shutil.rmtree(os.environ['SAVE_PATH']+"/data_sample")
