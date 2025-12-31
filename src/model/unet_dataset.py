import shutil, os, dotenv
import json
import numpy as np
import PIL
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset

dotenv.load_dotenv()

assert os.environ.get('SAVE_PATH') is not None, "Please set the SAVE_PATH in the .env file"

class UNetDataset(Dataset):
    def __init__(self):
        path = os.environ.get('SAVE_PATH') + "/"
        folder = "data_sample/success/"
        self.data = []
        for file in os.listdir(path+folder):
            try:
                meta = json.load(open(path+folder+file+'/meta.json'))
                if meta['similarity_score'] < 0.94:
                    continue
                original = np.array(PIL.Image.open(path+folder+file+'/base.jpeg')) if meta['base'] == "original" else np.array(PIL.Image.open(path+folder+file+'/other.jpeg'))
                edited = np.array(PIL.Image.open(path+folder+file+'/base.jpeg')) if meta['base'] == "edited" else np.array(PIL.Image.open(path+folder+file+'/other.jpeg'))
                mask = np.array(PIL.Image.open(path+folder+file+'/mask.png'))
                sub_mask = np.array(PIL.Image.open(path+folder+file+'/subtraction_mask.png'))
                union_mask = np.array(PIL.Image.open(path+folder+file+'/union_mask.png'))
                self.data.append({
                    'original': original,
                    'edited': edited,
                    'mask': mask,
                    'sub_mask': sub_mask,
                    'union_mask': union_mask
                })
            except Exception as e:
              print(e)
              continue

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

if __name__ == "__main__":
    dataset = UNetDataset()
    print(len(dataset))