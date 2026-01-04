import os, dotenv, json, PIL.Image
import numpy as np
from einops import rearrange
from torch.utils.data import DataLoader, Dataset

dotenv.load_dotenv()

assert os.environ.get('SAVE_PATH') is not None, "Please set the SAVE_PATH in the .env file"

class UNetDataset(Dataset):
    def __init__(self, hw=512, path="/content/"):
        self.hw = hw
        folder = "data_sample/success/"
        self.data = []
        for file in os.listdir(path+folder):
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
              continue

    def __len__(self):
        return len(self.data)
    
    def preprocess_img(self, img, is_mask):
        """
        Expects img of PIL.Image, reshape via padding with zeros on the minor axis
        """
        W, H = img.size
        ratio = self.hw / max(H, W)
        img = img.resize((int(H*ratio), int(W*ratio)), resample=PIL.Image.NEAREST if is_mask else PIL.Image.BILINEAR)
        img = np.array(img, dtype=np.float32)
        img = rearrange(img, 'H W C -> C H W')
        C, H, W = img.shape
        if H < 512:
            img = np.concatenate([img, np.zeros((C, 512-H, W))], axis=1)
        else:
            img = np.concatenate([img, np.zeros((C, H, 512-W))], axis=2)
        if is_mask:
            return img[0] / 255
        else:
            return img / 127.5 - 1

    def __getitem__(self, idx):
        """
        Does not handle reshape but permutes dims from H W C to C H W and normalises to [-1, 1]
        """
        paths = self.data[idx]
        datapoint = {}
        for key in {'original', 'edited', 'mask', 'sub_mask', 'union_mask'}:
            datapoint[key] = PIL.Image.open(paths[key])
            datapoint[key] = self.preprocess_img(datapoint[key], key in {'mask', 'sub_mask', 'union_mask'})
        return datapoint

if __name__ == "__main__":
    import shutil
    shutil.unpack_archive('C:\\Users\\lex\\Downloads\\data.zip', os.environ['SAVE_PATH']+"\\data_sample")
    dataset = UNetDataset(hw=512, path=os.environ['SAVE_PATH'])
    print(f"Dataset size: {len(dataset)}")
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    for batch in dataloader:
        print(batch['original'].shape)
        print(batch['edited'].shape)
        print(batch['mask'].shape)
        break
    shutil.rmtree(os.environ['SAVE_PATH']+"\\data_sample")