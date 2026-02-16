import torch
from unet import UNet
from unet_dataset import unet_processor
import json

if __name__ == "__main__":
    import PIL
    config = json.load(open("model/config.json"))

    processor = unet_processor(config['hw'], 'cpu')
    unet = UNet(config['filters'], in_channels=7, n_heads=8)
    unet.load_state_dict(torch.load("model/unet_final.pt"))

    hamster = PIL.Image.open("hamster.jpg")
    durag = PIL.Image.open("durag_hamster.png")
    original = processor.preprocess_img(hamster, is_mask=False)
    edited = processor.preprocess_img(durag, is_mask=False)
    delta = processor.dino_delta(hamster, durag)
    x = torch.cat([original, edited, delta], dim=0)
    print(x.shape)
