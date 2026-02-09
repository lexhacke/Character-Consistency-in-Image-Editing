import torch
import torch.nn as nn
from attention import SpatialAttention
from unet import ResBlock, UNet

if __name__ == "__main__":
    hw = 128
    filters = [32, 64, 128, 256]
    unet = UNet(filters, in_channels=7, n_heads=8)
    print(unet(torch.randn(4, 7, hw, hw)).shape)
