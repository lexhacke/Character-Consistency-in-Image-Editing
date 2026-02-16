import torch
import torch.nn as nn
from attention import SpatialAttention
import json

class ResBlock(nn.Module):
    """
    ResNet-Style Residual Block Module with Group Normalization and Conv2d layers.
    - Applies automatic channel reshaping if input and output channels differ.
    """
    def __init__(self, in_c, out_c):
        super().__init__()
        assert in_c % 32 == 0, f"Channels must be divisible by Groups (32) recieved: {in_c}"
        self.in_c = in_c
        self.out_c = out_c
        self.reshape = False
        if in_c != out_c:
            self.reshape = True
            self.conv_reshape = nn.Conv2d(in_c, out_c, kernel_size=3, stride=1, padding=1)
        self.norm1 = nn.GroupNorm(num_groups=32, num_channels=out_c, eps=1e-6, affine=True)
        self.conv1 = nn.Conv2d(out_c, out_c, kernel_size=3, stride=1, padding=1)
        self.norm2 = nn.GroupNorm(num_groups=32, num_channels=out_c, eps=1e-6, affine=True)
        self.conv2 = nn.Conv2d(out_c, out_c, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        B, C, H, W = x.shape
        assert C == self.in_c, f"expected {self.in_c} channels, got {C}"
        if self.reshape:
            x = self.conv_reshape(x)
        res = x
        x = self.norm1(x)
        x = x * torch.sigmoid(x)
        x = self.conv1(x)
        x = self.norm2(x)
        x = x * torch.sigmoid(x)
        x = self.conv2(x)
        x = x + res
        return x

class UNet(nn.Module):
    def __init__(self, filters, in_channels=9, n_heads=8):
        """
        UNet architecture with residual blocks and spatial attention.
        - filters: List of filter sizes for each level of the UNet (must be divisible by 32)
        - in_channels: Number of input channels (default: 9 - 3 for original image, 3 for edited image, 3 for mask)
        - n_heads: Number of attention heads (default: 8)
        """
        super().__init__()
        self.input_proj = nn.Conv2d(in_channels, filters[0], kernel_size=3, stride=1, padding=1)

        self.down_path = nn.ModuleList()
        for i in range(len(filters)-1):
            self.down_path.append(
                nn.ModuleList([
                    ResBlock(filters[i], filters[i+1]),
                    nn.Conv2d(filters[i+1], filters[i+1], kernel_size=4, stride=2, padding=1) # Pooling
                ])
            )

        self.bottleneck = nn.Sequential(
            ResBlock(filters[-1], filters[-1]),
            SpatialAttention(filters[-1], Linear=False, n_heads=n_heads),
            ResBlock(filters[-1], filters[-1])
        )

        self.up_path = nn.ModuleList()
        for i in reversed(range(len(filters)-1)):
            self.up_path.append(
                nn.ModuleList([
                    nn.Upsample(scale_factor=2, mode='bilinear'),
                    ResBlock(filters[i+1]*2, filters[i])
                ])
            )

        self.output_proj = nn.Conv2d(filters[0]*2, 1, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        if hasattr(self, 'in_layer'):
            x = self.in_layer(x)
        x = self.input_proj(x)
        skip_stack = [x]

        for down in self.down_path:
            x = down[0](x) # ResBlock
            skip_stack.append(x)
            x = down[1](x) # Pooling

        x = self.bottleneck(x)

        for up in self.up_path:
            x = torch.cat([up[0](x), skip_stack.pop()], dim=1)
            x = up[1](x)

        x = self.output_proj(torch.cat([x, skip_stack.pop()], dim=1))
        return x

if __name__ == "__main__":
    config = json.load(open("model/config.json"))
    unet = UNet(config['filters'], in_channels=7, n_heads=8)
    unet.load_state_dict(torch.load("model/unet_final.pt"))
    print(unet(torch.randn(4, 7, config['hw'], config['hw'])).shape)
