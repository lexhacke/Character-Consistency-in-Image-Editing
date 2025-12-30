import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

class SpatialAttention(nn.Module):
    """
    Spatial Self-Attention Module for 2D feature maps.
    First, we compute normalised query, key, and value tensors using a Conv2d:
    - A 3x3 kernel if Linear=False
    - A 1x1 kernel if Linear=True
    Then, we apply scaled dot-product attention across the spatial dimensions.
    Finally, we project the output back to the original embedding dimension.
    """
    def __init__(self, emb_dim, Linear=False, n_heads=8):
        super().__init__()
        self.n_heads = n_heads
        self.norm = nn.GroupNorm(num_groups=32, eps=1e-6, affine=True, num_channels=emb_dim)
        self.qkv = nn.Conv2d(emb_dim, 3*emb_dim, kernel_size=3 if not Linear else 1, stride=1, padding=1 if not Linear else 0, bias=False)
        self.proj = nn.Conv2d(emb_dim, emb_dim, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, x):
        B, C, H, W = x.shape
        assert C % 32 == 0, "Channels must be divisible by Groups (32)"
        assert C % self.n_heads == 0, "Channels must be divisible by number of heads"
        q, k, v = self.qkv(self.norm(x)).chunk(3, dim=1)

        # to 1D per head
        q = rearrange(q, "B (h D) H W -> B h (H W) D", H=H, W=W, h=self.n_heads)
        k = rearrange(k, "B (h D) H W -> B h (H W) D", H=H, W=W, h=self.n_heads)
        v = rearrange(v, "B (h D) H W -> B h (H W) D", H=H, W=W, h=self.n_heads)

        dx = F.scaled_dot_product_attention(q, k, v)
        dx = rearrange(dx, "B h (H W) D -> B (h D) H W", H=H, W=W, h=self.n_heads)
        dx = self.proj(dx)
        return x + dx

if __name__ == "__main__":
    print(SpatialAttention(64, n_heads=8)(torch.randn(32, 64, 16, 16)).shape)
