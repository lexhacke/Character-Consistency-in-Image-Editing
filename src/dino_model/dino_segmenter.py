from typing import Dict
import torch
from torch import nn
from PIL import Image
from transformers import AutoImageProcessor, AutoModel
from einops import rearrange
from torch.nn import functional as F

RMSNorm = lambda x : x / x.pow(2).mean(dim=-1, keepdim=True).add(1e-6).sqrt()

class SwiGLU(nn.Module):
    def __init__(self, dim: int, upsample=2):
        """
        dim = embedding dimension
        tokens = number of tokens per embedding
        """
        super().__init__()
        self.dim = dim
        self.linearIn = nn.Linear(dim, upsample*dim, bias=True)
        self.gate = nn.Linear(dim, upsample*dim, bias=True)
        self.linearOut = nn.Linear(upsample*dim, dim, bias=True)

    def forward(self, x: torch.Tensor):
        """
        Requires input to be B N D where N=tokens
        """
        x = self.linearOut(F.silu(self.linearIn(x)) * self.gate(x))
        return RMSNorm(x)

class DinoSegmenter(nn.Module):
    """Minimal segmenter that compares original and edited images with DINO."""

    def __init__(
        self,
        layers=1,
        mlp_mult=4,
        model_name: str = "facebook/dinov3-vits16-pretrain-lvd1689m",
        prefix_tokens: int = 5,
        output_channels: int = 1,
    ) -> None:
        super().__init__()
        self.model_name = model_name
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.backbone = AutoModel.from_pretrained(model_name)
        self.dim = self.backbone.config.hidden_size
        self.prefix_tokens = prefix_tokens
        self.output_channels = output_channels
        size = self.processor.size
        if isinstance(size, dict):
            self.image_size = size.get("height") or size.get("width") or 224
        else:
            self.image_size = size or 224
        self.patch_size = getattr(self.backbone.config, "patch_size", 16)
        self.patch_area = self.patch_size * self.patch_size
        self.layers = nn.ModuleList([SwiGLU(2 * self.dim, upsample=mlp_mult) for _ in range(layers)])
        self.output = nn.Linear(2 * self.dim, self.output_channels * self.patch_area)
        self.activation = nn.Sigmoid()

    def forward(
        self,
        original_inputs: Dict[str, torch.Tensor],
        edited_inputs: Dict[str, torch.Tensor],
        **model_kwargs,
    ) -> torch.Tensor:
        """Encode two sets of inputs and concatenate their last hidden states."""

        original_outputs = self.backbone(**original_inputs, **model_kwargs)
        edited_outputs = self.backbone(**edited_inputs, **model_kwargs)

        original_hidden = original_outputs.last_hidden_state
        edited_hidden = edited_outputs.last_hidden_state

        if original_hidden.shape[:2] != edited_hidden.shape[:2]:
            raise ValueError(
                "Original and edited embeddings must share batch and sequence dimensions; "
                f"got {original_hidden.shape[:2]} vs {edited_hidden.shape[:2]}"
            )

        concatenated = torch.cat([original_hidden, edited_hidden], dim=-1)[:, self.prefix_tokens :, :]

        for layer in self.layers:
            concatenated = layer(concatenated)
        mask = self.activation(self.output(concatenated))

        tokens = mask.shape[1]
        grid = int(tokens ** 0.5)
        if grid * grid != tokens:
            raise ValueError(f"Unexpected token count {tokens}; cannot form square grid.")

        mask = rearrange(
            mask,
            "B (H W) (P p C) -> B C (P H) (p W)",
            H=grid,
            W=grid,
            P=self.patch_size,
            p=self.patch_size,
            C=self.output_channels,
        )
        return mask


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    with torch.no_grad():
        segmenter = DinoSegmenter()
        original, edit = Image.open(r"src\hamster.jpg"), Image.open(r"src\durag_hamster.png")
        original_inputs = segmenter.processor(images=original, return_tensors="pt")
        edited_inputs = segmenter.processor(images=edit, return_tensors="pt")
        mask = segmenter(original_inputs, edited_inputs)
        print(mask.shape)
        plt.imshow(mask[0, 0].cpu(), cmap="gray")
        plt.show()
