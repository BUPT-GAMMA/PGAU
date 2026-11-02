import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.model_utils import apply_final_activation, init_weights, normalise_state_dict


class ConvBNReLU(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class SETR(nn.Module):
    """SETR-PUP style transformer segmentation baseline for dense regression maps."""

    def __init__(
        self,
        in_channels: int = 11,
        out_channels: int = 2,
        image_size: int = 256,
        patch_size: int = 16,
        embed_dim: Optional[int] = None,
        depth: int = 6,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        base_channels: int = 64,
        final_activation: Optional[str] = None,
        pretrained: Optional[str] = None,
        init_type: Optional[str] = "kaiming",
        **_: object,
    ) -> None:
        super().__init__()
        if patch_size <= 0 or patch_size & (patch_size - 1) != 0:
            raise ValueError("patch_size must be a positive power of two")
        if embed_dim is None:
            embed_dim = max(int(base_channels) * 4, 64)
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")

        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.image_size = int(image_size)
        self.patch_size = int(patch_size)
        self.embed_dim = int(embed_dim)
        self.final_activation = final_activation

        grid_size = max(1, math.ceil(self.image_size / self.patch_size))
        self.patch_embed = nn.Conv2d(
            self.in_channels,
            self.embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            bias=True,
        )
        self.pos_embed = nn.Parameter(torch.zeros(1, grid_size * grid_size, self.embed_dim))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.embed_dim,
            nhead=num_heads,
            dim_feedforward=int(self.embed_dim * mlp_ratio),
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.norm = nn.LayerNorm(self.embed_dim)

        decoder_channels = max(int(base_channels) * 2, 32)
        up_blocks = []
        in_width = self.embed_dim
        for _ in range(int(math.log2(self.patch_size))):
            up_blocks.append(ConvBNReLU(in_width, decoder_channels, kernel_size=3))
            in_width = decoder_channels
        self.decoder = nn.ModuleList(up_blocks)
        self.refine = nn.Sequential(
            ConvBNReLU(decoder_channels, decoder_channels, kernel_size=3),
            nn.Conv2d(decoder_channels, self.out_channels, kernel_size=1),
        )

        if init_type is not None:
            init_weights(self, init_type=init_type)
            nn.init.trunc_normal_(self.pos_embed, std=0.02)
        if pretrained:
            self.load(pretrained)

    @staticmethod
    def _pad_to_multiple(x: torch.Tensor, multiple: int) -> Tuple[torch.Tensor, Tuple[int, int]]:
        height, width = x.shape[-2:]
        pad_h = (multiple - height % multiple) % multiple
        pad_w = (multiple - width % multiple) % multiple
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h))
        return x, (height, width)

    def _position_embedding(self, height: int, width: int) -> torch.Tensor:
        tokens = self.pos_embed
        base = int(math.sqrt(tokens.shape[1]))
        if (height, width) == (base, base):
            return tokens
        tokens = tokens.reshape(1, base, base, self.embed_dim).permute(0, 3, 1, 2)
        tokens = F.interpolate(tokens, size=(height, width), mode="bilinear", align_corners=False)
        return tokens.flatten(2).transpose(1, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, original_size = self._pad_to_multiple(x, self.patch_size)
        patches = self.patch_embed(x)
        grid_h, grid_w = patches.shape[-2:]

        tokens = patches.flatten(2).transpose(1, 2)
        tokens = tokens + self._position_embedding(grid_h, grid_w).to(dtype=tokens.dtype, device=tokens.device)
        tokens = self.norm(self.encoder(tokens))

        features = tokens.transpose(1, 2).reshape(x.shape[0], self.embed_dim, grid_h, grid_w)
        for block in self.decoder:
            features = F.interpolate(features, scale_factor=2, mode="bilinear", align_corners=False)
            features = block(features)

        out = self.refine(features)
        if out.shape[-2:] != original_size:
            out = F.interpolate(out, size=original_size, mode="bilinear", align_corners=False)
        return apply_final_activation(out, self.final_activation)

    def init_weights(self, init_type: str = "kaiming", gain: float = 0.02, **_: object) -> None:
        init_weights(self, init_type=init_type, gain=gain)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def load(self, checkpoint_path: str, strict: bool = False) -> None:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        state_dict = normalise_state_dict(checkpoint)
        missing, unexpected = self.load_state_dict(state_dict, strict=strict)
        if missing or unexpected:
            print(f"Loaded {checkpoint_path} with missing={len(missing)}, unexpected={len(unexpected)}")


PGAU = SETR
