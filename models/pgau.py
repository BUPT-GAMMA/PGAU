from typing import Iterable, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.model_utils import apply_final_activation, init_weights, normalise_state_dict


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UpBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, mode: str = "bilinear") -> None:
        super().__init__()
        if mode == "transpose":
            self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1)
            conv_in = out_channels
        elif mode == "bilinear":
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
            conv_in = in_channels
        elif mode == "nearest":
            self.up = nn.Upsample(scale_factor=2, mode="nearest")
            conv_in = in_channels
        else:
            raise ValueError(f"unknown upsample mode: {mode}")

        self.conv = nn.Sequential(
            nn.Conv2d(conv_in, out_channels, kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(self.up(x))


class AttentionGate(nn.Module):

    def __init__(self, gate_channels: int, skip_channels: int, inter_channels: int) -> None:
        super().__init__()
        self.gate_proj = nn.Sequential(
            nn.Conv2d(gate_channels, inter_channels, kernel_size=1, bias=True),
            nn.BatchNorm2d(inter_channels),
        )
        self.skip_proj = nn.Sequential(
            nn.Conv2d(skip_channels, inter_channels, kernel_size=1, bias=True),
            nn.BatchNorm2d(inter_channels),
        )
        self.psi = nn.Sequential(
            nn.Conv2d(inter_channels, 1, kernel_size=1, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, gate: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        if gate.shape[-2:] != skip.shape[-2:]:
            gate = F.interpolate(gate, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        attention = self.relu(self.gate_proj(gate) + self.skip_proj(skip))
        attention = self.psi(attention)
        return skip * attention


class IdentityAttention(nn.Module):
    def forward(self, gate: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        return skip


class PGAU(nn.Module):

    def __init__(
        self,
        in_channels: int = 11,
        out_channels: int = 1,
        channels: Optional[Sequence[int]] = None,
        base_channels: int = 64,
        up_mode: str = "bilinear",
        final_activation: Optional[str] = None,
        use_attention: bool = True,
        init_type: Optional[str] = "kaiming",
        pretrained: Optional[str] = None,
        **_: object,
    ) -> None:
        super().__init__()
        if channels is None:
            channels = [base_channels, base_channels * 2, base_channels * 4, base_channels * 8, base_channels * 16]
        channels = [int(c) for c in channels]

        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.final_activation = final_activation
        self.use_attention = bool(use_attention)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        c1, c2, c3, c4, c5 = channels
        self.enc1 = ConvBlock(self.in_channels, c1)
        self.enc2 = ConvBlock(c1, c2)
        self.enc3 = ConvBlock(c2, c3)
        self.enc4 = ConvBlock(c3, c4)
        self.bottleneck = ConvBlock(c4, c5)

        self.up4 = UpBlock(c5, c4, mode=up_mode)
        attention = AttentionGate if self.use_attention else None
        self.att4 = attention(c4, c4, max(c3, 1)) if attention else IdentityAttention()
        self.dec4 = ConvBlock(c4 + c4, c4)

        self.up3 = UpBlock(c4, c3, mode=up_mode)
        self.att3 = attention(c3, c3, max(c2, 1)) if attention else IdentityAttention()
        self.dec3 = ConvBlock(c3 + c3, c3)

        self.up2 = UpBlock(c3, c2, mode=up_mode)
        self.att2 = attention(c2, c2, max(c1, 1)) if attention else IdentityAttention()
        self.dec2 = ConvBlock(c2 + c2, c2)

        self.up1 = UpBlock(c2, c1, mode=up_mode)
        self.att1 = attention(c1, c1, max(c1 // 2, 1)) if attention else IdentityAttention()
        self.dec1 = ConvBlock(c1 + c1, c1)

        self.head = nn.Conv2d(c1, self.out_channels, kernel_size=1)

        if pretrained:
            self.load(pretrained)
        elif init_type is not None:
            init_weights(self, init_type=init_type)

    @staticmethod
    def _pad_to_multiple(x: torch.Tensor, multiple: int = 16) -> Tuple[torch.Tensor, Tuple[int, int]]:
        height, width = x.shape[-2:]
        pad_h = (multiple - height % multiple) % multiple
        pad_w = (multiple - width % multiple) % multiple
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h))
        return x, (pad_h, pad_w)

    @staticmethod
    def _crop(x: torch.Tensor, pad_hw: Tuple[int, int]) -> torch.Tensor:
        pad_h, pad_w = pad_hw
        if pad_h:
            x = x[..., :-pad_h, :]
        if pad_w:
            x = x[..., :, :-pad_w]
        return x

    @staticmethod
    def _concat(skip: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
        if skip.shape[-2:] != up.shape[-2:]:
            skip = F.interpolate(skip, size=up.shape[-2:], mode="bilinear", align_corners=False)
        return torch.cat((skip, up), dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, pad_hw = self._pad_to_multiple(x)

        x1 = self.enc1(x)
        x2 = self.enc2(self.pool(x1))
        x3 = self.enc3(self.pool(x2))
        x4 = self.enc4(self.pool(x3))
        x5 = self.bottleneck(self.pool(x4))

        d4 = self.up4(x5)
        d4 = self.dec4(self._concat(self.att4(d4, x4), d4))

        d3 = self.up3(d4)
        d3 = self.dec3(self._concat(self.att3(d3, x3), d3))

        d2 = self.up2(d3)
        d2 = self.dec2(self._concat(self.att2(d2, x2), d2))

        d1 = self.up1(d2)
        d1 = self.dec1(self._concat(self.att1(d1, x1), d1))

        out = self._crop(self.head(d1), pad_hw)
        return apply_final_activation(out, self.final_activation)

    def init_weights(self, init_type: str = "kaiming", gain: float = 0.02, **_: object) -> None:
        init_weights(self, init_type=init_type, gain=gain)

    def load(self, checkpoint_path: str, strict: bool = False) -> None:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        state_dict = normalise_state_dict(checkpoint)
        missing, unexpected = self.load_state_dict(state_dict, strict=strict)
        if missing or unexpected:
            print(f"Loaded {checkpoint_path} with missing={len(missing)}, unexpected={len(unexpected)}")


class AttU_Net(PGAU):

    def __init__(
        self,
        in_channel: int = 11,
        num_classes: int = 1,
        channel_list: Optional[Iterable[int]] = None,
        checkpoint: bool = False,
        convTranspose: bool = True,
        **kwargs: object,
    ) -> None:
        super().__init__(
            in_channels=in_channel,
            out_channels=num_classes,
            channels=list(channel_list) if channel_list is not None else None,
            up_mode="transpose" if convTranspose else "bilinear",
            init_type=None if checkpoint else kwargs.pop("init_type", "kaiming"),
            **kwargs,
        )
