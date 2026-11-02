from typing import Optional, Sequence, Tuple

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


class UNetPlusPlus(nn.Module):
    """UNet++ with nested dense skip paths for map-to-map IR-drop regression."""

    def __init__(
        self,
        in_channels: int = 11,
        out_channels: int = 2,
        channels: Optional[Sequence[int]] = None,
        base_channels: int = 64,
        up_mode: str = "bilinear",
        final_activation: Optional[str] = None,
        init_type: Optional[str] = "kaiming",
        pretrained: Optional[str] = None,
        **_: object,
    ) -> None:
        super().__init__()
        if channels is None:
            channels = [base_channels, base_channels * 2, base_channels * 4, base_channels * 8, base_channels * 16]
        channels = [int(c) for c in channels]

        if up_mode not in ("bilinear", "nearest", "transpose"):
            raise ValueError(f"unknown upsample mode: {up_mode}")

        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.up_mode = up_mode
        self.final_activation = final_activation
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        c0, c1, c2, c3, c4 = channels
        self.conv0_0 = ConvBlock(self.in_channels, c0)
        self.conv1_0 = ConvBlock(c0, c1)
        self.conv2_0 = ConvBlock(c1, c2)
        self.conv3_0 = ConvBlock(c2, c3)
        self.conv4_0 = ConvBlock(c3, c4)

        self.conv0_1 = ConvBlock(c0 + c1, c0)
        self.conv1_1 = ConvBlock(c1 + c2, c1)
        self.conv2_1 = ConvBlock(c2 + c3, c2)
        self.conv3_1 = ConvBlock(c3 + c4, c3)

        self.conv0_2 = ConvBlock(c0 * 2 + c1, c0)
        self.conv1_2 = ConvBlock(c1 * 2 + c2, c1)
        self.conv2_2 = ConvBlock(c2 * 2 + c3, c2)

        self.conv0_3 = ConvBlock(c0 * 3 + c1, c0)
        self.conv1_3 = ConvBlock(c1 * 3 + c2, c1)

        self.conv0_4 = ConvBlock(c0 * 4 + c1, c0)
        self.head = nn.Conv2d(c0, self.out_channels, kernel_size=1)

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

    def _upsample(self, x: torch.Tensor, size: Tuple[int, int]) -> torch.Tensor:
        if self.up_mode == "nearest":
            return F.interpolate(x, size=size, mode="nearest")
        return F.interpolate(x, size=size, mode="bilinear", align_corners=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, pad_hw = self._pad_to_multiple(x)

        x0_0 = self.conv0_0(x)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x2_0 = self.conv2_0(self.pool(x1_0))
        x3_0 = self.conv3_0(self.pool(x2_0))
        x4_0 = self.conv4_0(self.pool(x3_0))

        x0_1 = self.conv0_1(torch.cat([x0_0, self._upsample(x1_0, x0_0.shape[-2:])], dim=1))
        x1_1 = self.conv1_1(torch.cat([x1_0, self._upsample(x2_0, x1_0.shape[-2:])], dim=1))
        x2_1 = self.conv2_1(torch.cat([x2_0, self._upsample(x3_0, x2_0.shape[-2:])], dim=1))
        x3_1 = self.conv3_1(torch.cat([x3_0, self._upsample(x4_0, x3_0.shape[-2:])], dim=1))

        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self._upsample(x1_1, x0_0.shape[-2:])], dim=1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self._upsample(x2_1, x1_0.shape[-2:])], dim=1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self._upsample(x3_1, x2_0.shape[-2:])], dim=1))

        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self._upsample(x1_2, x0_0.shape[-2:])], dim=1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self._upsample(x2_2, x1_0.shape[-2:])], dim=1))

        x0_4 = self.conv0_4(
            torch.cat([x0_0, x0_1, x0_2, x0_3, self._upsample(x1_3, x0_0.shape[-2:])], dim=1)
        )
        out = self._crop(self.head(x0_4), pad_hw)
        return apply_final_activation(out, self.final_activation)

    def init_weights(self, init_type: str = "kaiming", gain: float = 0.02, **_: object) -> None:
        init_weights(self, init_type=init_type, gain=gain)

    def load(self, checkpoint_path: str, strict: bool = False) -> None:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        state_dict = normalise_state_dict(checkpoint)
        missing, unexpected = self.load_state_dict(state_dict, strict=strict)
        if missing or unexpected:
            print(f"Loaded {checkpoint_path} with missing={len(missing)}, unexpected={len(unexpected)}")


NestedUNet = UNetPlusPlus
UNetPP = UNetPlusPlus
PGAU = UNetPlusPlus
