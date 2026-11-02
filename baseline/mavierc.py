from typing import Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.model_utils import apply_final_activation, init_weights, normalise_state_dict


class DoubleConv3d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, mid_channels: Optional[int] = None) -> None:
        super().__init__()
        if mid_channels is None:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv3d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.double_conv(x)


class Down3d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2)),
            DoubleConv3d(in_channels, out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.maxpool_conv(x)


class Up3d(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int, mode: str = "bilinear") -> None:
        super().__init__()
        if mode not in ("bilinear", "nearest", "transpose"):
            raise ValueError(f"unknown upsample mode: {mode}")

        self.mode = mode
        if mode == "transpose":
            self.up = nn.ConvTranspose3d(
                in_channels,
                out_channels,
                kernel_size=(1, 2, 2),
                stride=(1, 2, 2),
            )
            self.reduce = None
        else:
            self.up = None
            self.reduce = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm3d(out_channels),
                nn.ReLU(inplace=True),
            )
        self.conv = DoubleConv3d(skip_channels + out_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        if self.mode == "transpose":
            x = self.up(x)
        else:
            interp_mode = "nearest" if self.mode == "nearest" else "trilinear"
            if interp_mode == "nearest":
                x = F.interpolate(x, scale_factor=(1, 2, 2), mode=interp_mode)
            else:
                x = F.interpolate(x, scale_factor=(1, 2, 2), mode=interp_mode, align_corners=False)
            x = self.reduce(x)

        if x.shape[-3:] != skip.shape[-3:]:
            if self.mode == "nearest":
                x = F.interpolate(x, size=skip.shape[-3:], mode="nearest")
            else:
                x = F.interpolate(x, size=skip.shape[-3:], mode="trilinear", align_corners=False)
        return self.conv(torch.cat([skip, x], dim=1))


class MAVIREC(nn.Module):
    """3D U-Net style MAVIREC baseline adapted to 2D feature-map regression.

    The input feature channels are treated as the 3D depth axis. Spatial
    downsampling and upsampling follow U-Net, and the final 3D prediction is
    reduced along depth to produce a 2D IR-drop map.
    """

    def __init__(
        self,
        in_channels: int = 11,
        out_channels: int = 2,
        channels: Optional[Sequence[int]] = None,
        base_channels: int = 32,
        up_mode: str = "bilinear",
        final_activation: Optional[str] = None,
        depth_reduction: str = "mean",
        init_type: Optional[str] = "kaiming",
        pretrained: Optional[str] = None,
        **_: object,
    ) -> None:
        super().__init__()
        if channels is None:
            channels = [base_channels, base_channels * 2, base_channels * 4, base_channels * 8, base_channels * 16]
        channels = [int(c) for c in channels]
        if depth_reduction not in ("mean", "max", "center"):
            raise ValueError("depth_reduction must be one of: mean, max, center")

        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.final_activation = final_activation
        self.depth_reduction = depth_reduction

        c1, c2, c3, c4, c5 = channels
        self.inc = DoubleConv3d(1, c1)
        self.down1 = Down3d(c1, c2)
        self.down2 = Down3d(c2, c3)
        self.down3 = Down3d(c3, c4)
        self.down4 = Down3d(c4, c5)
        self.up1 = Up3d(c5, c4, c4, mode=up_mode)
        self.up2 = Up3d(c4, c3, c3, mode=up_mode)
        self.up3 = Up3d(c3, c2, c2, mode=up_mode)
        self.up4 = Up3d(c2, c1, c1, mode=up_mode)
        self.head = nn.Conv3d(c1, self.out_channels, kernel_size=1)

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

    def _reduce_depth(self, x: torch.Tensor) -> torch.Tensor:
        if self.depth_reduction == "mean":
            return x.mean(dim=2)
        if self.depth_reduction == "max":
            return x.max(dim=2).values
        return x[:, :, x.shape[2] // 2]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, pad_hw = self._pad_to_multiple(x)
        x = x.unsqueeze(1)

        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        out = self._reduce_depth(self.head(x))
        out = self._crop(out, pad_hw)
        return apply_final_activation(out, self.final_activation)

    def init_weights(self, init_type: str = "kaiming", gain: float = 0.02, **_: object) -> None:
        init_weights(self, init_type=init_type, gain=gain)

    def load(self, checkpoint_path: str, strict: bool = False) -> None:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        state_dict = normalise_state_dict(checkpoint)
        missing, unexpected = self.load_state_dict(state_dict, strict=strict)
        if missing or unexpected:
            print(f"Loaded {checkpoint_path} with missing={len(missing)}, unexpected={len(unexpected)}")


MAVIRec = MAVIREC
MAVIREC3D = MAVIREC
PGAU = MAVIREC
