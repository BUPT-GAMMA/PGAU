from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.model_utils import apply_final_activation, init_weights, normalise_state_dict


def _make_divisible(value: float, divisor: int = 8, min_value: Optional[int] = None) -> int:
    if min_value is None:
        min_value = divisor
    new_value = max(min_value, int(value + divisor / 2) // divisor * divisor)
    if new_value < 0.9 * value:
        new_value += divisor
    return int(new_value)


class ConvBNAct(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        groups: int = 1,
        activation: Optional[str] = "relu",
    ) -> None:
        super().__init__()
        padding = (kernel_size - 1) // 2
        layers = [
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                groups=groups,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
        ]
        if activation == "relu":
            layers.append(nn.ReLU(inplace=True))
        elif activation == "hswish":
            layers.append(nn.Hardswish(inplace=True))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class SqueezeExcitation(nn.Module):
    def __init__(self, channels: int, reduction: int = 4) -> None:
        super().__init__()
        squeeze_channels = _make_divisible(channels // reduction, 8)
        self.block = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, squeeze_channels, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(squeeze_channels, channels, kernel_size=1),
            nn.Hardsigmoid(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.block(x)


class InvertedResidual(nn.Module):
    def __init__(
        self,
        in_channels: int,
        expanded_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        use_se: bool,
        activation: str,
    ) -> None:
        super().__init__()
        layers = []
        if expanded_channels != in_channels:
            layers.append(ConvBNAct(in_channels, expanded_channels, kernel_size=1, activation=activation))
        layers.append(
            ConvBNAct(
                expanded_channels,
                expanded_channels,
                kernel_size=kernel_size,
                stride=stride,
                groups=expanded_channels,
                activation=activation,
            )
        )
        if use_se:
            layers.append(SqueezeExcitation(expanded_channels))
        layers.append(ConvBNAct(expanded_channels, out_channels, kernel_size=1, activation=None))

        self.block = nn.Sequential(*layers)
        self.use_residual = stride == 1 and in_channels == out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.block(x)
        if self.use_residual:
            out = out + x
        return out


class MobileNetV3Encoder(nn.Module):
    def __init__(self, in_channels: int, width_mult: float = 1.0) -> None:
        super().__init__()
        stem_channels = _make_divisible(16 * width_mult)
        self.stem = ConvBNAct(in_channels, stem_channels, kernel_size=3, stride=2, activation="hswish")

        settings = [
            # kernel, expansion, output, se, activation, stride
            (3, 16, 16, False, "relu", 1),
            (3, 64, 24, False, "relu", 2),
            (3, 72, 24, False, "relu", 1),
            (5, 72, 40, True, "relu", 2),
            (5, 120, 40, True, "relu", 1),
            (5, 120, 40, True, "relu", 1),
            (3, 240, 80, False, "hswish", 2),
            (3, 200, 80, False, "hswish", 1),
            (3, 184, 80, False, "hswish", 1),
            (3, 184, 80, False, "hswish", 1),
            (3, 480, 112, True, "hswish", 1),
            (3, 672, 112, True, "hswish", 1),
            (5, 672, 160, True, "hswish", 2),
            (5, 960, 160, True, "hswish", 1),
            (5, 960, 160, True, "hswish", 1),
        ]

        blocks = []
        current_channels = stem_channels
        low_channels = None
        for index, (kernel, expansion, output, use_se, activation, stride) in enumerate(settings):
            expanded_channels = _make_divisible(expansion * width_mult)
            out_channels = _make_divisible(output * width_mult)
            blocks.append(
                InvertedResidual(
                    current_channels,
                    expanded_channels,
                    out_channels,
                    kernel_size=kernel,
                    stride=stride,
                    use_se=use_se,
                    activation=activation,
                )
            )
            current_channels = out_channels
            if index == 2:
                low_channels = out_channels

        self.blocks = nn.ModuleList(blocks)
        self.low_level_index = 2
        self.low_channels = int(low_channels if low_channels is not None else current_channels)
        self.out_channels = int(current_channels)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.stem(x)
        low = None
        for index, block in enumerate(self.blocks):
            x = block(x)
            if index == self.low_level_index:
                low = x
        if low is None:
            low = x
        return low, x


class LRASPPDecoder(nn.Module):
    def __init__(self, low_channels: int, high_channels: int, out_channels: int, decoder_channels: int) -> None:
        super().__init__()
        self.high_proj = ConvBNAct(high_channels, decoder_channels, kernel_size=1, activation="relu")
        self.scale = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(high_channels, decoder_channels, kernel_size=1),
            nn.Hardsigmoid(inplace=True),
        )
        self.low_proj = ConvBNAct(low_channels, decoder_channels, kernel_size=1, activation="relu")
        self.fuse = nn.Sequential(
            ConvBNAct(decoder_channels * 2, decoder_channels, kernel_size=3, activation="relu"),
            nn.Conv2d(decoder_channels, out_channels, kernel_size=1),
        )

    def forward(self, low: torch.Tensor, high: torch.Tensor) -> torch.Tensor:
        high_context = self.high_proj(high) * self.scale(high)
        high_context = F.interpolate(high_context, size=low.shape[-2:], mode="bilinear", align_corners=False)
        low_context = self.low_proj(low)
        return self.fuse(torch.cat([low_context, high_context], dim=1))


class LRASPPMobileNetV3(nn.Module):
    """MobileNetV3-Large encoder with the LR-ASPP multi-scale decoder."""

    def __init__(
        self,
        in_channels: int = 11,
        out_channels: int = 2,
        base_channels: int = 64,
        width_mult: Optional[float] = None,
        decoder_channels: Optional[int] = None,
        final_activation: Optional[str] = None,
        pretrained: Optional[str] = None,
        init_type: Optional[str] = "kaiming",
        **_: object,
    ) -> None:
        super().__init__()
        if width_mult is None:
            width_mult = max(float(base_channels) / 64.0, 0.25)
        if decoder_channels is None:
            decoder_channels = max(_make_divisible(base_channels), 8)

        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.final_activation = final_activation
        self.encoder = MobileNetV3Encoder(self.in_channels, width_mult=float(width_mult))
        self.decoder = LRASPPDecoder(
            self.encoder.low_channels,
            self.encoder.out_channels,
            self.out_channels,
            decoder_channels=int(decoder_channels),
        )

        if pretrained:
            self.load(pretrained)
        elif init_type is not None:
            init_weights(self, init_type=init_type)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        original_size = x.shape[-2:]
        low, high = self.encoder(x)
        out = self.decoder(low, high)
        if out.shape[-2:] != original_size:
            out = F.interpolate(out, size=original_size, mode="bilinear", align_corners=False)
        return apply_final_activation(out, self.final_activation)

    def init_weights(self, init_type: str = "kaiming", gain: float = 0.02, **_: object) -> None:
        init_weights(self, init_type=init_type, gain=gain)

    def load(self, checkpoint_path: str, strict: bool = False) -> None:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        state_dict = normalise_state_dict(checkpoint)
        missing, unexpected = self.load_state_dict(state_dict, strict=strict)
        if missing or unexpected:
            print(f"Loaded {checkpoint_path} with missing={len(missing)}, unexpected={len(unexpected)}")


class MobileNetV3Segmentation(nn.Module):
    """MobileNetV3-Large with a direct lightweight dense-regression head.

    This excludes LR-ASPP so the two model types remain structurally distinct.
    """

    def __init__(
        self,
        in_channels: int = 11,
        out_channels: int = 2,
        base_channels: int = 64,
        width_mult: Optional[float] = None,
        decoder_channels: Optional[int] = None,
        final_activation: Optional[str] = None,
        pretrained: Optional[str] = None,
        init_type: Optional[str] = "kaiming",
        **_: object,
    ) -> None:
        super().__init__()
        if width_mult is None:
            width_mult = max(float(base_channels) / 64.0, 0.25)
        if decoder_channels is None:
            decoder_channels = max(_make_divisible(base_channels), 8)
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.final_activation = final_activation
        self.encoder = MobileNetV3Encoder(self.in_channels, width_mult=float(width_mult))
        self.head = nn.Sequential(
            ConvBNAct(self.encoder.out_channels, int(decoder_channels), kernel_size=1, activation="relu"),
            nn.Conv2d(int(decoder_channels), self.out_channels, kernel_size=1),
        )
        if pretrained:
            self.load(pretrained)
        elif init_type is not None:
            init_weights(self, init_type=init_type)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        original_size = x.shape[-2:]
        _, high = self.encoder(x)
        out = self.head(high)
        out = F.interpolate(out, size=original_size, mode="bilinear", align_corners=False)
        return apply_final_activation(out, self.final_activation)

    def init_weights(self, init_type: str = "kaiming", gain: float = 0.02, **_: object) -> None:
        init_weights(self, init_type=init_type, gain=gain)

    def load(self, checkpoint_path: str, strict: bool = False) -> None:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        state_dict = normalise_state_dict(checkpoint)
        missing, unexpected = self.load_state_dict(state_dict, strict=strict)
        if missing or unexpected:
            print(f"Loaded {checkpoint_path} with missing={len(missing)}, unexpected={len(unexpected)}")


MobileNetV3 = MobileNetV3Segmentation
LRASPP = LRASPPMobileNetV3
PGAU = MobileNetV3Segmentation
