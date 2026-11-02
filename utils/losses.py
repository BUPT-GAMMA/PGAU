from dataclasses import dataclass
from typing import Iterable, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def _gaussian_kernel(kernel_size: int, sigma: float) -> np.ndarray:
    if kernel_size % 2 == 0:
        raise ValueError("kernel_size must be odd")
    radius = kernel_size // 2
    x = np.arange(-radius, radius + 1, dtype=np.float64)
    kernel = np.exp(-0.5 * (x / float(sigma)) ** 2)
    kernel /= kernel.sum()
    return kernel


@dataclass
class LDSWeightTable:
    min_value: float
    bin_width: float
    weights: np.ndarray

    @classmethod
    def from_arrays(
        cls,
        arrays: Iterable[np.ndarray],
        bin_width: float = 1e-4,
        kernel_size: int = 5,
        sigma: float = 2.0,
        eps: float = 1e-12,
    ) -> "LDSWeightTable":
        if bin_width <= 0:
            raise ValueError("bin_width must be positive")
        counts = {}
        for array in arrays:
            array = np.asarray(array, dtype=np.float64)
            array = array[np.isfinite(array)]
            if array.size:
                # Discretise labels at the requested precision.
                # Rounding avoids putting float32 values such as 0.0001 into
                # the preceding bin because of binary representation error.
                bin_ids, bin_counts = np.unique(np.rint(array / bin_width).astype(np.int64), return_counts=True)
                for bin_id, count in zip(bin_ids, bin_counts):
                    counts[int(bin_id)] = counts.get(int(bin_id), 0) + int(count)
        if not counts:
            raise ValueError("cannot build LDS table from empty labels")
        min_bin = min(counts)
        max_bin = max(counts)
        min_value = float(min_bin * bin_width)
        hist = np.zeros(max_bin - min_bin + 1, dtype=np.float64)
        for bin_id, count in counts.items():
            hist[bin_id - min_bin] = count
        kernel = _gaussian_kernel(kernel_size, sigma)
        radius = kernel_size // 2
        smooth = np.convolve(np.pad(hist, (radius, radius), mode="constant"), kernel, mode="valid")
        density = smooth / max(float(smooth.sum()), eps)
        weights = 1.0 / np.sqrt(density + eps)
        observed_mean = float(np.sum(weights * hist) / max(float(hist.sum()), eps))
        if observed_mean > 0:
            weights /= observed_mean
        return cls(min_value=min_value, bin_width=bin_width, weights=weights.astype(np.float32))

    def save(self, path: str) -> None:
        np.savez(path, min_value=self.min_value, bin_width=self.bin_width, weights=self.weights)

    @classmethod
    def load(cls, path: str) -> "LDSWeightTable":
        data = np.load(path)
        return cls(float(data["min_value"]), float(data["bin_width"]), data["weights"].astype(np.float32))

    def lookup_tensor(self, target: torch.Tensor) -> torch.Tensor:
        weights = torch.as_tensor(self.weights, device=target.device, dtype=target.dtype)
        indices = torch.round((target - self.min_value) / self.bin_width).long()
        indices = torch.clamp(indices, 0, weights.numel() - 1)
        return weights[indices]


class L1Loss(nn.Module):
    def __init__(self, loss_weight: float = 1.0) -> None:
        super().__init__()
        self.loss_weight = loss_weight

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.loss_weight * F.l1_loss(pred, target)


class MSELoss(nn.Module):
    def __init__(self, loss_weight: float = 1.0) -> None:
        super().__init__()
        self.loss_weight = loss_weight

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.loss_weight * F.mse_loss(pred, target)


class LDSL1Loss(nn.Module):
    def __init__(
        self,
        weight_table: LDSWeightTable,
        loss_weight: float = 1.0,
        collapse_channels: bool = True,
    ) -> None:
        super().__init__()
        self.weight_table = weight_table
        self.loss_weight = loss_weight
        self.collapse_channels = collapse_channels

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        error = torch.abs(pred - target)
        weight_target = target
        if self.collapse_channels and target.ndim == 4 and target.shape[1] > 1:
            weight_target = target.sum(dim=1, keepdim=True)
        weights = self.weight_table.lookup_tensor(weight_target)
        if weights.shape[1] == 1 and error.shape[1] > 1:
            weights = weights.expand_as(error)
        return self.loss_weight * torch.mean(error * weights)


class LDSMSELoss(nn.Module):
    def __init__(
        self,
        weight_table: LDSWeightTable,
        loss_weight: float = 1.0,
        collapse_channels: bool = True,
    ) -> None:
        super().__init__()
        self.weight_table = weight_table
        self.loss_weight = loss_weight
        self.collapse_channels = collapse_channels

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        error = (pred - target) ** 2
        weight_target = target
        if self.collapse_channels and target.ndim == 4 and target.shape[1] > 1:
            weight_target = target.sum(dim=1, keepdim=True)
        weights = self.weight_table.lookup_tensor(weight_target)
        if weights.shape[1] == 1 and error.shape[1] > 1:
            weights = weights.expand_as(error)
        return self.loss_weight * torch.mean(error * weights)


def build_loss(name: str, lds_table: Optional[LDSWeightTable] = None, **kwargs) -> nn.Module:
    name = name.lower()
    if name in {"l1", "mae", "l1loss"}:
        return L1Loss(**kwargs)
    if name in {"mse", "mseloss"}:
        return MSELoss(**kwargs)
    if name in {"lds_l1", "ldsl1", "lds"}:
        if lds_table is None:
            raise ValueError("lds_l1 requires an LDSWeightTable")
        return LDSL1Loss(lds_table, **kwargs)
    if name in {"lds_mse", "ldsmse"}:
        if lds_table is None:
            raise ValueError("lds_mse requires an LDSWeightTable")
        return LDSMSELoss(lds_table, **kwargs)
    raise KeyError(f"unknown loss: {name}")
