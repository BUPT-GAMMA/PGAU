from collections import OrderedDict
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def normalise_state_dict(state_dict):
    if isinstance(state_dict, dict):
        for key in ("state_dict", "model", "model_state_dict"):
            if key in state_dict and isinstance(state_dict[key], dict):
                state_dict = state_dict[key]
                break

    normalised = OrderedDict()
    for key, value in state_dict.items():
        if key.startswith("module."):
            key = key[len("module.") :]
        normalised[key] = value
    return normalised


def init_weights(module: nn.Module, init_type: str = "kaiming", gain: float = 0.02) -> None:
    for layer in module.modules():
        if isinstance(layer, nn.LayerNorm):
            nn.init.constant_(layer.bias, 0.0)
            nn.init.constant_(layer.weight, 1.0)
        elif isinstance(layer, (nn.Conv2d, nn.Conv3d, nn.ConvTranspose2d, nn.ConvTranspose3d, nn.Linear)):
            if init_type == "normal":
                nn.init.normal_(layer.weight, 0.0, gain)
            elif init_type == "xavier":
                nn.init.xavier_normal_(layer.weight, gain=gain)
            elif init_type == "kaiming":
                nn.init.kaiming_normal_(layer.weight, a=0, mode="fan_in")
            elif init_type == "orthogonal":
                nn.init.orthogonal_(layer.weight, gain=gain)
            else:
                raise ValueError(f"unknown init_type: {init_type}")
            if layer.bias is not None:
                nn.init.constant_(layer.bias, 0.0)
        elif isinstance(layer, (nn.BatchNorm2d, nn.BatchNorm3d)):
            if layer.weight is not None:
                nn.init.normal_(layer.weight, 1.0, gain)
            if layer.bias is not None:
                nn.init.constant_(layer.bias, 0.0)


def apply_final_activation(x: torch.Tensor, activation: Optional[str]) -> torch.Tensor:
    if activation in (None, "none", "linear", ""):
        return x
    if activation == "sigmoid":
        return torch.sigmoid(x)
    if activation == "relu":
        return F.relu(x)
    raise ValueError(f"unknown final_activation: {activation}")
