from typing import Mapping

from baseline import LRASPPMobileNetV3, MAVIREC, MobileNetV3Segmentation, SETR, UNetPlusPlus
from models.pgau import PGAU


MODEL_REGISTRY = {
    "pgau": PGAU,
    "unet": PGAU,
    "unetpp": UNetPlusPlus,
    "setr": SETR,
    "mobilenetv3": MobileNetV3Segmentation,
    "lr_aspp": LRASPPMobileNetV3,
    "mavirec": MAVIREC,
}


def build_model(args: Mapping):
    args = dict(args)
    model_type = str(args.pop("model_type", "pgau")).lower()
    if model_type not in MODEL_REGISTRY:
        raise KeyError(f"unknown model_type '{model_type}'. Available: {sorted(MODEL_REGISTRY)}")
    if model_type == "unet":
        args["use_attention"] = False
    return MODEL_REGISTRY[model_type](**args)
