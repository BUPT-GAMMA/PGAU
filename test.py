import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict

import numpy as np
import torch
from torch import autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets.npy_dataset import NpyMapDataset
from models.build_model import build_model
from utils.metrics import evaluate_regression, scale_metrics


def load_model_state(model: torch.nn.Module, state_dict: Dict, allow_mismatch: bool = False):
    if not allow_mismatch:
        model.load_state_dict(state_dict, strict=True)
        return [], []
    model_state = model.state_dict()
    compatible = {}
    skipped = []
    for key, value in state_dict.items():
        normalized_key = key[len("module.") :] if key.startswith("module.") else key
        if normalized_key in model_state and model_state[normalized_key].shape == value.shape:
            compatible[normalized_key] = value
        else:
            skipped.append(key)
    missing, unexpected = model.load_state_dict(compatible, strict=False)
    return skipped, list(missing) + list(unexpected)


def resize_map(array: np.ndarray, out_shape) -> np.ndarray:
    if tuple(array.shape) == tuple(out_shape):
        return array
    tensor = torch.from_numpy(array[None, None].astype(np.float32))
    tensor = torch.nn.functional.interpolate(tensor, size=out_shape, mode="bilinear", align_corners=False)
    return tensor[0, 0].numpy()


def total_map(array: np.ndarray) -> np.ndarray:
    if array.ndim == 3 and array.shape[0] > 1:
        return array.sum(axis=0)
    if array.ndim == 3:
        return array[0]
    return array


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/pgau_test.json", help="JSON config file.")
    parser.add_argument("--checkpoint", default="runs/pgau/pgau_circuitnet/best.pth")
    parser.add_argument("--test-csv", default="processed/maps/test.csv")
    parser.add_argument("--dataroot", default=None)
    parser.add_argument("--save-dir", default=None)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--in-channels", type=int, default=None)
    parser.add_argument("--out-channels", type=int, default=None)
    parser.add_argument("--base-channels", type=int, default=None)
    parser.add_argument("--up-mode", default=None, choices=["bilinear", "nearest", "transpose"])
    parser.add_argument("--final-activation", default=None)
    parser.add_argument("--model-type", default=None)
    parser.add_argument("--use-attention", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--target-scale", type=float, default=None)
    parser.add_argument("--metric-scale", type=float, default=None)
    parser.add_argument("--allow-state-mismatch", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--gpu", default=None)
    parser.add_argument("--amp", action="store_true", default=False)
    parser.add_argument("--save-pred", action="store_true")
    args = parser.parse_args()
    explicit = _explicit_cli_dests(parser, sys.argv[1:])

    with open(args.config, "r") as handle:
        config = json.load(handle)
    for key, value in config.items():
        if hasattr(args, key) and key not in explicit:
            setattr(args, key, value)
    return args


def _explicit_cli_dests(parser: argparse.ArgumentParser, argv) -> set:
    explicit = set()
    for action in parser._actions:
        for option in action.option_strings:
            if option in argv or any(token.startswith(option + "=") for token in argv):
                explicit.add(action.dest)
    return explicit


def get_device(args):
    if args.cpu or not torch.cuda.is_available():
        return torch.device("cpu")
    if args.gpu is not None:
        return torch.device("cuda", int(args.gpu))
    return torch.device("cuda", 0)


def main() -> None:
    args = parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    ckpt_config: Dict = checkpoint.get("config", {}) if isinstance(checkpoint, dict) else {}

    dataset = NpyMapDataset(args.test_csv, dataroot=args.dataroot)
    sample = dataset[0]
    in_channels = args.in_channels or ckpt_config.get("in_channels") or int(sample["feature"].shape[0])
    out_channels = args.out_channels or ckpt_config.get("out_channels") or int(sample["label"].shape[0])
    base_channels = args.base_channels or ckpt_config.get("base_channels", 64)
    up_mode = args.up_mode or ckpt_config.get("up_mode", "bilinear")
    final_activation = args.final_activation if args.final_activation is not None else ckpt_config.get("final_activation")
    model_type = args.model_type or ckpt_config.get("model_type", "pgau")
    use_attention = args.use_attention if args.use_attention is not None else ckpt_config.get("use_attention", True)
    target_scale = args.target_scale if args.target_scale is not None else ckpt_config.get("target_scale", 1.0)
    metric_scale = args.metric_scale if args.metric_scale is not None else ckpt_config.get("metric_scale", 1.0)
    if target_scale <= 0 or metric_scale <= 0:
        raise ValueError("target_scale and metric_scale must be positive")

    model = build_model(
        {
            "model_type": model_type,
            "in_channels": in_channels,
            "out_channels": out_channels,
            "base_channels": base_channels,
            "up_mode": up_mode,
            "final_activation": final_activation,
            "use_attention": use_attention,
        }
    )
    state_dict = checkpoint.get("model", checkpoint.get("state_dict", checkpoint))
    skipped, unresolved = load_model_state(model, state_dict, allow_mismatch=args.allow_state_mismatch)
    if args.allow_state_mismatch and (skipped or unresolved):
        print(f"partial checkpoint load: skipped={len(skipped)}, unresolved={len(unresolved)}")

    device = get_device(args)
    model = model.to(device).eval()
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    save_dir = args.save_dir or os.path.join(os.path.dirname(args.checkpoint), "test_result")
    os.makedirs(save_dir, exist_ok=True)

    rows = []
    sums = {"mae": 0.0, "mse": 0.0, "rmse": 0.0, "cc": 0.0, "mirde": 0.0}
    instance_sums = {"mae": 0.0, "mse": 0.0, "rmse": 0.0, "cc": 0.0, "mirde": 0.0}
    instance_count = 0
    count = 0
    inference_seconds = 0.0

    with torch.no_grad():
        for batch in tqdm(loader, desc="Testing", ncols=100):
            feature = batch["feature"].to(device, non_blocking=True)
            label = batch["label"].numpy()
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            started = time.perf_counter()
            with autocast(device_type=device.type, enabled=args.amp and device.type == "cuda"):
                pred = model(feature).detach().cpu().numpy()
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            batch_seconds = time.perf_counter() - started
            inference_seconds += batch_seconds

            for i, name in enumerate(batch["name"]):
                pred_total = total_map(pred[i]) / target_scale
                label_total = total_map(label[i])
                map_metrics = scale_metrics(evaluate_regression(label_total, pred_total), metric_scale)
                for key, value in map_metrics.items():
                    sums[key] += value
                row = {"name": name, **{f"map_{k}": v for k, v in map_metrics.items()}}

                if "instance_count_path" in batch and "instance_ir_drop_path" in batch:
                    count_path = batch["instance_count_path"][i]
                    ir_path = batch["instance_ir_drop_path"][i]
                    pixel_counts = np.load(count_path).astype(int)
                    resized_pred = resize_map(pred_total, pixel_counts.shape)
                    pred_instances = np.repeat(resized_pred.reshape(-1), pixel_counts.reshape(-1))
                    target_instances = np.load(ir_path)
                    n = min(len(pred_instances), len(target_instances))
                    if n > 0:
                        inst_metrics = scale_metrics(
                            evaluate_regression(target_instances[:n], pred_instances[:n]), metric_scale
                        )
                        for key, value in inst_metrics.items():
                            instance_sums[key] += value
                        instance_count += 1
                        row.update({f"instance_{k}": v for k, v in inst_metrics.items()})

                if args.save_pred:
                    np.save(os.path.join(save_dir, f"{name}_pred.npy"), pred[i].transpose(1, 2, 0))
                rows.append(row)
                row["inference_seconds"] = batch_seconds / max(len(batch["name"]), 1)
                count += 1

    metric_path = os.path.join(save_dir, "metrics.csv")
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with open(metric_path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    summary = {f"map_{key}": value / max(count, 1) for key, value in sums.items()}
    if instance_count:
        summary.update({f"instance_{key}": value / instance_count for key, value in instance_sums.items()})
    summary["inference_seconds"] = inference_seconds
    summary["inference_ms_per_sample"] = 1000.0 * inference_seconds / max(count, 1)
    summary["metric_scale"] = metric_scale
    with open(os.path.join(save_dir, "summary.json"), "w") as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps(summary, indent=2))
    print(f"metrics saved to {Path(metric_path).resolve()}")


if __name__ == "__main__":
    main()
