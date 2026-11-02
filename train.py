import argparse
import json
import math
import os
import random
import sys
from pathlib import Path
from typing import Dict, Iterable, Optional

import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from torch import autocast
try:
    from torch.amp import GradScaler
except ImportError:  # PyTorch 2.0 compatibility
    from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets.npy_dataset import NpyMapDataset, _load_npy, to_chw
from models.build_model import build_model
from utils.logger import build_logger, timestamped_dir
from utils.losses import LDSWeightTable, build_loss
from utils.metrics import evaluate_regression, scale_metrics


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/pgau_circuitnet.json", help="JSON config file.")
    parser.add_argument("--train-csv", default="processed/maps/train.csv", help="CSV with feature,label rows.")
    parser.add_argument("--val-csv", default="processed/maps/val.csv", help="Optional validation CSV.")
    parser.add_argument("--dataroot", default=None, help="Prefix for relative paths in CSV files.")
    parser.add_argument("--save-dir", default="runs/pgau", help="Directory for logs and checkpoints.")
    parser.add_argument("--run-name", default="pgau_circuitnet", help="Run directory prefix.")
    parser.add_argument("--run-dir", default=None, help="Exact run directory (used by cross-validation).")
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=0.0002)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--save-every", type=int, default=50)

    parser.add_argument("--in-channels", type=int, default=11)
    parser.add_argument("--out-channels", type=int, default=2)
    parser.add_argument("--base-channels", type=int, default=64)
    parser.add_argument("--up-mode", default="bilinear", choices=["bilinear", "nearest", "transpose"])
    parser.add_argument("--final-activation", default=None, choices=[None, "none", "sigmoid", "relu"])
    parser.add_argument("--model-type", default="pgau")
    parser.add_argument("--use-attention", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--pretrained", default=None)
    parser.add_argument("--resume", default=None)

    parser.add_argument("--loss", default="lds_l1", choices=["l1", "mse", "lds_l1", "lds_mse"])
    parser.add_argument("--loss-weight", type=float, default=1.0)
    parser.add_argument("--use-lds", action="store_true", help="Force LDS when --loss is l1/mse.")
    parser.add_argument("--lds-bin-width", type=float, default=0.0001)
    parser.add_argument("--lds-kernel-size", type=int, default=5)
    parser.add_argument("--lds-sigma", type=float, default=2.0)
    parser.add_argument("--target-scale", type=float, default=1.0, help="Scale labels for optimization; metrics remain physical.")
    parser.add_argument("--metric-scale", type=float, default=1.0, help="Display/log metric unit scale, e.g. 1000 for mV.")
    parser.add_argument("--selection-metric", default="mae", help="Validation metric used for best.pth.")

    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--gpu", default=None, help="GPU id. Defaults to cuda:0 when CUDA is available.")
    parser.add_argument("--amp", action="store_true", default=False, help="Mixed precision training on CUDA.")
    return parser


def _explicit_cli_dests(parser: argparse.ArgumentParser, argv) -> set:
    explicit = set()
    for action in parser._actions:
        for option in action.option_strings:
            if option in argv or any(token.startswith(option + "=") for token in argv):
                explicit.add(action.dest)
    return explicit


def parse_args():
    parser = _build_parser()
    args = parser.parse_args()
    explicit = _explicit_cli_dests(parser, sys.argv[1:])

    with open(args.config, "r") as handle:
        config = json.load(handle)
    for key, value in config.items():
        if hasattr(args, key) and key not in explicit:
            setattr(args, key, value)
    return args


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(args) -> torch.device:
    if args.cpu or not torch.cuda.is_available():
        return torch.device("cpu")
    if args.gpu is not None:
        return torch.device("cuda", int(args.gpu))
    return torch.device("cuda", 0)


def infer_channels(dataset: NpyMapDataset, args) -> None:
    sample = dataset[0]
    if args.in_channels is None:
        args.in_channels = int(sample["feature"].shape[0])
    if args.out_channels is None:
        args.out_channels = int(sample["label"].shape[0])


def make_loader(dataset: NpyMapDataset, args, shuffle: bool) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available() and not args.cpu,
        drop_last=False,
    )


def iter_label_arrays(dataset: NpyMapDataset, target_scale: float = 1.0) -> Iterable[np.ndarray]:
    for sample in dataset.samples:
        label = to_chw(_load_npy(sample.label_path))
        if label.shape[0] > 1:
            label = label.sum(axis=0, keepdims=True)
        yield label * float(target_scale)


def build_criterion(args, train_dataset: NpyMapDataset, run_dir: str):
    loss_name = args.loss
    if args.use_lds and not loss_name.startswith("lds"):
        loss_name = "lds_l1" if loss_name == "l1" else "lds_mse"

    lds_table = None
    if loss_name.startswith("lds"):
        lds_table = LDSWeightTable.from_arrays(
            iter_label_arrays(train_dataset, target_scale=args.target_scale),
            bin_width=args.lds_bin_width * args.target_scale,
            kernel_size=args.lds_kernel_size,
            sigma=args.lds_sigma,
        )
        lds_table.save(os.path.join(run_dir, "lds_weights.npz"))
    return build_loss(loss_name, lds_table=lds_table, loss_weight=args.loss_weight)


def amp_enabled(args, device: torch.device) -> bool:
    return bool(args.amp and device.type == "cuda")


def load_checkpoint(
    path: str,
    model: nn.Module,
    optimizer=None,
    scheduler=None,
    device: torch.device = torch.device("cpu"),
    total_epochs: Optional[int] = None,
):
    checkpoint = torch.load(path, map_location=device)
    state_dict = checkpoint.get("model", checkpoint.get("state_dict", checkpoint))
    model.load_state_dict(state_dict, strict=True)
    if optimizer is not None and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
    if scheduler is not None and "scheduler" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler"])
    epoch = int(checkpoint.get("epoch", 0))
    if scheduler is not None and total_epochs is not None and hasattr(scheduler, "T_max"):
        new_t_max = max(int(total_epochs), 1)
        if int(scheduler.T_max) != new_t_max:
            scheduler.T_max = new_t_max
            scheduler.last_epoch = epoch
            eta_min = float(getattr(scheduler, "eta_min", 0.0))
            resumed_lrs = [
                eta_min + (float(base_lr) - eta_min) * (1.0 + math.cos(math.pi * epoch / new_t_max)) / 2.0
                for base_lr in scheduler.base_lrs
            ]
            for group, lr in zip(optimizer.param_groups, resumed_lrs):
                group["lr"] = lr
            scheduler._last_lr = resumed_lrs
    return epoch, float(checkpoint.get("best_mae", float("inf")))


def total_ir_drop(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.sum(dim=1) if tensor.ndim == 4 and tensor.shape[1] > 1 else tensor[:, 0]


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    use_amp: bool = False,
    target_scale: float = 1.0,
    metric_scale: float = 1.0,
) -> Dict[str, float]:
    model.eval()
    totals = {"mae": 0.0, "mse": 0.0, "rmse": 0.0, "cc": 0.0, "mirde": 0.0}
    instance_totals = {"mae": 0.0, "mse": 0.0, "rmse": 0.0, "cc": 0.0, "mirde": 0.0}
    instance_count = 0
    count = 0
    for batch in loader:
        feature = batch["feature"].to(device, non_blocking=True)
        label = batch["label"].to(device, non_blocking=True)
        with autocast(device_type=device.type, enabled=use_amp and device.type == "cuda"):
            pred = model(feature)
        pred_total = (total_ir_drop(pred) / target_scale).detach().cpu().numpy()
        label_total = total_ir_drop(label).detach().cpu().numpy()
        for i in range(pred_total.shape[0]):
            result = scale_metrics(evaluate_regression(label_total[i], pred_total[i]), metric_scale)
            for key, value in result.items():
                totals[key] += value
            count += 1
            if "instance_count_path" in batch and "instance_ir_drop_path" in batch:
                pixel_counts = np.load(batch["instance_count_path"][i]).astype(int)
                pred_tensor = torch.from_numpy(pred_total[i][None, None].astype(np.float32))
                resized = torch.nn.functional.interpolate(
                    pred_tensor, size=pixel_counts.shape, mode="bilinear", align_corners=False
                )[0, 0].numpy()
                pred_instances = np.repeat(resized.reshape(-1), pixel_counts.reshape(-1))
                target_instances = np.load(batch["instance_ir_drop_path"][i])
                n = min(len(pred_instances), len(target_instances))
                if n:
                    instance_result = scale_metrics(
                        evaluate_regression(target_instances[:n], pred_instances[:n]), metric_scale
                    )
                    for key, value in instance_result.items():
                        instance_totals[key] += value
                    instance_count += 1
    metrics = {key: value / max(count, 1) for key, value in totals.items()}
    if instance_count:
        metrics.update({f"instance_{key}": value / instance_count for key, value in instance_totals.items()})
    return metrics


def save_checkpoint(path: str, model: nn.Module, optimizer, scheduler, epoch: int, best_mae: float, args) -> None:
    torch.save(
        {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "best_mae": best_mae,
            "config": vars(args),
        },
        path,
    )


def main() -> None:
    args = parse_args()
    if args.target_scale <= 0 or args.metric_scale <= 0:
        raise ValueError("target_scale and metric_scale must be positive")
    set_seed(args.seed)
    if args.run_dir:
        run_dir = args.run_dir
    elif args.resume:
        run_dir = os.path.dirname(os.path.abspath(args.resume))
    else:
        run_dir = timestamped_dir(args.save_dir, args.run_name)
    os.makedirs(run_dir, exist_ok=True)
    logger = build_logger(run_dir, filename="train.log")
    with open(os.path.join(run_dir, "config.json"), "w") as handle:
        json.dump(vars(args), handle, indent=2)

    train_dataset = NpyMapDataset(args.train_csv, dataroot=args.dataroot)
    val_dataset: Optional[NpyMapDataset] = NpyMapDataset(args.val_csv, dataroot=args.dataroot) if args.val_csv else None
    infer_channels(train_dataset, args)

    device = get_device(args)
    logger.info(f"device={device}, train_samples={len(train_dataset)}, val_samples={len(val_dataset) if val_dataset else 0}")
    logger.info(f"in_channels={args.in_channels}, out_channels={args.out_channels}")

    train_loader = make_loader(train_dataset, args, shuffle=True)
    val_loader = make_loader(val_dataset, args, shuffle=False) if val_dataset else None

    model = build_model(
        {
            "model_type": args.model_type,
            "in_channels": args.in_channels,
            "out_channels": args.out_channels,
            "base_channels": args.base_channels,
            "up_mode": args.up_mode,
            "final_activation": args.final_activation,
            "pretrained": args.pretrained,
            "use_attention": args.use_attention,
        }
    ).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1), eta_min=1e-7)
    criterion = build_criterion(args, train_dataset, run_dir)
    try:
        scaler = GradScaler("cuda", enabled=amp_enabled(args, device))
    except TypeError:  # torch.cuda.amp.GradScaler in PyTorch 2.0
        scaler = GradScaler(enabled=amp_enabled(args, device))

    start_epoch = 0
    best_mae = float("inf")
    if args.resume:
        start_epoch, best_mae = load_checkpoint(
            args.resume, model, optimizer, scheduler, device=device, total_epochs=args.epochs
        )
        logger.info(f"resumed from {args.resume} at epoch {start_epoch}")

    for epoch in range(start_epoch + 1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        progress = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}", ncols=100)
        for batch in progress:
            feature = batch["feature"].to(device, non_blocking=True)
            label = batch["label"].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with autocast(device_type=device.type, enabled=amp_enabled(args, device)):
                pred = model(feature)
                loss = criterion(pred, label * args.target_scale)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += float(loss.detach().cpu())
            progress.set_postfix(loss=f"{epoch_loss / (progress.n or 1):.6f}")

        scheduler.step()
        avg_loss = epoch_loss / max(len(train_loader), 1)
        message = f"epoch={epoch} train_loss={avg_loss:.6f} lr={scheduler.get_last_lr()[0]:.3e}"

        if val_loader is not None:
            val_metrics = evaluate(
                model,
                val_loader,
                device,
                use_amp=args.amp,
                target_scale=args.target_scale,
                metric_scale=args.metric_scale,
            )
            message += " " + " ".join(f"val_{k}={v:.6f}" for k, v in val_metrics.items())
            if args.selection_metric not in val_metrics:
                raise KeyError(f"selection metric {args.selection_metric!r} not found in {sorted(val_metrics)}")
            if val_metrics[args.selection_metric] < best_mae:
                best_mae = val_metrics[args.selection_metric]
                save_checkpoint(os.path.join(run_dir, "best.pth"), model, optimizer, scheduler, epoch, best_mae, args)
        else:
            best_mae = avg_loss
            save_checkpoint(os.path.join(run_dir, "best.pth"), model, optimizer, scheduler, epoch, best_mae, args)

        logger.info(message)
        save_checkpoint(os.path.join(run_dir, "last.pth"), model, optimizer, scheduler, epoch, best_mae, args)
        if args.save_every > 0 and epoch % args.save_every == 0:
            save_checkpoint(os.path.join(run_dir, f"epoch_{epoch}.pth"), model, optimizer, scheduler, epoch, best_mae, args)

    logger.info(f"finished. best_mae={best_mae:.6f}, run_dir={Path(run_dir).resolve()}")


if __name__ == "__main__":
    main()
