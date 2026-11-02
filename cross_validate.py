import argparse
import csv
import json
import random
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np


def read_rows(path: Path) -> List[List[str]]:
    rows: List[List[str]] = []
    with open(path, "r", newline="") as handle:
        for row in csv.reader(handle):
            if not row or not row[0].strip() or row[0].strip().startswith("#"):
                continue
            if row[0].strip().lower() in {"feature", "feature_path"}:
                continue
            rows.append([cell.strip() for cell in row])
    if not rows:
        raise ValueError(f"no samples found in {path}")
    return rows


def write_rows(path: Path, rows: Sequence[Sequence[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as handle:
        csv.writer(handle).writerows(rows)


def make_folds(rows: Sequence[List[str]], folds: int, seed: int, val_ratio: float, stratify_real: bool = False):
    if folds < 2 or folds > len(rows):
        raise ValueError(f"folds must be in [2, {len(rows)}]")
    if not 0.0 <= val_ratio < 1.0:
        raise ValueError("val_ratio must be in [0, 1)")
    indices = list(range(len(rows)))
    rng = random.Random(seed)
    rng.shuffle(indices)
    real_indices = [index for index in indices if any("realdesign" in cell.lower() for cell in rows[index])]
    real_set = set(real_indices)
    fake_indices = [index for index in indices if index not in real_set]
    if stratify_real and real_indices and fake_indices:
        real_folds = np.array_split(real_indices, folds)
        fake_folds = np.array_split(fake_indices, folds)
        test_folds = [list(real_folds[index]) + list(fake_folds[index]) for index in range(folds)]
        for fold in test_folds:
            rng.shuffle(fold)
    else:
        test_folds = [list(part) for part in np.array_split(indices, folds)]
    result = []
    for fold_index, test_indices in enumerate(test_folds):
        test_set = set(test_indices)
        remaining = [index for index in indices if index not in test_set]
        fold_rng = random.Random(seed + fold_index + 1)
        fold_rng.shuffle(remaining)
        val_count = max(1, int(round(len(remaining) * val_ratio))) if val_ratio > 0 else 0
        val_count = min(val_count, max(len(remaining) - 1, 0))
        val_indices = remaining[:val_count]
        train_indices = remaining[val_count:]
        result.append(
            {
                "train": [rows[index] for index in train_indices],
                "val": [rows[index] for index in val_indices],
                "test": [rows[index] for index in test_indices],
            }
        )
    return result


def run_command(command: Sequence[str], cwd: Path) -> None:
    print("running:", " ".join(command), flush=True)
    subprocess.run(list(command), cwd=cwd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare or run K-fold PGAU experiments.")
    parser.add_argument("--csv", required=True, help="CSV containing every labeled sample.")
    parser.add_argument("--config", required=True, help="Base training JSON config.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--folds", type=int, default=6)
    parser.add_argument("--val-ratio", type=float, default=0.1, help="Validation fraction within each training fold.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-folds", type=int, default=None, help="Run/prepare only the first N folds (smoke tests).")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument(
        "--stratify-real",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Distribute realdesign and fakedesign samples across folds when both exist.",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    with open(args.config, "r") as handle:
        base_config: Dict = json.load(handle)

    folds = make_folds(read_rows(Path(args.csv)), args.folds, args.seed, args.val_ratio, args.stratify_real)
    selected_folds = folds[: args.max_folds] if args.max_folds is not None else folds
    summaries = []
    manifest = []
    for fold_index, split in enumerate(selected_folds):
        fold_dir = output / f"fold_{fold_index}"
        split_dir = fold_dir / "splits"
        for split_name, rows in split.items():
            write_rows(split_dir / f"{split_name}.csv", rows)

        run_dir = fold_dir / "run"
        train_config = dict(base_config)
        train_config.update(
            {
                "train_csv": str(split_dir / "train.csv"),
                "val_csv": str(split_dir / "val.csv"),
                "run_dir": str(run_dir),
                "run_name": f"fold_{fold_index}",
                "seed": args.seed + fold_index,
                "resume": None,
            }
        )
        train_config_path = fold_dir / "train_config.json"
        with open(train_config_path, "w") as handle:
            json.dump(train_config, handle, indent=2)

        test_dir = fold_dir / "test_result"
        test_config = {
            key: train_config.get(key)
            for key in (
                "batch_size",
                "num_workers",
                "in_channels",
                "out_channels",
                "base_channels",
                "up_mode",
                "final_activation",
                "model_type",
                "use_attention",
                "target_scale",
                "metric_scale",
                "cpu",
                "gpu",
                "amp",
            )
            if key in train_config
        }
        test_config.update(
            {
                "checkpoint": str(run_dir / "best.pth"),
                "test_csv": str(split_dir / "test.csv"),
                "save_dir": str(test_dir),
                "save_pred": False,
            }
        )
        test_config_path = fold_dir / "test_config.json"
        with open(test_config_path, "w") as handle:
            json.dump(test_config, handle, indent=2)
        manifest.append({"fold": fold_index, **{key: len(value) for key, value in split.items()}})

        if not args.prepare_only:
            run_command([sys.executable, "train.py", "--config", str(train_config_path)], project_root)
            run_command([sys.executable, "test.py", "--config", str(test_config_path)], project_root)
            with open(test_dir / "summary.json", "r") as handle:
                summary = json.load(handle)
            summaries.append({"fold": fold_index, **summary})

    with open(output / "fold_manifest.json", "w") as handle:
        json.dump(manifest, handle, indent=2)
    if summaries:
        numeric_keys = sorted(
            key for key, value in summaries[0].items() if key != "fold" and isinstance(value, (int, float))
        )
        aggregate = {
            key: {
                "mean": float(np.mean([row[key] for row in summaries])),
                "std": float(np.std([row[key] for row in summaries])),
            }
            for key in numeric_keys
        }
        with open(output / "cross_validation_summary.json", "w") as handle:
            json.dump({"folds": summaries, "aggregate": aggregate}, handle, indent=2)
    print(f"prepared {len(selected_folds)} folds under {output}")


if __name__ == "__main__":
    main()
