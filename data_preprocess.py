import argparse
import csv
import json
import os
import random
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm


DEFAULT_FEATURE_COLUMNS = [
    "loop_r",
    "vdd_r",
    "gnd_r",
    "min_res_vdd",
    "min_res_vss",
    "freq",
    "toggle_rate",
    "total_power",
    "composite_power",
    "neighbor_power",
    "neighbor_count",
]

DEFAULT_X_STEP = 1.44
DEFAULT_Y_STEP = 1.152
DEFAULT_PADDING = 10.0

POWER_COLUMNS = [
    "inst_name",
    "cell_name",
    "freq",
    "toggle_rate",
    "leakage_power",
    "switching_power",
    "internal_power",
    "total_power",
    "library",
    "PG_pin",
    "PG_volt",
    "powerDomain",
    "bbox",
]


def minmax_norm(array: np.ndarray) -> np.ndarray:
    array = np.asarray(array, dtype=np.float32)
    finite = np.isfinite(array)
    if not finite.any():
        return np.zeros_like(array, dtype=np.float32)
    lo = float(array[finite].min())
    hi = float(array[finite].max())
    if hi <= lo:
        return np.zeros_like(array, dtype=np.float32)
    out = (array - lo) / (hi - lo)
    out[~finite] = 0
    return out.astype(np.float32)


def parse_bbox(power: pd.DataFrame) -> pd.DataFrame:
    power = power.copy()
    power["bbox"] = power["bbox"].astype(str).str.replace(r"[\[\]]", "", regex=True)
    power[["bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2"]] = power["bbox"].str.split(",", expand=True)
    for col in ["bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2"]:
        power[col] = pd.to_numeric(power[col], errors="coerce")
    power["center_x"] = power[["bbox_x1", "bbox_x2"]].mean(axis=1)
    power["center_y"] = power[["bbox_y1", "bbox_y2"]].mean(axis=1)
    return power.drop(columns=["bbox"])


def find_power_report(design_dir: Path) -> Path:
    candidates = sorted(design_dir.glob("*.inst.power.rpt")) + sorted(design_dir.glob("*.inst.power.rpt.gz"))
    if not candidates:
        raise FileNotFoundError(f"no *.inst.power.rpt found in {design_dir}")
    return candidates[0]


def find_report(design_dir: Path, name: str) -> Path:
    path = design_dir / name
    if path.exists():
        return path
    compressed = design_dir / f"{name}.gz"
    if compressed.exists():
        return compressed
    raise FileNotFoundError(f"missing {name} in {design_dir}")


def read_power_report(path: Path) -> pd.DataFrame:
    power = pd.read_csv(path, sep=r"\s+", skiprows=2, names=POWER_COLUMNS, low_memory=False)
    power = parse_bbox(power)
    power = power.drop_duplicates(subset="inst_name", keep="first")
    return power


def read_eff_res(path: Path) -> pd.DataFrame:
    columns = [
        "loop_r",
        "vdd_r",
        "gnd_r",
        "vdd_x",
        "vdd_y",
        "vdd_layer",
        "vdd_net",
        "vdd_pin",
        "vss_x",
        "vss_y",
        "vss_layer",
        "vss_net",
        "vss_pin",
        "inst_name",
    ]
    eff = pd.read_csv(path, sep=r"\s+", skiprows=1, names=columns, low_memory=False)
    eff.replace(to_replace=r"[\(\)]", value="", regex=True, inplace=True)
    return eff


def read_min_res(path: Path) -> pd.DataFrame:
    min_res = pd.read_csv(path, sep=r"\s+", skiprows=1, names=["min_res", "net_name", "inst_name"], low_memory=False)
    vdd = min_res[min_res["net_name"].astype(str).str.upper().eq("VDD")][["inst_name", "min_res"]]
    vss = min_res[min_res["net_name"].astype(str).str.upper().isin(["VSS", "GND"])][["inst_name", "min_res"]]
    data = pd.merge(vdd.rename(columns={"min_res": "min_res_vdd"}), vss.rename(columns={"min_res": "min_res_vss"}), on="inst_name", how="outer")
    return data


def read_static_ir(path: Path) -> pd.DataFrame:
    columns = ["inst_vdd", "vdd_drop", "gnd_bounce", "ideal_vdd", "pwr_net", "location", "inst_name"]
    ir = pd.read_csv(path, sep=r"\s+", skiprows=1, names=columns, low_memory=False)
    ir[["location_x", "location_y"]] = ir["location"].astype(str).str.split(",", expand=True)
    return ir.drop(columns=["location"])


def coerce_numeric(data: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    data = data.copy()
    for col in columns:
        if col in data.columns:
            data[col] = pd.to_numeric(data[col], errors="coerce")
    return data


def add_augmented_features(data: pd.DataFrame, neighbor_radius: float = 5.0) -> pd.DataFrame:
    data = data.copy()
    for col in ["internal_power", "switching_power", "toggle_rate", "leakage_power", "total_power"]:
        data[col] = pd.to_numeric(data[col], errors="coerce").fillna(0.0)
    data["composite_power"] = (data["internal_power"] + data["switching_power"]) * data["toggle_rate"] + data["leakage_power"]

    coords = data[["center_x", "center_y"]].to_numpy(dtype=np.float64)
    power = data["total_power"].to_numpy(dtype=np.float64)
    neighbor_power = np.zeros(len(data), dtype=np.float64)
    neighbor_count = np.zeros(len(data), dtype=np.float64)
    valid = np.isfinite(coords).all(axis=1)
    if valid.any():
        try:
            from scipy.spatial import cKDTree

            tree = cKDTree(coords[valid])
            valid_indices = np.flatnonzero(valid)
            for local_idx, original_idx in enumerate(valid_indices):
                local_neighbors = tree.query_ball_point(coords[original_idx], r=neighbor_radius)
                original_neighbors = valid_indices[local_neighbors]
                original_neighbors = original_neighbors[original_neighbors != original_idx]
                neighbor_count[original_idx] = len(original_neighbors)
                if len(original_neighbors):
                    neighbor_power[original_idx] = float(np.mean(power[original_neighbors]))
        except Exception:
            valid_indices = np.flatnonzero(valid)
            for original_idx in valid_indices:
                dist = np.linalg.norm(coords[valid_indices] - coords[original_idx], axis=1)
                original_neighbors = valid_indices[(dist <= neighbor_radius) & (valid_indices != original_idx)]
                neighbor_count[original_idx] = len(original_neighbors)
                if len(original_neighbors):
                    neighbor_power[original_idx] = float(np.mean(power[original_neighbors]))
    data["neighbor_power"] = neighbor_power
    data["neighbor_count"] = neighbor_count
    return data


def align_one_design(
    design_dir: Path,
    output_dir: Path,
    neighbor_radius: float = 5.0,
    final_test: bool = False,
    output_name: Optional[str] = None,
) -> Path:
    power = read_power_report(find_power_report(design_dir))
    eff = read_eff_res(find_report(design_dir, "eff_res.rpt"))
    min_res = read_min_res(find_report(design_dir, "min_path_res.rpt"))

    static_ir_path = None
    try:
        static_ir_path = find_report(design_dir, "static_ir")
    except FileNotFoundError:
        static_ir_path = None

    if not final_test and static_ir_path is not None:
        data = read_static_ir(static_ir_path)
        data = pd.merge(data, eff, on="inst_name", how="left")
        data = pd.merge(data, power, on="inst_name", how="left")
    else:
        data = pd.merge(power, eff, on="inst_name", how="left")
    data = pd.merge(data, min_res, on="inst_name", how="left")

    numeric_columns = [
        "loop_r",
        "vdd_r",
        "gnd_r",
        "vdd_x",
        "vdd_y",
        "vss_x",
        "vss_y",
        "bbox_x1",
        "bbox_y1",
        "bbox_x2",
        "bbox_y2",
        "center_x",
        "center_y",
        "freq",
        "toggle_rate",
        "leakage_power",
        "switching_power",
        "internal_power",
        "total_power",
        "min_res_vdd",
        "min_res_vss",
        "vdd_drop",
        "gnd_bounce",
        "location_x",
        "location_y",
    ]
    data = coerce_numeric(data, numeric_columns)
    required = [
        "center_x",
        "center_y",
        "loop_r",
        "vdd_r",
        "gnd_r",
        "min_res_vdd",
        "min_res_vss",
        "freq",
        "toggle_rate",
        "total_power",
    ]
    if not final_test and {"vdd_drop", "gnd_bounce", "location_x", "location_y"}.issubset(data.columns):
        required.extend(["vdd_drop", "gnd_bounce", "location_x", "location_y"])
    data = data.dropna(subset=[col for col in required if col in data.columns])
    data = add_augmented_features(data, neighbor_radius=neighbor_radius)

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{output_name or design_dir.name}.csv"
    data.to_csv(out_path, index=False)
    return out_path


def find_design_dirs(root: Path) -> List[Path]:
    design_dirs = []
    eff_paths = list(root.rglob("eff_res.rpt")) + list(root.rglob("eff_res.rpt.gz"))
    for eff_path in eff_paths:
        design_dir = eff_path.parent
        has_min_res = (design_dir / "min_path_res.rpt").exists() or (design_dir / "min_path_res.rpt.gz").exists()
        has_power = list(design_dir.glob("*.inst.power.rpt")) or list(design_dir.glob("*.inst.power.rpt.gz"))
        if has_min_res and has_power:
            design_dirs.append(design_dir)
    return sorted(set(design_dirs))


def physical_indices(
    x,
    y,
    x_step: float = DEFAULT_X_STEP,
    y_step: float = DEFAULT_Y_STEP,
    padding: float = DEFAULT_PADDING,
    shape: Optional[Tuple[int, int]] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = pd.to_numeric(pd.Series(x), errors="coerce").to_numpy(dtype=np.float64)
    y = pd.to_numeric(pd.Series(y), errors="coerce").to_numpy(dtype=np.float64)
    valid = np.isfinite(x) & np.isfinite(y)
    rows = np.zeros_like(x, dtype=np.int64)
    cols = np.zeros_like(y, dtype=np.int64)
    # CircuitNet coordinates follow the released preprocessing convention:
    # non-boundary locations are assigned to the next physical grid index.
    rows[valid] = np.ceil((x[valid] - padding) / x_step).astype(np.int64)
    cols[valid] = np.ceil((y[valid] - padding) / y_step).astype(np.int64)
    rows = np.maximum(rows, 0)
    cols = np.maximum(cols, 0)
    if shape is not None:
        rows = np.clip(rows, 0, shape[0] - 1)
        cols = np.clip(cols, 0, shape[1] - 1)
    return rows, cols, valid


def coordinate_pairs_for_extent(data: pd.DataFrame, final_test: bool) -> List[Tuple[pd.Series, pd.Series]]:
    pairs = [(data["center_x"], data["center_y"])]
    if {"vdd_x", "vdd_y"}.issubset(data.columns):
        pairs.append((data["vdd_x"], data["vdd_y"]))
    if {"vss_x", "vss_y"}.issubset(data.columns):
        pairs.append((data["vss_x"], data["vss_y"]))
    if not final_test and {"location_x", "location_y"}.issubset(data.columns):
        pairs.append((data["location_x"], data["location_y"]))
    return pairs


def infer_raw_shape(
    data: pd.DataFrame,
    final_test: bool,
    x_step: float,
    y_step: float,
    padding: float,
) -> Tuple[int, int]:
    max_row = 0
    max_col = 0
    found = False
    for x, y in coordinate_pairs_for_extent(data, final_test):
        rows, cols, valid = physical_indices(x, y, x_step=x_step, y_step=y_step, padding=padding)
        if valid.any():
            max_row = max(max_row, int(rows[valid].max()))
            max_col = max(max_col, int(cols[valid].max()))
            found = True
    if not found:
        raise ValueError("no finite coordinates found")
    return max_row + 1, max_col + 1


def feature_coordinates(data: pd.DataFrame, column: str) -> Tuple[pd.Series, pd.Series]:
    # Every selected CircuitNet feature is an instance-level attribute.  Using
    # VDD/VSS pin coordinates for resistance channels misaligns those channels
    # with power, frequency and label maps (and can move a value to a different
    # grid cell).  The instance centre agrees with static_ir locations to
    # floating-point placement precision in the released data.
    return data["center_x"], data["center_y"]


def read_aligned_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in {".ftr", ".feather"}:
        return pd.read_feather(path)
    return pd.read_csv(path)


def table_sample_name(path: Path) -> str:
    if path.stem in {"data_alignment", "aligned"}:
        return path.parent.name
    return path.stem


def resize_map(array: np.ndarray, out_shape: Tuple[int, int], order: int = 1) -> np.ndarray:
    if tuple(array.shape) == tuple(out_shape):
        return array.astype(np.float32)
    from scipy import ndimage

    zoom = (out_shape[0] / array.shape[0], out_shape[1] / array.shape[1])
    return ndimage.zoom(array, zoom, order=order).astype(np.float32)


def aggregate_feature_map(
    data: pd.DataFrame,
    column: str,
    raw_shape: Tuple[int, int],
    out_shape: Tuple[int, int],
    x_step: float,
    y_step: float,
    padding: float,
    resize_order: int,
) -> np.ndarray:
    values = pd.to_numeric(data[column], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
    x, y = feature_coordinates(data, column)
    rows, cols, valid = physical_indices(x, y, x_step=x_step, y_step=y_step, padding=padding, shape=raw_shape)
    out = np.zeros(raw_shape, dtype=np.float32)
    np.add.at(out, (rows[valid], cols[valid]), values[valid])
    return minmax_norm(resize_map(out, out_shape, order=resize_order))


def build_maps_for_table(
    table_path: Path,
    output_dir: Path,
    feature_columns: Sequence[str] = DEFAULT_FEATURE_COLUMNS,
    image_size: int = 256,
    label_mode: str = "sum",
    final_test: bool = False,
    x_step: float = DEFAULT_X_STEP,
    y_step: float = DEFAULT_Y_STEP,
    padding: float = DEFAULT_PADDING,
    resize_order: int = 1,
) -> Dict[str, str]:
    data = read_aligned_table(table_path)
    if (
        "composite_power" not in data.columns
        or "neighbor_power" not in data.columns
        or "neighbor_count" not in data.columns
    ):
        data = add_augmented_features(data)
    missing = [col for col in feature_columns if col not in data.columns]
    if missing:
        raise KeyError(f"{table_path} missing feature columns: {missing}")

    out_shape = (image_size, image_size)
    raw_shape = infer_raw_shape(data, final_test=final_test, x_step=x_step, y_step=y_step, padding=padding)
    feature_maps = [
        aggregate_feature_map(
            data,
            col,
            raw_shape,
            out_shape,
            x_step=x_step,
            y_step=y_step,
            padding=padding,
            resize_order=resize_order,
        )
        for col in feature_columns
    ]
    feature = np.stack(feature_maps, axis=-1).astype(np.float32)

    feature_dir = output_dir / "feature"
    label_dir = output_dir / "label"
    info_dir = output_dir / "instance_info"
    feature_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)
    info_dir.mkdir(parents=True, exist_ok=True)

    name = table_sample_name(table_path)
    feature_path = feature_dir / f"{name}.npy"
    np.save(feature_path, feature)
    result = {"feature": str(feature_path)}

    if not final_test and {"vdd_drop", "gnd_bounce", "location_x", "location_y"}.issubset(data.columns):
        rows, cols, valid = physical_indices(
            data["location_x"],
            data["location_y"],
            x_step=x_step,
            y_step=y_step,
            padding=padding,
            shape=raw_shape,
        )
        values_vdd = pd.to_numeric(data["vdd_drop"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
        values_gnd = pd.to_numeric(data["gnd_bounce"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
        if label_mode == "sum":
            raw_label = np.zeros((raw_shape[0], raw_shape[1], 1), dtype=np.float32)
            np.maximum.at(raw_label[:, :, 0], (rows[valid], cols[valid]), (values_vdd + values_gnd)[valid])
        else:
            raw_label = np.zeros((raw_shape[0], raw_shape[1], 2), dtype=np.float32)
            np.maximum.at(raw_label[:, :, 0], (rows[valid], cols[valid]), values_vdd[valid])
            np.maximum.at(raw_label[:, :, 1], (rows[valid], cols[valid]), values_gnd[valid])
        label = np.stack(
            [
                np.maximum(resize_map(raw_label[:, :, channel], out_shape, order=resize_order), 0.0)
                for channel in range(raw_label.shape[2])
            ],
            axis=-1,
        ).astype(np.float32)
        label_path = label_dir / f"{name}.npy"
        np.save(label_path, label)
        result["label"] = str(label_path)

        cell_lists: List[List[Tuple[str, float]]] = [[] for _ in range(raw_shape[0] * raw_shape[1])]
        names = data.get("inst_name", pd.Series([f"inst_{i}" for i in range(len(data))])).astype(str).to_numpy()
        total_ir = values_vdd + values_gnd
        for row, col, is_valid, inst_name, ir in zip(rows, cols, valid, names, total_ir):
            if is_valid:
                cell_lists[int(row) * raw_shape[1] + int(col)].append((inst_name, float(ir)))
        instance_count = np.array([len(cell) for cell in cell_lists], dtype=np.int32).reshape(raw_shape)
        instance_names = np.array([inst for cell in cell_lists for inst, _ in cell], dtype=object)
        instance_ir = np.array([ir for cell in cell_lists for _, ir in cell], dtype=np.float32)

        count_path = info_dir / f"{name}_instance_count.npy"
        ir_path = info_dir / f"{name}_instance_IR_drop.npy"
        name_path = info_dir / f"{name}_instance_name.npy"
        np.save(count_path, instance_count)
        np.save(ir_path, instance_ir)
        np.save(name_path, instance_names)
        result.update(
            {
                "instance_count": str(count_path),
                "instance_ir_drop": str(ir_path),
                "instance_name": str(name_path),
            }
        )

    meta_dir = output_dir / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)
    meta_path = meta_dir / f"{name}.json"
    with open(meta_path, "w") as handle:
        json.dump(
            {
                "feature_columns": list(feature_columns),
                "image_size": image_size,
                "label_mode": label_mode,
                "raw_shape": list(raw_shape),
                "x_step": x_step,
                "y_step": y_step,
                "padding": padding,
                "resize_order": resize_order,
            },
            handle,
            indent=2,
        )
    return result


def write_split_csv(
    records: List[Dict[str, str]],
    output_dir: Path,
    seed: int,
    train_ratio: float,
    val_ratio: float,
    split_mode: str = "random",
) -> None:
    validate_split_ratios(train_ratio, val_ratio)
    random.Random(seed).shuffle(records)
    n = len(records)
    if split_mode == "transfer":
        test = [
            row
            for row in records
            if Path(row["feature"]).stem.lower().startswith(("nvdla-small", "vortex-small"))
        ]
        train_pool = [row for row in records if row not in test]
        n_val = min(int(len(train_pool) * val_ratio), max(len(train_pool) - 1, 0))
        if len(train_pool) > 1 and val_ratio > 0 and n_val == 0:
            n_val = 1
        train = train_pool[n_val:]
        val = train_pool[:n_val]
        if not train or not test:
            raise ValueError("transfer split requires RISCY/RISCY-FPU/zero-riscy train designs and nvdla/Vortex test designs")
    else:
        n_train = int(n * train_ratio)
        if n > 0 and train_ratio > 0 and n_train == 0:
            n_train = 1
        n_train = min(n_train, n)
        n_val = int(n * val_ratio)
        remaining = max(n - n_train, 0)
        if remaining > 1 and val_ratio > 0 and n_val == 0:
            n_val = 1
        n_val = min(n_val, remaining)
        train = records[:n_train]
        val = records[n_train : n_train + n_val]
        test = records[n_train + n_val :]
    splits = {"all.csv": records, "train.csv": train, "val.csv": val, "test.csv": test}
    for filename, rows in splits.items():
        with open(output_dir / filename, "w", newline="") as handle:
            writer = csv.writer(handle)
            for row in rows:
                if "label" not in row:
                    continue
                writer.writerow(
                    [
                        row["feature"],
                        row["label"],
                        row.get("instance_count", ""),
                        row.get("instance_ir_drop", ""),
                        row.get("instance_name", ""),
                    ]
                )


def discover_iccad_samples(root: Path) -> Dict[str, Path]:
    samples: Dict[str, Path] = {}
    for label_path in sorted(root.glob("*/*_ir_drop.csv")) + sorted(root.glob("*_ir_drop.csv")):
        name = label_path.name[: -len("_ir_drop.csv")]
        sample_dir = label_path.parent
        required = [sample_dir / f"{name}_{suffix}.csv" for suffix in ("current", "eff_dist", "pdn_density", "ir_drop")]
        if all(path.exists() for path in required):
            samples[name] = sample_dir
    return samples


def load_csv_map(path: Path) -> np.ndarray:
    array = np.loadtxt(path, delimiter=",", dtype=np.float32)
    if array.ndim != 2 or not np.isfinite(array).all():
        raise ValueError(f"{path} must contain one finite 2D numeric matrix, got {array.shape}")
    return array


def split_records(records: List[Dict[str, str]], seed: int, train_ratio: float, val_ratio: float) -> Dict[str, List[Dict[str, str]]]:
    validate_split_ratios(train_ratio, val_ratio)
    records = list(records)
    random.Random(seed).shuffle(records)
    n = len(records)
    n_train = min(max(int(n * train_ratio), 1 if n else 0), n)
    remaining = n - n_train
    n_val = min(max(int(n * val_ratio), 1 if remaining > 1 and val_ratio > 0 else 0), remaining)
    return {
        "train": records[:n_train],
        "val": records[n_train : n_train + n_val],
        "test": records[n_train + n_val :],
    }


def validate_split_ratios(train_ratio: float, val_ratio: float) -> None:
    if not 0.0 <= train_ratio <= 1.0:
        raise ValueError("train_ratio must be in [0, 1]")
    if not 0.0 <= val_ratio <= 1.0:
        raise ValueError("val_ratio must be in [0, 1]")
    if train_ratio + val_ratio > 1.0:
        raise ValueError("train_ratio + val_ratio must not exceed 1")


def write_map_csv(path: Path, records: Sequence[Dict[str, str]]) -> None:
    with open(path, "w", newline="") as handle:
        writer = csv.writer(handle)
        for record in records:
            writer.writerow([record["feature"], record["label"]])


def cmd_iccad(args) -> None:
    input_root = Path(args.input)
    output_dir = Path(args.output)
    samples = discover_iccad_samples(input_root)
    names = sorted(samples)
    if args.limit is not None:
        names = names[: args.limit]
    if not names:
        raise FileNotFoundError(f"no complete ICCAD2023 samples found under {input_root}")

    placeholder_records = [{"name": name} for name in names]
    split = split_records(placeholder_records, args.seed, args.train_ratio, args.val_ratio)
    train_names = {record["name"] for record in split["train"]}

    channel_names = ["current", "eff_dist", "pdn_density"]
    channel_min = np.full(len(channel_names), np.inf, dtype=np.float64)
    channel_max = np.full(len(channel_names), -np.inf, dtype=np.float64)
    if args.normalize == "train_minmax":
        for name in tqdm(names, desc="ICCAD train statistics", ncols=100):
            if name not in train_names:
                continue
            for index, channel in enumerate(channel_names):
                array = load_csv_map(samples[name] / f"{name}_{channel}.csv")
                channel_min[index] = min(channel_min[index], float(array.min()))
                channel_max[index] = max(channel_max[index], float(array.max()))

    feature_dir = output_dir / "feature"
    label_dir = output_dir / "label"
    feature_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)
    out_shape = (args.image_size, args.image_size)
    records_by_name: Dict[str, Dict[str, str]] = {}
    sample_meta: Dict[str, Dict[str, object]] = {}
    for name in tqdm(names, desc="ICCAD maps", ncols=100):
        raw_features = [load_csv_map(samples[name] / f"{name}_{channel}.csv") for channel in channel_names]
        label_raw = load_csv_map(samples[name] / f"{name}_ir_drop.csv")
        raw_shape = label_raw.shape
        if any(feature.shape != raw_shape for feature in raw_features):
            raise ValueError(f"{name} feature/label shapes do not agree")

        features = []
        for index, array in enumerate(raw_features):
            array = resize_map(array, out_shape, order=args.resize_order)
            if args.normalize == "per_map":
                array = minmax_norm(array)
            elif args.normalize == "train_minmax":
                scale = channel_max[index] - channel_min[index]
                array = np.zeros_like(array) if scale <= 0 else (array - channel_min[index]) / scale
                array = np.clip(array, 0.0, 1.0).astype(np.float32)
            features.append(array.astype(np.float32))
        feature = np.stack(features, axis=-1)
        label = resize_map(label_raw, out_shape, order=args.resize_order)[..., None].astype(np.float32)
        feature_path = feature_dir / f"{name}.npy"
        label_path = label_dir / f"{name}.npy"
        np.save(feature_path, feature)
        np.save(label_path, label)
        records_by_name[name] = {"feature": str(feature_path), "label": str(label_path)}
        sample_meta[name] = {"source": str(samples[name]), "raw_shape": list(raw_shape)}

    for split_name, placeholders in split.items():
        write_map_csv(output_dir / f"{split_name}.csv", [records_by_name[item["name"]] for item in placeholders])
    write_map_csv(output_dir / "all.csv", [records_by_name[name] for name in names])
    with open(output_dir / "meta.json", "w") as handle:
        json.dump(
            {
                "dataset": "ICCAD2023",
                "feature_columns": channel_names,
                "image_size": args.image_size,
                "normalization": args.normalize,
                "channel_min": channel_min.tolist() if args.normalize == "train_minmax" else None,
                "channel_max": channel_max.tolist() if args.normalize == "train_minmax" else None,
                "split": {key: [item["name"] for item in value] for key, value in split.items()},
                "samples": sample_meta,
            },
            handle,
            indent=2,
        )
    print(f"processed={len(names)} train={len(split['train'])} val={len(split['val'])} test={len(split['test'])}")


def cmd_align(args) -> None:
    input_root = Path(args.input)
    output_dir = Path(args.output)
    if (input_root / "eff_res.rpt").exists() or (input_root / "eff_res.rpt.gz").exists():
        design_dirs = [input_root]
    else:
        design_dirs = find_design_dirs(input_root)
    if not design_dirs:
        raise FileNotFoundError(f"no complete CircuitNet design directories found under {input_root}")
    if getattr(args, "limit", None) is not None:
        design_dirs = design_dirs[: args.limit]
    print(f"found {len(design_dirs)} designs")
    for design_dir in tqdm(design_dirs):
        if design_dir == input_root:
            output_name = design_dir.name
        else:
            output_name = "__".join(design_dir.relative_to(input_root).parts)
        align_one_design(
            design_dir,
            output_dir,
            neighbor_radius=args.neighbor_radius,
            final_test=args.final_test,
            output_name=output_name,
        )


def cmd_make_maps(args) -> None:
    input_path = Path(args.input)
    output_dir = Path(args.output)
    tables = (
        [input_path]
        if input_path.is_file()
        else sorted(list(input_path.rglob("*.csv")) + list(input_path.rglob("*.ftr")) + list(input_path.rglob("*.feather")))
    )
    if getattr(args, "limit", None) is not None:
        tables = tables[: args.limit]
    if not tables:
        raise FileNotFoundError(f"no aligned CSV files found in {input_path}")
    output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    feature_columns = args.feature_columns.split(",") if args.feature_columns else DEFAULT_FEATURE_COLUMNS
    for table in tqdm(tables):
        records.append(
            build_maps_for_table(
                table,
                output_dir,
                feature_columns=feature_columns,
                image_size=args.image_size,
                label_mode=args.label_mode,
                final_test=args.final_test,
                x_step=args.x_step,
                y_step=args.y_step,
                padding=args.padding,
                resize_order=args.resize_order,
            )
        )
    if not args.final_test:
        write_split_csv(
            records,
            output_dir,
            seed=args.seed,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            split_mode=args.split_mode,
        )


def cmd_all(args) -> None:
    work_dir = Path(args.output)
    aligned_dir = work_dir / "aligned"
    map_dir = work_dir / "maps"
    align_args = argparse.Namespace(
        input=args.input,
        output=str(aligned_dir),
        neighbor_radius=args.neighbor_radius,
        final_test=args.final_test,
        limit=args.limit,
    )
    cmd_align(align_args)
    map_args = argparse.Namespace(
        input=str(aligned_dir),
        output=str(map_dir),
        feature_columns=args.feature_columns,
        image_size=args.image_size,
        label_mode=args.label_mode,
        final_test=args.final_test,
        x_step=args.x_step,
        y_step=args.y_step,
        padding=args.padding,
        resize_order=args.resize_order,
        seed=args.seed,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        split_mode=args.split_mode,
        limit=args.limit,
    )
    cmd_make_maps(map_args)


def parse_args():
    parser = argparse.ArgumentParser(description="PGAU preprocessing for CircuitNet/ICCAD-style reports.")
    sub = parser.add_subparsers(dest="command", required=True)

    def common(p):
        p.add_argument("--neighbor-radius", type=float, default=5.0)
        p.add_argument("--final-test", action="store_true", help="Build features without labels.")
        p.add_argument("--limit", type=int, default=None, help="Process only the first N discovered samples.")

    align = sub.add_parser("align", help="Align raw report files into per-design CSV tables.")
    align.add_argument("--input", required=True, help="Raw root or one design directory.")
    align.add_argument("--output", required=True, help="Output aligned CSV directory.")
    common(align)
    align.set_defaults(func=cmd_align)

    maps = sub.add_parser("make-maps", help="Build 256x256 npy feature/label maps from aligned CSV files.")
    maps.add_argument("--input", required=True, help="Aligned CSV file or directory.")
    maps.add_argument("--output", required=True, help="Output map directory.")
    maps.add_argument("--feature-columns", default=",".join(DEFAULT_FEATURE_COLUMNS))
    maps.add_argument("--image-size", type=int, default=256)
    maps.add_argument("--label-mode", choices=["two_channel", "sum"], default="sum")
    maps.add_argument("--x-step", type=float, default=DEFAULT_X_STEP)
    maps.add_argument("--y-step", type=float, default=DEFAULT_Y_STEP)
    maps.add_argument("--padding", type=float, default=DEFAULT_PADDING)
    maps.add_argument("--resize-order", type=int, default=1, choices=[0, 1, 2, 3])
    maps.add_argument("--seed", type=int, default=0)
    maps.add_argument("--train-ratio", type=float, default=0.8)
    maps.add_argument("--val-ratio", type=float, default=0.1)
    maps.add_argument("--split-mode", choices=["random", "transfer"], default="random")
    common(maps)
    maps.set_defaults(func=cmd_make_maps)

    all_cmd = sub.add_parser("all", help="Run align and make-maps.")
    all_cmd.add_argument("--input", required=True)
    all_cmd.add_argument("--output", required=True)
    all_cmd.add_argument("--feature-columns", default=",".join(DEFAULT_FEATURE_COLUMNS))
    all_cmd.add_argument("--image-size", type=int, default=256)
    all_cmd.add_argument("--label-mode", choices=["two_channel", "sum"], default="sum")
    all_cmd.add_argument("--x-step", type=float, default=DEFAULT_X_STEP)
    all_cmd.add_argument("--y-step", type=float, default=DEFAULT_Y_STEP)
    all_cmd.add_argument("--padding", type=float, default=DEFAULT_PADDING)
    all_cmd.add_argument("--resize-order", type=int, default=1, choices=[0, 1, 2, 3])
    all_cmd.add_argument("--seed", type=int, default=0)
    all_cmd.add_argument("--train-ratio", type=float, default=0.8)
    all_cmd.add_argument("--val-ratio", type=float, default=0.1)
    all_cmd.add_argument("--split-mode", choices=["random", "transfer"], default="random")
    common(all_cmd)
    all_cmd.set_defaults(func=cmd_all)

    iccad = sub.add_parser("iccad", help="Build 3-channel maps from ICCAD2023 current/distance/density CSV grids.")
    iccad.add_argument("--input", required=True, help="Root containing fakedesign*/realdesign* directories.")
    iccad.add_argument("--output", required=True)
    iccad.add_argument("--image-size", type=int, default=256)
    iccad.add_argument("--resize-order", type=int, default=1, choices=[0, 1, 2, 3])
    iccad.add_argument("--normalize", choices=["train_minmax", "per_map", "none"], default="per_map")
    iccad.add_argument("--seed", type=int, default=0)
    iccad.add_argument("--train-ratio", type=float, default=0.8)
    iccad.add_argument("--val-ratio", type=float, default=0.1)
    iccad.add_argument("--limit", type=int, default=None)
    iccad.set_defaults(func=cmd_iccad)
    return parser.parse_args()


if __name__ == "__main__":
    parsed = parse_args()
    parsed.func(parsed)
