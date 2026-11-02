import argparse
import json
import os
import platform
import sys
import time
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

from data_preprocess import add_augmented_features


ANALYSIS_FEATURE_COLUMNS = [
    "loop_r", "vdd_r", "gnd_r", "cell_name", "freq", "min_res_vdd", "min_res_vss",
    "toggle_rate", "leakage_power", "switching_power", "internal_power", "total_power",
    "PG_pin", "PG_volt", "powerDomain", "composite_power", "neighbor_power", "neighbor_count",
]
AUGMENTED_FEATURES = {"composite_power", "neighbor_power", "neighbor_count"}


def discover_tables(path: Path) -> List[Path]:
    if path.is_file():
        return [path]
    return sorted(list(path.rglob("*.csv")) + list(path.rglob("*.ftr")) + list(path.rglob("*.feather")))


def load_tables(
    path: Path,
    max_rows: int = 0,
    max_files: int = 0,
    seed: int = 0,
    recompute_augmented: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, float]]:
    """Load a deterministic, approximately design-balanced sample.

    A per-file quota prevents the first large design from consuming the whole
    row budget. Augmented features are computed before row sampling so that
    neighborhood features retain the complete within-design context.
    """
    files = discover_tables(path)
    if not files:
        raise ValueError(f"no CSV/Feather tables found in {path}")
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(files))
    if max_files > 0:
        order = order[: min(max_files, len(order))]
    selected_files = [files[int(index)] for index in order]
    per_file_quota = max(1, int(np.ceil(max_rows / len(selected_files)))) if max_rows > 0 else 0

    frames: List[pd.DataFrame] = []
    manifest: List[Dict] = []
    construction_totals = {feature: 0.0 for feature in AUGMENTED_FEATURES}
    for file_index, file in enumerate(selected_files):
        read_start = time.perf_counter()
        frame = pd.read_feather(file) if file.suffix in {".ftr", ".feather"} else pd.read_csv(file)
        read_time = time.perf_counter() - read_start
        input_rows = len(frame)
        needs_augmentation = recompute_augmented or not AUGMENTED_FEATURES.issubset(frame.columns)
        augmentation_time = 0.0
        if needs_augmentation:
            augmentation_start = time.perf_counter()
            frame = add_augmented_features(frame)
            augmentation_time = time.perf_counter() - augmentation_start
            composite_share = min(augmentation_time, max(augmentation_time * 0.01, 1e-9))
            construction_totals["composite_power"] += composite_share
            neighbor_share = max(augmentation_time - composite_share, 0.0)
            construction_totals["neighbor_power"] += neighbor_share
            construction_totals["neighbor_count"] += neighbor_share
        if per_file_quota and len(frame) > per_file_quota:
            frame = frame.sample(per_file_quota, random_state=seed + file_index)
        frame = frame.copy()
        frame["__source_file"] = str(file.resolve())
        frames.append(frame)
        manifest.append({
            "source_file": str(file.resolve()), "input_rows": input_rows, "sampled_rows": len(frame),
            "read_time_s": read_time, "augmentation_recomputed": needs_augmentation,
            "augmentation_time_s": augmentation_time,
        })
    data = pd.concat(frames, ignore_index=True)
    if max_rows > 0 and len(data) > max_rows:
        data = data.sample(max_rows, random_state=seed).reset_index(drop=True)
    if data.empty:
        raise ValueError(f"no rows loaded from {path}")
    return data, pd.DataFrame(manifest), construction_totals


def build_target(data: pd.DataFrame, target: str) -> np.ndarray:
    if target == "ir_drop":
        missing = sorted({"vdd_drop", "gnd_bounce"}.difference(data.columns))
        if missing:
            raise KeyError(f"target ir_drop requires columns: {missing}")
        return (pd.to_numeric(data["vdd_drop"], errors="coerce")
                + pd.to_numeric(data["gnd_bounce"], errors="coerce")).to_numpy(dtype=np.float32)
    if target not in data.columns:
        raise KeyError(f"target column {target!r} is unavailable")
    return pd.to_numeric(data[target], errors="coerce").to_numpy(dtype=np.float32)


def encode_features(data: pd.DataFrame, features: Sequence[str]) -> Tuple[pd.DataFrame, Dict, Dict[str, float]]:
    encoded = pd.DataFrame(index=data.index)
    schema: Dict[str, Dict] = {}
    timings: Dict[str, float] = {}
    for feature in features:
        start = time.perf_counter()
        source = data[feature]
        numeric = pd.to_numeric(source, errors="coerce")
        numeric_fraction = float(numeric.notna().mean())
        if numeric_fraction >= 0.95:
            numeric = numeric.replace([np.inf, -np.inf], np.nan)
            fill_value = float(numeric.median()) if numeric.notna().any() else 0.0
            encoded[feature] = numeric.fillna(fill_value).astype(np.float32)
            schema[feature] = {
                "type": "numeric", "missing_fraction": float(numeric.isna().mean()),
                "fill_value": fill_value, "unique_values": int(numeric.nunique(dropna=True)),
            }
        else:
            values = source.fillna("<missing>").astype(str)
            encoded[feature] = values.astype("category")
            schema[feature] = {
                "type": "categorical", "missing_fraction": float(source.isna().mean()),
                "unique_values": int(values.nunique()),
                "categories": encoded[feature].cat.categories.astype(str).tolist(),
            }
        timings[feature] = time.perf_counter() - start
    return encoded, schema, timings


def split_indices(groups: Sequence[str], validation_fraction: float, seed: int) -> Tuple[np.ndarray, np.ndarray, str]:
    from sklearn.model_selection import GroupShuffleSplit, train_test_split
    indices = np.arange(len(groups))
    unique_groups = np.unique(np.asarray(groups, dtype=str))
    if len(unique_groups) >= 2:
        splitter = GroupShuffleSplit(n_splits=1, test_size=validation_fraction, random_state=seed)
        train_index, validation_index = next(splitter.split(indices, groups=groups))
        return train_index, validation_index, "group_by_source_file"
    train_index, validation_index = train_test_split(
        indices, test_size=validation_fraction, random_state=seed, shuffle=True)
    return np.asarray(train_index), np.asarray(validation_index), "random_rows_single_source"


def _rf_compatible(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    for column in result.columns:
        if isinstance(result[column].dtype, pd.CategoricalDtype):
            result[column] = result[column].cat.codes.astype(np.float32)
    return result


def build_estimator(model_name: str, seed: int, n_estimators: int, max_depth: int,
                    learning_rate: float, device: str, early_stopping_rounds: int):
    if model_name == "xgboost":
        from xgboost import XGBRegressor
        return XGBRegressor(
            n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate,
            min_child_weight=1.0, subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
            random_state=seed, n_jobs=-1, objective="reg:squarederror", eval_metric="rmse",
            tree_method="hist", device=device, enable_categorical=True, importance_type="gain",
            early_stopping_rounds=early_stopping_rounds if early_stopping_rounds > 0 else None,
        )
    from sklearn.ensemble import RandomForestRegressor
    return RandomForestRegressor(
        n_estimators=n_estimators, max_depth=max_depth if max_depth > 0 else None,
        random_state=seed, n_jobs=-1)


def regression_metrics(target: np.ndarray, prediction: np.ndarray) -> Dict[str, float]:
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    target = np.asarray(target, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    cc = 0.0
    if target.size > 1 and np.std(target) > 0 and np.std(prediction) > 0:
        cc = float(np.corrcoef(target, prediction)[0, 1])
    mse = float(mean_squared_error(target, prediction))
    return {"mae": float(mean_absolute_error(target, prediction)), "rmse": float(np.sqrt(mse)),
            "r2": float(r2_score(target, prediction)), "cc": cc}


def fit_repeated_models(x_train: pd.DataFrame, y_train: np.ndarray, x_validation: pd.DataFrame,
                        y_validation: np.ndarray, args):
    importances, ranks, metrics, models = [], [], [], []
    for repeat in range(args.repeats):
        repeat_seed = args.seed + repeat
        model = build_estimator(args.model, repeat_seed, args.n_estimators, args.max_depth,
                                args.learning_rate, args.device, args.early_stopping_rounds)
        fit_x_train = x_train if args.model == "xgboost" else _rf_compatible(x_train)
        fit_x_validation = x_validation if args.model == "xgboost" else _rf_compatible(x_validation)
        fit_kwargs = {}
        if args.model == "xgboost" and args.early_stopping_rounds > 0:
            fit_kwargs = {"eval_set": [(fit_x_validation, y_validation)], "verbose": False}
        start = time.perf_counter()
        model.fit(fit_x_train, y_train, **fit_kwargs)
        fit_time = time.perf_counter() - start
        prediction = model.predict(fit_x_validation)
        repeat_metrics = regression_metrics(y_validation, prediction)
        repeat_metrics.update({
            "repeat": repeat, "seed": repeat_seed, "fit_time_s": fit_time,
            "best_iteration": int(getattr(model, "best_iteration", args.n_estimators - 1)),
        })
        importance = np.asarray(model.feature_importances_, dtype=np.float64)
        importances.append(importance)
        ranks.append(pd.Series(-importance).rank(method="average").to_numpy())
        metrics.append(repeat_metrics)
        models.append(model)
    return models, np.stack(importances), np.stack(ranks), pd.DataFrame(metrics)


def permutation_importance_mae(model, frame: pd.DataFrame, target: np.ndarray, model_name: str,
                               repeats: int, max_rows: int, seed: int) -> Tuple[np.ndarray, np.ndarray, int]:
    if repeats <= 0:
        zeros = np.zeros(frame.shape[1], dtype=np.float64)
        return zeros, zeros, 0
    rng = np.random.default_rng(seed)
    if max_rows > 0 and len(frame) > max_rows:
        selected = rng.choice(len(frame), size=max_rows, replace=False)
        work, work_target = frame.iloc[selected].copy(), target[selected]
    else:
        work, work_target = frame.copy(), target
    predict_frame = work if model_name == "xgboost" else _rf_compatible(work)
    baseline = float(np.mean(np.abs(work_target - model.predict(predict_frame))))
    values = np.zeros((repeats, frame.shape[1]), dtype=np.float64)
    for repeat in range(repeats):
        for column_index, column in enumerate(frame.columns):
            shuffled = work.copy()
            permuted_values = rng.permutation(shuffled[column].to_numpy())
            if isinstance(work[column].dtype, pd.CategoricalDtype):
                shuffled[column] = pd.Categorical(
                    permuted_values,
                    categories=work[column].cat.categories,
                    ordered=work[column].cat.ordered,
                )
            else:
                shuffled[column] = permuted_values
            model_input = shuffled if model_name == "xgboost" else _rf_compatible(shuffled)
            permuted_mae = float(np.mean(np.abs(work_target - model.predict(model_input))))
            values[repeat, column_index] = permuted_mae - baseline
    return values.mean(axis=0), values.std(axis=0), len(work)


def plot_result(result: pd.DataFrame, output_png: Path) -> None:
    os.environ.setdefault("MPLCONFIGDIR", str(output_png.parent / ".mplconfig"))
    import matplotlib.pyplot as plt
    ordered = result.sort_values("gain_importance_mean", ascending=True)
    height = max(6.0, len(ordered) * 0.38)
    fig, (importance_axis, time_axis) = plt.subplots(1, 2, figsize=(14, height), sharey=True)
    importance_axis.barh(ordered["feature"], ordered["gain_importance_mean"],
                         xerr=ordered["gain_importance_std"], color="#3b82f6", alpha=0.85, capsize=2)
    importance_axis.set_xlabel("XGBoost gain importance (mean +/- std)")
    importance_axis.set_ylabel("Feature")
    time_axis.barh(ordered["feature"], ordered["processing_time_s"], color="#f97316", alpha=0.85)
    time_axis.set_xlabel("Feature construction + encoding time (s)")
    fig.tight_layout()
    fig.savefig(output_png, dpi=180, bbox_inches="tight")
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(description="Feature importance analysis for PGAU aligned tables.")
    parser.add_argument("--input", required=True, help="Aligned CSV/Feather file or directory.")
    parser.add_argument("--output", default="feature_analysis", help="Output directory.")
    parser.add_argument("--features", default=",".join(ANALYSIS_FEATURE_COLUMNS))
    parser.add_argument("--target", default="ir_drop", help="Target column, or ir_drop for vdd_drop+gnd_bounce.")
    parser.add_argument("--max-rows", type=int, default=200000)
    parser.add_argument("--max-files", type=int, default=100, help="Maximum source designs sampled; 0 uses all.")
    parser.add_argument("--model", choices=["xgboost", "random_forest"], default="xgboost")
    parser.add_argument("--n-estimators", type=int, default=500)
    parser.add_argument("--max-depth", type=int, default=6)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--early-stopping-rounds", type=int, default=30)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--repeats", type=int, default=3, help="Independent seeds for stability statistics.")
    parser.add_argument("--permutation-repeats", type=int, default=3)
    parser.add_argument("--permutation-max-rows", type=int, default=20000)
    parser.add_argument("--device", default="cpu", help="XGBoost device, for example cpu or cuda:4.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--top-k", type=int, default=11, help="Write the top-K ranked feature names.")
    parser.add_argument(
        "--selection-method", choices=["gain", "permutation", "consensus"], default="gain",
        help="Ranking used for selected_features.json; consensus combines gain and permutation ranks.")
    parser.add_argument("--strict-features", action="store_true", help="Fail instead of skipping unavailable columns.")
    parser.add_argument("--recompute-augmented", action=argparse.BooleanOptionalAction, default=False,
                        help="Recompute composite/neighborhood features before sampling and measure their time.")
    return parser.parse_args()


def main():
    args = parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    data, manifest, construction_times = load_tables(
        Path(args.input), args.max_rows, args.max_files, args.seed, args.recompute_augmented)
    features = [feature.strip() for feature in args.features.split(",") if feature.strip()]
    missing = [feature for feature in features if feature not in data.columns]
    if missing and args.strict_features:
        raise KeyError(f"missing features: {missing}")
    if missing:
        print(f"skipping unavailable features: {missing}")
        features = [feature for feature in features if feature in data.columns]
    if not features:
        raise ValueError("none of the requested features are available")

    target = build_target(data, args.target)
    valid_target = np.isfinite(target)
    dropped_targets = int((~valid_target).sum())
    data, target = data.loc[valid_target].reset_index(drop=True), target[valid_target]
    x, schema, encoding_times = encode_features(data, features)
    groups = data["__source_file"].astype(str).to_numpy()
    train_index, validation_index, split_method = split_indices(groups, args.validation_fraction, args.seed)
    train_groups, validation_groups = set(groups[train_index]), set(groups[validation_index])
    if split_method == "group_by_source_file" and train_groups.intersection(validation_groups):
        raise RuntimeError("source-file leakage detected between feature-analysis train and validation sets")

    models, gain_values, rank_values, validation_metrics = fit_repeated_models(
        x.iloc[train_index], target[train_index], x.iloc[validation_index], target[validation_index], args)
    permutation_mean, permutation_std, permutation_rows = permutation_importance_mae(
        models[0], x.iloc[validation_index], target[validation_index], args.model,
        args.permutation_repeats, args.permutation_max_rows, args.seed)
    top_k = min(args.top_k, len(features))
    top_frequency = np.mean(rank_values <= top_k, axis=0) if top_k else np.zeros(len(features))
    result = pd.DataFrame({
        "feature": features, "gain_importance_mean": gain_values.mean(axis=0),
        "gain_importance_std": gain_values.std(axis=0), "gain_rank_mean": rank_values.mean(axis=0),
        "gain_rank_std": rank_values.std(axis=0), "top_k_frequency": top_frequency,
        "permutation_mae_increase_mean": permutation_mean,
        "permutation_mae_increase_std": permutation_std,
        "construction_time_s": [construction_times.get(feature, 0.0) for feature in features],
        "encoding_time_s": [encoding_times[feature] for feature in features],
    })
    result["processing_time_s"] = result["construction_time_s"] + result["encoding_time_s"]
    result["permutation_rank"] = pd.Series(-result["permutation_mae_increase_mean"]).rank(method="average")
    result["consensus_rank"] = (result["gain_rank_mean"] + result["permutation_rank"]) / 2.0
    selection_column = {
        "gain": "gain_rank_mean", "permutation": "permutation_rank", "consensus": "consensus_rank"
    }[args.selection_method]
    selected = result.sort_values(
        [selection_column, "gain_rank_mean"], ascending=[True, True]
    ).head(top_k)["feature"].tolist()
    result["selected"] = result["feature"].isin(selected)
    result = result.sort_values(["gain_importance_mean", "permutation_mae_increase_mean"],
                                ascending=[False, False]).reset_index(drop=True)
    result.to_csv(output / "feature_importance.csv", index=False)
    validation_metrics.to_csv(output / "validation_metrics.csv", index=False)
    manifest.to_csv(output / "input_manifest.csv", index=False)
    with open(output / "selected_features.json", "w") as handle:
        json.dump(selected, handle, indent=2)
    with open(output / "feature_schema.json", "w") as handle:
        json.dump(schema, handle, indent=2)
    try:
        import xgboost
        xgboost_version = xgboost.__version__
    except ImportError:
        xgboost_version = None
    metric_columns = ["mae", "rmse", "r2", "cc"]
    summary = {
        "model": models[0].__class__.__name__, "model_requested": args.model,
        "xgboost_version": xgboost_version, "python_version": platform.python_version(),
        "command": " ".join(sys.argv), "seed": args.seed, "repeats": args.repeats,
        "samples": int(len(x)), "source_files": int(len(np.unique(groups))),
        "train_samples": int(len(train_index)), "validation_samples": int(len(validation_index)),
        "train_source_files": len(train_groups), "validation_source_files": len(validation_groups),
        "split_method": split_method, "source_overlap": len(train_groups.intersection(validation_groups)),
        "dropped_nonfinite_targets": dropped_targets, "target": args.target, "device": args.device,
        "permutation_rows": permutation_rows, "selected_features": selected,
        "selection_method": args.selection_method,
        "validation_metrics_mean": validation_metrics[metric_columns].mean().to_dict(),
        "validation_metrics_std": validation_metrics[metric_columns].std(ddof=0).to_dict(),
    }
    with open(output / "summary.json", "w") as handle:
        json.dump(summary, handle, indent=2)
    plot_result(result, output / "feature_importance.png")
    print(result.to_string(index=False))
    print(json.dumps(summary, indent=2))
    print(f"saved to {output.resolve()}")


if __name__ == "__main__":
    main()
