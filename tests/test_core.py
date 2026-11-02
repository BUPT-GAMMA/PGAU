import unittest
import importlib.util
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from cross_validate import make_folds
from analysis_feature import (
    encode_features,
    fit_repeated_models,
    load_tables,
    permutation_importance_mae,
    split_indices,
)
from data_preprocess import add_augmented_features, physical_indices, validate_split_ratios
from datasets.npy_dataset import NpyMapDataset, to_chw
from models.pgau import PGAU
from models.build_model import build_model
from utils.losses import LDSWeightTable, build_loss
from utils.metrics import evaluate_regression, scale_metrics
from test import load_model_state
from train import load_checkpoint


class CoreTests(unittest.TestCase):
    def test_dataset_batches_are_nchw_256(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            feature_path = root / "feature.npy"
            label_path = root / "label.npy"
            csv_path = root / "samples.csv"
            np.save(feature_path, np.zeros((256, 256, 11), dtype=np.float32))
            np.save(label_path, np.zeros((256, 256, 1), dtype=np.float32))
            csv_path.write_text(f"{feature_path},{label_path}\n")
            batch = next(iter(DataLoader(NpyMapDataset(str(csv_path)), batch_size=1)))
            self.assertEqual(tuple(batch["feature"].shape), (1, 11, 256, 256))
            self.assertEqual(tuple(batch["label"].shape), (1, 1, 256, 256))

    def test_resume_retargets_cosine_schedule_when_epochs_change(self):
        source = torch.nn.Linear(2, 1)
        source_optimizer = torch.optim.AdamW(source.parameters(), lr=2e-4)
        source_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(source_optimizer, T_max=1, eta_min=1e-7)
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "last.pth"
            torch.save(
                {
                    "epoch": 1,
                    "best_mae": 1.0,
                    "model": source.state_dict(),
                    "optimizer": source_optimizer.state_dict(),
                    "scheduler": source_scheduler.state_dict(),
                },
                checkpoint,
            )
            target = torch.nn.Linear(2, 1)
            optimizer = torch.optim.AdamW(target.parameters(), lr=2e-4)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=2, eta_min=1e-7)
            epoch, _ = load_checkpoint(
                str(checkpoint), target, optimizer, scheduler, total_epochs=2
            )
            self.assertEqual(epoch, 1)
            self.assertEqual(scheduler.T_max, 2)
            self.assertAlmostEqual(optimizer.param_groups[0]["lr"], 0.00010005, places=8)

    def test_split_ratio_validation(self):
        validate_split_ratios(0.8, 0.1)
        with self.assertRaises(ValueError):
            validate_split_ratios(0.9, 0.2)
        with self.assertRaises(ValueError):
            make_folds([["a"], ["b"], ["c"]], folds=2, seed=0, val_ratio=1.0)

    def test_partial_checkpoint_load_skips_wrong_shapes(self):
        source = PGAU(3, 2, base_channels=4)
        target = PGAU(3, 1, base_channels=4)
        skipped, unresolved = load_model_state(target, source.state_dict(), allow_mismatch=True)
        self.assertTrue(any("head" in key for key in skipped))
        self.assertTrue(any("head" in key for key in unresolved))
        with self.assertRaises(RuntimeError):
            load_model_state(target, source.state_dict(), allow_mismatch=False)

    def test_mobilenetv3_and_lr_aspp_are_distinct(self):
        common = {"in_channels": 3, "out_channels": 1, "base_channels": 8}
        mobile = build_model({"model_type": "mobilenetv3", **common})
        lr_aspp = build_model({"model_type": "lr_aspp", **common})
        self.assertNotEqual(type(mobile), type(lr_aspp))
        mobile_parameters = sum(parameter.numel() for parameter in mobile.parameters())
        lr_aspp_parameters = sum(parameter.numel() for parameter in lr_aspp.parameters())
        self.assertNotEqual(mobile_parameters, lr_aspp_parameters)
        x = torch.randn(2, 3, 32, 32)
        self.assertEqual(tuple(mobile(x).shape), (2, 1, 32, 32))
        self.assertEqual(tuple(lr_aspp(x).shape), (2, 1, 32, 32))

    def test_to_chw_handles_small_maps(self):
        chw = np.zeros((3, 16, 16), dtype=np.float32)
        hwc = np.zeros((16, 16, 3), dtype=np.float32)
        self.assertEqual(to_chw(chw).shape, (3, 16, 16))
        self.assertEqual(to_chw(hwc).shape, (3, 16, 16))

    def test_feature_analysis_balanced_loading_and_group_split(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for design, rows in (("a", 100), ("b", 40)):
                frame = pd.DataFrame(
                    {
                        "vdd_drop": np.linspace(0.0, 0.01, rows),
                        "gnd_bounce": np.zeros(rows),
                        "loop_r": np.arange(rows),
                        "composite_power": np.ones(rows),
                        "neighbor_power": np.ones(rows),
                        "neighbor_count": np.ones(rows),
                    }
                )
                frame.to_csv(root / f"{design}.csv", index=False)
            data, manifest, _ = load_tables(root, max_rows=20, max_files=2, seed=0)
            self.assertEqual(len(data), 20)
            self.assertEqual(manifest["sampled_rows"].tolist(), [10, 10])
            groups = data["__source_file"].to_numpy()
            train, validation, method = split_indices(groups, validation_fraction=0.5, seed=0)
            self.assertEqual(method, "group_by_source_file")
            self.assertFalse(set(groups[train]) & set(groups[validation]))

    @unittest.skipUnless(importlib.util.find_spec("xgboost"), "xgboost is not installed")
    def test_xgboost_native_category_and_permutation(self):
        rows = 80
        frame = pd.DataFrame(
            {
                "numeric": np.linspace(0.0, 1.0, rows),
                "category": np.where(np.arange(rows) % 2, "a", "b"),
            }
        )
        x, schema, _ = encode_features(frame, ["numeric", "category"])
        self.assertEqual(schema["category"]["type"], "categorical")
        target = frame["numeric"].to_numpy(dtype=np.float32)
        args = SimpleNamespace(
            repeats=1,
            seed=0,
            model="xgboost",
            n_estimators=10,
            max_depth=2,
            learning_rate=0.1,
            device="cpu",
            early_stopping_rounds=0,
        )
        models, gains, _, metrics = fit_repeated_models(x.iloc[:60], target[:60], x.iloc[60:], target[60:], args)
        self.assertEqual(gains.shape, (1, 2))
        self.assertTrue(np.isfinite(metrics[["mae", "rmse", "r2", "cc"]].to_numpy()).all())
        mean, std, count = permutation_importance_mae(
            models[0], x.iloc[60:], target[60:], "xgboost", repeats=1, max_rows=20, seed=0
        )
        self.assertEqual(count, 20)
        self.assertTrue(np.isfinite(mean).all() and np.isfinite(std).all())

    def test_pgau_attention_and_ablation_preserve_odd_size(self):
        x = torch.randn(2, 3, 35, 47)
        for use_attention in (True, False):
            model = PGAU(3, 1, base_channels=4, use_attention=use_attention)
            y = model(x)
            self.assertEqual(tuple(y.shape), (2, 1, 35, 47))
            y.mean().backward()
            self.assertTrue(all(torch.isfinite(parameter.grad).all() for parameter in model.parameters() if parameter.grad is not None))

    def test_lds_streaming_histogram_and_loss(self):
        arrays = (np.full((1, 2, 2), value, dtype=np.float32) for value in (0.0, 0.0001, 0.0002))
        table = LDSWeightTable.from_arrays(arrays, bin_width=0.0001, kernel_size=5, sigma=2.0)
        self.assertEqual(len(table.weights), 3)
        self.assertTrue(np.isfinite(table.weights).all())
        pred = torch.zeros(1, 1, 2, 2, requires_grad=True)
        target = torch.full_like(pred, 0.0001)
        loss = build_loss("lds_l1", lds_table=table)(pred, target)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))

    def test_augmented_features(self):
        frame = pd.DataFrame(
            {
                "internal_power": [1.0, 2.0],
                "switching_power": [3.0, 4.0],
                "toggle_rate": [0.5, 0.25],
                "leakage_power": [0.1, 0.2],
                "total_power": [10.0, 20.0],
                "center_x": [0.0, 1.0],
                "center_y": [0.0, 0.0],
            }
        )
        result = add_augmented_features(frame, neighbor_radius=2.0)
        np.testing.assert_allclose(result["composite_power"], [2.1, 1.7])
        np.testing.assert_allclose(result["neighbor_power"], [20.0, 10.0])
        np.testing.assert_allclose(result["neighbor_count"], [1.0, 1.0])

    def test_physical_index_mapping(self):
        rows, cols, valid = physical_indices([10.0, 10.72, 11.44], [10.0, 10.576, 11.152])
        np.testing.assert_array_equal(rows, [0, 1, 1])
        np.testing.assert_array_equal(cols, [0, 1, 1])
        self.assertTrue(valid.all())

    def test_metric_scaling(self):
        metrics = evaluate_regression([0.0, 0.001], [0.0, 0.0])
        scaled = scale_metrics(metrics, 1000.0)
        self.assertAlmostEqual(scaled["mae"], 0.5)
        self.assertAlmostEqual(scaled["mse"], 0.5)
        self.assertAlmostEqual(scaled["rmse"], np.sqrt(0.5))

    def test_cross_validation_coverage(self):
        rows = [[str(index), str(index)] for index in range(12)]
        folds = make_folds(rows, folds=6, seed=0, val_ratio=0.1)
        tests = [row[0] for fold in folds for row in fold["test"]]
        self.assertEqual(sorted(tests, key=int), [str(index) for index in range(12)])
        for fold in folds:
            train = {row[0] for row in fold["train"]}
            val = {row[0] for row in fold["val"]}
            test = {row[0] for row in fold["test"]}
            self.assertFalse(train & val or train & test or val & test)

    def test_cross_validation_stratifies_real_designs(self):
        rows = [[f"feature/fakedesign{index}.npy", "label.npy"] for index in range(100)]
        rows += [[f"feature/realdesign{index}.npy", "label.npy"] for index in range(20)]
        folds = make_folds(rows, folds=6, seed=0, val_ratio=0.1, stratify_real=True)
        real_counts = [sum("realdesign" in row[0] for row in fold["test"]) for fold in folds]
        self.assertEqual(sum(real_counts), 20)
        self.assertLessEqual(max(real_counts) - min(real_counts), 1)


if __name__ == "__main__":
    unittest.main()
