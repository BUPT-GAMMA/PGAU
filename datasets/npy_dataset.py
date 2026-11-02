import csv
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass
class SampleInfo:
    feature_path: str
    label_path: Optional[str] = None
    instance_count_path: Optional[str] = None
    instance_ir_drop_path: Optional[str] = None
    instance_name_path: Optional[str] = None

    @property
    def name(self) -> str:
        return Path(self.feature_path).stem


def _resolve_path(dataroot: Optional[str], path: str) -> str:
    path = os.path.expanduser(path)
    if os.path.isabs(path) or dataroot is None:
        return path
    return os.path.join(dataroot, path)


def _load_npy(path: str) -> np.ndarray:
    data = np.load(path, allow_pickle=False)
    if isinstance(data, np.lib.npyio.NpzFile):
        first_key = data.files[0]
        data = data[first_key]
    return np.asarray(data)


def to_chw(array: np.ndarray) -> np.ndarray:
    array = np.asarray(array)
    if array.ndim == 2:
        array = array[None, :, :]
    elif array.ndim == 3:
        first, _, last = array.shape
        # Preprocessing writes HWC while external users may provide CHW.  Use
        # the smaller edge as the channel axis, which also handles 16/32-pixel
        # smoke maps.  Shapes with equal edge sizes are ambiguous; HWC is the
        # repository-native format and is therefore the safe default.
        channels_first = first <= 32 and first < last
        channels_last = last <= 32 and last < first
        if channels_last or not channels_first:
            array = np.transpose(array, (2, 0, 1))
    else:
        raise ValueError(f"expected 2D or 3D array, got shape {array.shape}")
    return np.ascontiguousarray(array.astype(np.float32))


class NpyMapDataset(Dataset):

    def __init__(self, ann_file: str, dataroot: Optional[str] = None, require_label: bool = True) -> None:
        self.ann_file = ann_file
        self.dataroot = dataroot
        self.require_label = require_label
        self.samples = self._load_annotations()

    def _load_annotations(self) -> List[SampleInfo]:
        samples: List[SampleInfo] = []
        with open(self.ann_file, "r", newline="") as handle:
            reader = csv.reader(handle)
            for row in reader:
                if not row or not any(cell.strip() for cell in row):
                    continue
                row = [cell.strip() for cell in row]
                if row[0].startswith("#") or row[0].lower() in {"feature", "feature_path"}:
                    continue
                if self.require_label and (len(row) < 2 or not row[1]):
                    raise ValueError(f"annotation row must contain feature,label: {row}")

                values = row + [None] * (5 - len(row))
                samples.append(
                    SampleInfo(
                        feature_path=_resolve_path(self.dataroot, values[0]),
                        label_path=_resolve_path(self.dataroot, values[1]) if values[1] else None,
                        instance_count_path=_resolve_path(self.dataroot, values[2]) if values[2] else None,
                        instance_ir_drop_path=_resolve_path(self.dataroot, values[3]) if values[3] else None,
                        instance_name_path=_resolve_path(self.dataroot, values[4]) if values[4] else None,
                    )
                )
        if not samples:
            raise ValueError(f"no samples found in {self.ann_file}")
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict[str, object]:
        sample = self.samples[index]
        feature = torch.from_numpy(to_chw(_load_npy(sample.feature_path)))
        item: Dict[str, object] = {
            "feature": feature,
            "name": sample.name,
            "feature_path": sample.feature_path,
        }
        if sample.label_path is not None:
            item["label"] = torch.from_numpy(to_chw(_load_npy(sample.label_path)))
            item["label_path"] = sample.label_path

        for key in ("instance_count_path", "instance_ir_drop_path", "instance_name_path"):
            value = getattr(sample, key)
            if value:
                item[key] = value
        return item
