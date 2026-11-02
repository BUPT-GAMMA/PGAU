from typing import Dict, Optional

import numpy as np


def _flatten(a, mask: Optional[np.ndarray] = None) -> np.ndarray:
    a = np.asarray(a, dtype=np.float64)
    if mask is not None:
        mask = np.asarray(mask).astype(bool)
        a = a[mask]
    return a.reshape(-1)


def mae(target, pred, mask: Optional[np.ndarray] = None) -> float:
    target = _flatten(target, mask)
    pred = _flatten(pred, mask)
    return float(np.mean(np.abs(target - pred)))


def mse(target, pred, mask: Optional[np.ndarray] = None) -> float:
    target = _flatten(target, mask)
    pred = _flatten(pred, mask)
    return float(np.mean((target - pred) ** 2))


def rmse(target, pred, mask: Optional[np.ndarray] = None) -> float:
    return float(np.sqrt(mse(target, pred, mask)))


def pearson_coefficient(target, pred, mask: Optional[np.ndarray] = None) -> float:
    target = _flatten(target, mask)
    pred = _flatten(pred, mask)
    if target.size < 2 or np.std(target) == 0 or np.std(pred) == 0:
        return 0.0
    return float(np.corrcoef(target, pred)[0, 1])


def mirde(target, pred, mask: Optional[np.ndarray] = None) -> float:
    target = _flatten(target, mask)
    pred = _flatten(pred, mask)
    if target.size == 0:
        return 0.0
    index = int(np.argmax(target))
    return float(abs(target[index] - pred[index]))


def evaluate_regression(target, pred, mask: Optional[np.ndarray] = None) -> Dict[str, float]:
    return {
        "mae": mae(target, pred, mask),
        "mse": mse(target, pred, mask),
        "rmse": rmse(target, pred, mask),
        "cc": pearson_coefficient(target, pred, mask),
        "mirde": mirde(target, pred, mask),
    }


def scale_metrics(metrics: Dict[str, float], scale: float) -> Dict[str, float]:
    result = dict(metrics)
    for key in ("mae", "rmse", "mirde"):
        if key in result:
            result[key] *= scale
    if "mse" in result:
        result["mse"] *= scale * scale
    return result


max_drop_mae = mirde
