import logging
import os
import sys
from datetime import datetime
from pathlib import Path


def build_logger(save_dir: str, name: str = "pgau", filename: str = "run.log"):
    log_dir = Path(save_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    file_handler = logging.FileHandler(log_dir / filename)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger


def timestamped_dir(root: str, prefix: str) -> str:
    path = os.path.join(root, f"{prefix}-{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(path, exist_ok=True)
    return path
