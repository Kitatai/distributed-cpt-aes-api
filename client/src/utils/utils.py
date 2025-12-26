"""
Utility functions for the experiment.
"""

import os
import json
import csv
import random
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
import numpy as np
import torch

logger = logging.getLogger(__name__)


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    logger.info(f"Set random seed to {seed}")


def setup_logging(
    log_dir: Optional[str] = None,
    log_level: int = logging.INFO,
    log_file: Optional[str] = None,
):
    """
    Setup logging configuration.

    Args:
        log_dir: Directory for log files
        log_level: Logging level
        log_file: Specific log file name
    """
    handlers = [logging.StreamHandler()]

    if log_dir:
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)

        if log_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_file = f"experiment_{timestamp}.log"

        file_handler = logging.FileHandler(log_path / log_file)
        handlers.append(file_handler)

    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=handlers,
    )


def save_json(data: Any, path: str, indent: int = 2):
    """Save data to JSON file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=indent, ensure_ascii=False)
    logger.info(f"Saved JSON to {path}")


def load_json(path: str) -> Any:
    """Load data from JSON file."""
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_jsonl(data: List[Dict], path: str):
    """Save list of dicts to JSONL file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    logger.info(f"Saved JSONL to {path}")


def load_jsonl(path: str) -> List[Dict]:
    """Load list of dicts from JSONL file."""
    data = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


def save_csv(data: List[Dict], path: str):
    """Save list of dicts to CSV file."""
    if not data:
        logger.warning("No data to save to CSV")
        return

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = list(data[0].keys())
    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(data)
    logger.info(f"Saved CSV to {path}")


def load_csv(path: str) -> List[Dict]:
    """Load list of dicts from CSV file."""
    with open(path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        return list(reader)


def get_device() -> str:
    """Get best available device."""
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def get_gpu_memory_info() -> Dict[str, float]:
    """Get GPU memory information."""
    if not torch.cuda.is_available():
        return {"available": False}

    info = {
        "available": True,
        "device_count": torch.cuda.device_count(),
        "devices": [],
    }

    for i in range(torch.cuda.device_count()):
        device_info = {
            "name": torch.cuda.get_device_name(i),
            "total_memory_gb": torch.cuda.get_device_properties(i).total_memory / 1e9,
            "allocated_gb": torch.cuda.memory_allocated(i) / 1e9,
            "cached_gb": torch.cuda.memory_reserved(i) / 1e9,
        }
        info["devices"].append(device_info)

    return info


def format_time(seconds: float) -> str:
    """Format seconds to human readable string."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"


class Timer:
    """Simple timer context manager."""

    def __init__(self, name: str = ""):
        self.name = name
        self.start_time = None
        self.elapsed = None

    def __enter__(self):
        self.start_time = datetime.now()
        return self

    def __exit__(self, *args):
        self.elapsed = (datetime.now() - self.start_time).total_seconds()
        if self.name:
            logger.info(f"{self.name}: {format_time(self.elapsed)}")

    def get_elapsed(self) -> float:
        if self.elapsed is not None:
            return self.elapsed
        if self.start_time is not None:
            return (datetime.now() - self.start_time).total_seconds()
        return 0.0


def ensure_dir(path: Union[str, Path]) -> Path:
    """Ensure directory exists."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_checkpoint_epochs(checkpoint_dir: str) -> List[int]:
    """Get list of available checkpoint epochs."""
    checkpoint_path = Path(checkpoint_dir)
    if not checkpoint_path.exists():
        return []

    epochs = []
    for d in checkpoint_path.iterdir():
        if d.is_dir() and d.name.startswith("epoch_"):
            try:
                epoch = int(d.name.split("_")[1])
                adapter_path = d / "adapter"
                if adapter_path.exists():
                    epochs.append(epoch)
            except (ValueError, IndexError):
                continue

    return sorted(epochs)


def cleanup_gpu_memory():
    """Cleanup GPU memory."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


class ExperimentTracker:
    """Track experiment progress and results."""

    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results = {
            "start_time": datetime.now().isoformat(),
            "epochs": {},
            "config": None,
        }

    def set_config(self, config: Dict):
        """Set experiment configuration."""
        self.results["config"] = config

    def add_epoch_result(
        self,
        epoch: int,
        metrics: Dict,
        predictions_path: Optional[str] = None,
        adapter_path: Optional[str] = None,
    ):
        """Add results for an epoch."""
        self.results["epochs"][epoch] = {
            "metrics": metrics,
            "predictions_path": predictions_path,
            "adapter_path": adapter_path,
            "timestamp": datetime.now().isoformat(),
        }
        self._save()

    def set_best_epoch(self, epoch: int, metric_name: str, metric_value: float):
        """Set best epoch information."""
        self.results["best_epoch"] = {
            "epoch": epoch,
            "metric_name": metric_name,
            "metric_value": metric_value,
        }
        self._save()

    def finish(self):
        """Mark experiment as finished."""
        self.results["end_time"] = datetime.now().isoformat()
        self._save()

    def _save(self):
        """Save results to file."""
        save_json(self.results, self.output_dir / "experiment_results.json")

    def load(self) -> Dict:
        """Load existing results."""
        results_path = self.output_dir / "experiment_results.json"
        if results_path.exists():
            self.results = load_json(str(results_path))
        return self.results
