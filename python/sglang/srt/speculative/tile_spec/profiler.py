"""
Tile-Spec Profiler - Minimal implementation for cache management.

Cache structure: tile_spec/cache/{model}_{gpu}_{tp}/
"""

import logging
import re
from pathlib import Path
from typing import Optional, Tuple

import torch

from sglang.srt.speculative.tile_spec.core import Calibration, PiecewiseLinearLatency

logger = logging.getLogger(__name__)


def get_cache_dir(model_path: str, gpu_name: str, tp_size: int) -> Path:
    """Get cache directory path based on model, GPU, and TP configuration."""
    # Normalize model name
    model_name = model_path.split("/")[-1].lower()
    model_name = re.sub(r"[^a-z0-9-]", "", model_name)

    # Normalize GPU name
    gpu_short = gpu_name.lower()
    for pattern in ["nvidia ", "geforce ", "tesla ", "-sxm4", "-pcie", "-80gb", "-40gb"]:
        gpu_short = gpu_short.replace(pattern, "")
    gpu_short = re.sub(r"[^a-z0-9]", "", gpu_short)

    cache_name = f"{model_name}_{gpu_short}_tp{tp_size}"

    # Cache in project root tile_spec/cache/
    current = Path(__file__).parent
    while current != current.parent:
        if (current / "tile_spec").exists() and (current / "python").exists():
            return current / "tile_spec" / "cache" / cache_name
        current = current.parent

    # Fallback to ~/.cache/sglang/tile_spec/
    return Path.home() / ".cache" / "sglang" / "tile_spec" / cache_name


class TileSpecProfiler:
    """Manages tile-spec cache loading and saving."""

    def __init__(self, server_args):
        self.server_args = server_args
        self.gpu_name = torch.cuda.get_device_name(0)
        self.cache_dir = get_cache_dir(
            server_args.model_path,
            self.gpu_name,
            server_args.tp_size
        )
        logger.info(f"Tile-spec cache dir: {self.cache_dir}")

        # Load cached models if available
        self.latency_model = None
        self.calibration = None
        self._load_cached_models()

    def _load_cached_models(self) -> bool:
        """Try to load models from cache."""
        latency_path = self.cache_dir / "latency_model.npz"
        calibration_path = self.cache_dir / "calibration.npz"

        if latency_path.exists() and calibration_path.exists():
            try:
                self.latency_model = PiecewiseLinearLatency()
                self.latency_model.load(str(latency_path))

                self.calibration = Calibration()
                self.calibration.load(str(calibration_path))

                logger.info(f"Loaded tile-spec models from cache")
                logger.info(f"  Latency boundaries: {self.latency_model.boundaries}")
                return True
            except Exception as e:
                logger.warning(f"Failed to load cached models: {e}")
                self.latency_model = None
                self.calibration = None

        return False

    def get_models(self) -> Tuple[Optional[PiecewiseLinearLatency], Optional[Calibration]]:
        """Get fitted models (or None if not available)."""
        return self.latency_model, self.calibration
