"""
Tile-Spec Profiler - Online profiling during actual speculation path.

Profiles latency and calibration by recording data during real verify() calls.
Cache structure: tile_spec/cache/{model}_{gpu}_{tp}/
"""

import logging
import re
from collections import defaultdict
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch

from sglang.srt.speculative.tile_spec.core import Calibration, PiecewiseLinearLatency

logger = logging.getLogger(__name__)


def get_cache_dir(model_path: str, gpu_name: str, tp_size: int) -> Path:
    """Get cache directory path based on model, GPU, and TP configuration."""
    model_name = model_path.split("/")[-1].lower()
    model_name = re.sub(r"[^a-z0-9-]", "", model_name)

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

    return Path.home() / ".cache" / "sglang" / "tile_spec" / cache_name


class TileSpecProfiler:
    """Profiles tile-spec latency/calibration and manages cache."""

    def __init__(self, server_args, min_samples: int = 100):
        self.cache_dir = get_cache_dir(
            server_args.model_path,
            torch.cuda.get_device_name(0),
            server_args.tp_size,
        )
        self.min_samples = min_samples

        # Profiling data
        self._profiling = False
        self._latency_data: List[Tuple[int, float]] = []
        self._calibration_data: List[Tuple[float, bool]] = []

        # Models
        self.latency_model: Optional[PiecewiseLinearLatency] = None
        self.calibration: Optional[Calibration] = None

        # Try load from cache
        self._load_cache()

    def _load_cache(self) -> bool:
        """Load models from cache if available."""
        latency_path = self.cache_dir / "latency_model.npz"
        calibration_path = self.cache_dir / "calibration.npz"

        if not (latency_path.exists() and calibration_path.exists()):
            return False

        try:
            self.latency_model = PiecewiseLinearLatency()
            self.latency_model.load(str(latency_path))
            self.calibration = Calibration()
            self.calibration.load(str(calibration_path))
            logger.info(f"Loaded tile-spec from cache: {self.cache_dir}")
            logger.info(f"  Boundaries: {self.latency_model.boundaries}")
            return True
        except Exception as e:
            logger.warning(f"Failed to load cache: {e}")
            self.latency_model = None
            self.calibration = None
            return False

    def needs_profiling(self) -> bool:
        return self.latency_model is None

    def start_profiling(self):
        """Start collecting profiling data."""
        self._profiling = True
        self._latency_data.clear()
        self._calibration_data.clear()
        logger.info(f"Tile-spec profiling started (need {self.min_samples} samples)")

    def is_profiling(self) -> bool:
        return self._profiling

    def record(
        self,
        num_tokens: int,
        latency_ms: float,
        scores: Optional[torch.Tensor] = None,
        accepted: Optional[torch.Tensor] = None,
    ):
        """Record one verification sample."""
        if not self._profiling:
            return

        self._latency_data.append((num_tokens, latency_ms))

        if scores is not None and accepted is not None:
            for s, a in zip(
                scores.detach().cpu().numpy().flatten(),
                accepted.detach().cpu().numpy().flatten(),
            ):
                self._calibration_data.append((float(s), bool(a)))

        if len(self._latency_data) >= self.min_samples:
            self._finish_profiling()

    def _finish_profiling(self):
        """Fit models and save to cache."""
        self._profiling = False

        # Aggregate latency by token count
        latency_by_tokens = defaultdict(list)
        for num_tokens, latency_ms in self._latency_data:
            latency_by_tokens[num_tokens].append(latency_ms)

        token_counts = sorted(latency_by_tokens.keys())
        latencies = [np.median(latency_by_tokens[t]) for t in token_counts]

        # Fit latency model
        self.latency_model = PiecewiseLinearLatency()
        self.latency_model.fit(token_counts, latencies)

        # Fit calibration model
        self.calibration = Calibration()
        if self._calibration_data:
            scores = np.array([s for s, _ in self._calibration_data])
            accepted = np.array([a for _, a in self._calibration_data])
            self.calibration.fit(scores, accepted)

        # Save to cache
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.latency_model.save(str(self.cache_dir / "latency_model.npz"))
        self.calibration.save(str(self.cache_dir / "calibration.npz"))

        logger.info(f"Tile-spec profiling complete ({len(self._latency_data)} samples)")
        logger.info(f"  Saved to: {self.cache_dir}")
        logger.info(f"  Boundaries: {self.latency_model.boundaries}")

        # Clear data
        self._latency_data.clear()
        self._calibration_data.clear()

    def get_models(self) -> Tuple[Optional[PiecewiseLinearLatency], Optional[Calibration]]:
        return self.latency_model, self.calibration
