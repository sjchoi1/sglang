"""
Tile-Spec Profiler - Automatic profiling at engine initialization.

Profiles latency by recording actual verify() calls during warmup.
Uses real sglang generation requests for accurate profiling.

Cache structure: tile_spec/cache/{model}_{gpu}_{tp}/
"""

import logging
import re
import time
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
    """Profiles tile-spec latency/calibration using actual sglang runs."""

    def __init__(self, server_args, min_samples: int = 100):
        self.server_args = server_args
        self.cache_dir = get_cache_dir(
            server_args.model_path,
            torch.cuda.get_device_name(0),
            server_args.tp_size,
        )
        self.min_samples = min_samples

        # Profiling state
        self._profiling_latency = False
        self._collecting_calibration = False
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

        if not latency_path.exists():
            return False

        try:
            self.latency_model = PiecewiseLinearLatency()
            self.latency_model.load(str(latency_path))

            if calibration_path.exists():
                self.calibration = Calibration()
                self.calibration.load(str(calibration_path))
            else:
                # Use default calibration
                self.calibration = Calibration()

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

    def is_profiling(self) -> bool:
        """Returns True if actively collecting latency data."""
        return self._profiling_latency

    def is_collecting_calibration(self) -> bool:
        return self._collecting_calibration

    def get_models(self) -> Tuple[Optional[PiecewiseLinearLatency], Optional[Calibration]]:
        return self.latency_model, self.calibration

    # =========================================================================
    # Latency Profiling (during actual verify() calls)
    # =========================================================================

    def start_profiling(self):
        """Start collecting latency data during verify() calls."""
        if not self.needs_profiling():
            return

        self._profiling_latency = True
        self._latency_data.clear()
        self._calibration_data.clear()
        logger.info("TileSpec: Started latency profiling (recording actual verify() calls)")

    def record_latency(self, num_tokens: int, latency_ms: float):
        """Record latency from a verify() call."""
        if not self._profiling_latency:
            return

        self._latency_data.append((num_tokens, latency_ms))

        # Log progress periodically
        if len(self._latency_data) % 50 == 0:
            logger.info(f"TileSpec: Collected {len(self._latency_data)} latency samples")

        # Auto-finish when we have enough diverse samples
        if len(self._latency_data) >= self.min_samples:
            # Check we have enough unique token counts for fitting
            unique_counts = len(set(t for t, _ in self._latency_data))
            if unique_counts >= 5:
                self.finish_latency_profiling()

    def record_calibration(
        self,
        scores: torch.Tensor,
        accepted: torch.Tensor,
    ) -> bool:
        """
        Record calibration data from a verify() call.

        Returns True if calibration collection is complete.
        """
        if not self._collecting_calibration:
            return False

        scores_np = scores.detach().cpu().numpy().flatten()
        accepted_np = accepted.detach().cpu().numpy().flatten()

        for s, a in zip(scores_np, accepted_np):
            self._calibration_data.append((float(s), bool(a)))

        if len(self._calibration_data) >= self.min_samples * 10:
            self._finish_calibration()
            return True

        return False

    def finish_latency_profiling(self):
        """Finish latency profiling and fit the model."""
        if not self._profiling_latency:
            return

        self._profiling_latency = False

        if len(self._latency_data) < 10:
            logger.warning(f"TileSpec: Only {len(self._latency_data)} samples, need more data")
            return

        # Aggregate latency by token count (use median)
        latency_by_tokens = defaultdict(list)
        for num_tokens, latency_ms in self._latency_data:
            latency_by_tokens[num_tokens].append(latency_ms)

        token_counts = sorted(latency_by_tokens.keys())
        latencies = [np.median(latency_by_tokens[t]) for t in token_counts]

        logger.info(f"TileSpec: Fitting latency model from {len(self._latency_data)} samples")
        logger.info(f"  Token counts: {token_counts}")

        # Fit latency model
        self.latency_model = PiecewiseLinearLatency()
        self.latency_model.fit(token_counts, latencies)

        # Use default calibration for now
        self.calibration = Calibration()

        # Enable calibration collection for future requests
        self._collecting_calibration = True

        # Save to cache
        self._save_cache()

        logger.info("=" * 60)
        logger.info("TileSpec: Latency profiling complete")
        logger.info(f"  Boundaries: {self.latency_model.boundaries}")
        logger.info(f"  Optimal k candidates: {self.latency_model.get_optimal_k_candidates()}")
        logger.info(f"  Cache: {self.cache_dir}")
        logger.info("=" * 60)

        self._latency_data.clear()

    def _finish_calibration(self):
        """Fit calibration model from collected data."""
        self._collecting_calibration = False

        if self._calibration_data:
            scores = np.array([s for s, _ in self._calibration_data])
            accepted = np.array([a for _, a in self._calibration_data])

            self.calibration = Calibration()
            self.calibration.fit(scores, accepted)

            # Update cache
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            self.calibration.save(str(self.cache_dir / "calibration.npz"))

            logger.info(f"TileSpec: Calibration complete ({len(self._calibration_data)} samples)")

        self._calibration_data.clear()

    def _save_cache(self):
        """Save models to cache."""
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        if self.latency_model:
            self.latency_model.save(str(self.cache_dir / "latency_model.npz"))

        if self.calibration:
            self.calibration.save(str(self.cache_dir / "calibration.npz"))

        logger.info(f"  Saved to cache: {self.cache_dir}")

    # =========================================================================
    # Warmup Profiling (run at init to trigger actual model runs)
    # =========================================================================

    def run_warmup_profiling(self, generate_func, num_prompts: int = 100):
        """
        Run warmup requests to profile latency.

        This runs actual generation requests through sglang to populate
        latency data from real verify() calls.

        Args:
            generate_func: Function to call for generation (e.g., engine.generate)
            num_prompts: Number of prompts to run
        """
        if not self.needs_profiling():
            return

        logger.info("=" * 60)
        logger.info("TileSpec: Running warmup profiling with actual generation")
        logger.info(f"  Will run {num_prompts} prompts")
        logger.info("=" * 60)

        # Start profiling
        self.start_profiling()

        # Generate prompts with varying lengths for diversity
        prompts = [
            f"Write a short story about topic number {i}. Make it interesting and engaging."
            for i in range(num_prompts)
        ]

        # Run generation
        for i, prompt in enumerate(prompts):
            try:
                generate_func(prompt, sampling_params={"temperature": 0, "max_new_tokens": 64})
            except Exception as e:
                logger.warning(f"Warmup request {i} failed: {e}")

            if (i + 1) % 20 == 0:
                logger.info(f"TileSpec: Warmup progress {i + 1}/{num_prompts}")

        # Finish profiling
        self.finish_latency_profiling()

    # =========================================================================
    # Legacy API (for compatibility)
    # =========================================================================

    def record(
        self,
        num_tokens: int,
        latency_ms: float,
        scores: Optional[torch.Tensor] = None,
        accepted: Optional[torch.Tensor] = None,
    ):
        """Legacy: Record verification sample."""
        self.record_latency(num_tokens, latency_ms)
        if scores is not None and accepted is not None:
            self.record_calibration(scores, accepted)
