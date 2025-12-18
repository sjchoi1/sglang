"""
Tile-Spec Profiler - Automatic profiling at engine initialization.

Profiles latency automatically during init via synthetic forward passes.
Calibration can be collected online during first N requests.

Cache structure: tile_spec/cache/{model}_{gpu}_{tp}/
"""

import logging
import re
import time
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
    """Profiles tile-spec latency/calibration with automatic init-time profiling."""

    def __init__(self, server_args, min_samples: int = 100):
        self.server_args = server_args
        self.cache_dir = get_cache_dir(
            server_args.model_path,
            torch.cuda.get_device_name(0),
            server_args.tp_size,
        )
        self.min_samples = min_samples

        # Profiling state
        self._collecting_calibration = False
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
        """For compatibility - always False since we profile at init."""
        return False

    def is_collecting_calibration(self) -> bool:
        return self._collecting_calibration

    def get_models(self) -> Tuple[Optional[PiecewiseLinearLatency], Optional[Calibration]]:
        return self.latency_model, self.calibration

    # =========================================================================
    # Automatic Init-Time Profiling
    # =========================================================================

    def profile_at_init(self, target_worker, enable_calibration: bool = True):
        """
        Run automatic profiling at engine initialization.

        Args:
            target_worker: The target model worker for latency profiling
            enable_calibration: If True, enable calibration collection during first N requests
        """
        if not self.needs_profiling():
            return

        logger.info("=" * 60)
        logger.info("TileSpec: Starting automatic profiling at init")
        logger.info("=" * 60)

        # Phase 1: Latency profiling with synthetic forward passes
        self._profile_latency_synthetic(target_worker)

        # Phase 2: Set up calibration collection (happens during first N requests)
        if enable_calibration:
            self._collecting_calibration = True
            self._calibration_data.clear()
            logger.info("Calibration: Will collect during first ~100 requests")
        else:
            self.calibration = Calibration()  # Default 0.5 for all bins
            logger.info("Calibration: Using default (0.5)")

        # Save latency model to cache
        self._save_cache()

        logger.info("=" * 60)
        logger.info(f"TileSpec: Latency profiling complete")
        logger.info(f"  Boundaries: {self.latency_model.boundaries}")
        logger.info(f"  Optimal k candidates: {self.latency_model.get_optimal_k_candidates()}")
        logger.info(f"  Cache: {self.cache_dir}")
        logger.info("=" * 60)

    def _profile_latency_synthetic(self, target_worker):
        """
        Profile latency with synthetic forward passes.

        Runs the target model with varying batch sizes to detect tile boundaries.
        """
        logger.info("Latency profiling with synthetic forward passes...")

        model_runner = target_worker.model_runner
        device = next(model_runner.model.parameters()).device
        vocab_size = model_runner.model_config.vocab_size

        # Token counts to profile - cover expected range with fine granularity near boundaries
        token_counts = [
            8, 16, 24, 32, 48, 56, 64, 72, 80, 96, 112, 120, 128, 136, 144,
            160, 176, 184, 192, 200, 208, 224, 240, 248, 256, 264, 272,
            288, 320, 352, 376, 384, 392, 400, 448, 480, 504, 512, 520
        ]

        # Warmup
        logger.info("  Warming up...")
        dummy_ids = torch.randint(0, vocab_size, (64,), device=device)
        for _ in range(5):
            with torch.no_grad():
                _ = model_runner.model.embed_tokens(dummy_ids)
        torch.cuda.synchronize()

        # Profile each token count
        latency_data = []
        num_runs = 10

        for n_tokens in token_counts:
            # Create dummy input
            dummy_ids = torch.randint(0, vocab_size, (n_tokens,), device=device)

            # Warmup for this size
            with torch.no_grad():
                for _ in range(3):
                    _ = model_runner.model.embed_tokens(dummy_ids)
            torch.cuda.synchronize()

            # Measure
            latencies = []
            for _ in range(num_runs):
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                with torch.no_grad():
                    # Run through embedding and first few layers
                    # This captures the key compute that determines tile efficiency
                    hidden = model_runner.model.embed_tokens(dummy_ids)
                    # Run through first decoder layer if available
                    if hasattr(model_runner.model, 'layers') and len(model_runner.model.layers) > 0:
                        layer = model_runner.model.layers[0]
                        if hasattr(layer, 'mlp'):
                            # Just run the MLP which is the bottleneck
                            hidden = layer.mlp(hidden)
                torch.cuda.synchronize()
                latencies.append((time.perf_counter() - t0) * 1000)

            median_lat = np.median(latencies)
            latency_data.append((n_tokens, median_lat))

        logger.info(f"  Profiled {len(token_counts)} token counts")

        # Fit latency model
        token_counts_arr = [t for t, _ in latency_data]
        latencies_arr = [l for _, l in latency_data]

        self.latency_model = PiecewiseLinearLatency()
        self.latency_model.fit(token_counts_arr, latencies_arr)

        logger.info(f"  Detected boundaries: {self.latency_model.boundaries}")

    # =========================================================================
    # Online Calibration Collection
    # =========================================================================

    def record_calibration(
        self,
        scores: torch.Tensor,
        accepted: torch.Tensor,
    ) -> bool:
        """
        Record calibration data from a verify() call.

        Args:
            scores: Draft confidence scores [bs, n_candidates]
            accepted: Boolean mask of accepted drafts [bs, n_candidates]

        Returns:
            True if calibration collection is complete
        """
        if not self._collecting_calibration:
            return False

        # Collect (score, accepted) pairs
        scores_np = scores.detach().cpu().numpy().flatten()
        accepted_np = accepted.detach().cpu().numpy().flatten()

        for s, a in zip(scores_np, accepted_np):
            self._calibration_data.append((float(s), bool(a)))

        # Check if we have enough data
        if len(self._calibration_data) >= self.min_samples * 10:  # ~10 tokens per request
            self._finish_calibration()
            return True

        return False

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
        else:
            self.calibration = Calibration()
            logger.info("TileSpec: No calibration data collected, using default")

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
    # Legacy API (for compatibility)
    # =========================================================================

    def start_profiling(self):
        """Legacy: Start collecting profiling data."""
        logger.warning("start_profiling() is deprecated - use profile_at_init() instead")

    def record(
        self,
        num_tokens: int,
        latency_ms: float,
        scores: Optional[torch.Tensor] = None,
        accepted: Optional[torch.Tensor] = None,
    ):
        """Legacy: Record one verification sample."""
        # Only process calibration data if collecting
        if scores is not None and accepted is not None:
            self.record_calibration(scores, accepted)
