"""
Tile-Spec Profiler - Automatic profiling at engine initialization.

Profiles latency and calibration by running actual sglang generation:
- Latency: Records verify() timing at varying batch sizes
- Calibration: Records (score, accepted) pairs from draft verification

Uses ShareGPT dataset for diverse prompts.
Cache structure: tile_spec/cache/{model}_{gpu}_{tp}/
"""

import json
import logging
import re
import time
import urllib.request
from collections import defaultdict
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch

from sglang.srt.speculative.tile_spec.core import Calibration, PiecewiseLinearLatency

logger = logging.getLogger(__name__)

# ShareGPT dataset URL
SHAREGPT_URL = "https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json"


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


def _download_sharegpt(cache_dir: Path, limit: int = 500) -> List[str]:
    """Download and extract ShareGPT prompts."""
    sharegpt_path = cache_dir / "sharegpt.json"

    if not sharegpt_path.exists():
        logger.info("Downloading ShareGPT dataset...")
        cache_dir.mkdir(parents=True, exist_ok=True)
        try:
            urllib.request.urlretrieve(SHAREGPT_URL, sharegpt_path)
        except Exception as e:
            logger.warning(f"Failed to download ShareGPT: {e}")
            return _get_synthetic_prompts(limit)

    try:
        with open(sharegpt_path) as f:
            data = json.load(f)

        prompts = []
        for item in data:
            if "conversations" in item:
                for conv in item["conversations"]:
                    if conv.get("from") == "human" and conv.get("value"):
                        text = conv["value"].strip()
                        if 50 < len(text) < 2000:
                            prompts.append(text)
                            if len(prompts) >= limit:
                                return prompts
        return prompts if prompts else _get_synthetic_prompts(limit)
    except Exception as e:
        logger.warning(f"Failed to load ShareGPT: {e}")
        return _get_synthetic_prompts(limit)


def _get_synthetic_prompts(limit: int) -> List[str]:
    """Generate synthetic prompts as fallback."""
    return [
        f"Write a detailed story about adventure number {i}. Include characters, plot, and a conclusion."
        for i in range(limit)
    ]


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

        if not latency_path.exists():
            return False

        try:
            self.latency_model = PiecewiseLinearLatency()
            self.latency_model.load(str(latency_path))

            if calibration_path.exists():
                self.calibration = Calibration()
                self.calibration.load(str(calibration_path))
            else:
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
        return self._profiling

    def get_models(self) -> Tuple[Optional[PiecewiseLinearLatency], Optional[Calibration]]:
        return self.latency_model, self.calibration

    # =========================================================================
    # Recording during verify()
    # =========================================================================

    def start_profiling(self):
        """Start collecting data during verify() calls."""
        if not self.needs_profiling():
            return

        self._profiling = True
        self._latency_data.clear()
        self._calibration_data.clear()
        logger.info("TileSpec: Started profiling (recording verify() calls)")

    def record(
        self,
        num_tokens: int,
        latency_ms: float,
        scores: Optional[torch.Tensor] = None,
        accepted: Optional[torch.Tensor] = None,
    ):
        """Record data from a verify() call."""
        if not self._profiling:
            return

        # Record latency
        self._latency_data.append((num_tokens, latency_ms))

        # Record calibration data
        if scores is not None and accepted is not None:
            scores_np = scores.detach().cpu().numpy().flatten()
            accepted_np = accepted.detach().cpu().numpy().flatten()
            for s, a in zip(scores_np, accepted_np):
                self._calibration_data.append((float(s), bool(a)))

        # Log progress
        if len(self._latency_data) % 20 == 0:
            unique = len(set(t for t, _ in self._latency_data))
            logger.info(
                f"TileSpec: {len(self._latency_data)} samples, "
                f"{unique} unique token counts, "
                f"{len(self._calibration_data)} calibration points"
            )

        # Auto-finish when we have enough diverse samples
        if len(self._latency_data) >= self.min_samples:
            unique_counts = len(set(t for t, _ in self._latency_data))
            if unique_counts >= 5:  # Need at least 5 different token counts
                self.finish_profiling()

    def finish_profiling(self):
        """Finish profiling and fit models."""
        if not self._profiling:
            return

        self._profiling = False

        if len(self._latency_data) < 10:
            logger.warning(f"TileSpec: Only {len(self._latency_data)} samples, need more")
            return

        # Fit latency model
        latency_by_tokens = defaultdict(list)
        for num_tokens, latency_ms in self._latency_data:
            latency_by_tokens[num_tokens].append(latency_ms)

        token_counts = sorted(latency_by_tokens.keys())
        latencies = [np.median(latency_by_tokens[t]) for t in token_counts]

        logger.info(f"TileSpec: Fitting from {len(self._latency_data)} samples")
        logger.info(f"  Token counts: {token_counts}")

        self.latency_model = PiecewiseLinearLatency()
        self.latency_model.fit(token_counts, latencies)

        # Fit calibration model
        self.calibration = Calibration()
        if self._calibration_data:
            scores = np.array([s for s, _ in self._calibration_data])
            accepted = np.array([a for _, a in self._calibration_data])
            self.calibration.fit(scores, accepted)
            logger.info(f"  Calibration: {len(self._calibration_data)} points")

        # Save to cache
        self._save_cache()

        logger.info("=" * 60)
        logger.info("TileSpec: Profiling complete")
        logger.info(f"  Boundaries: {self.latency_model.boundaries}")
        logger.info(f"  Optimal k: {self.latency_model.get_optimal_k_candidates()}")
        logger.info("=" * 60)

        self._latency_data.clear()
        self._calibration_data.clear()

    def _save_cache(self):
        """Save models to cache."""
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        if self.latency_model:
            self.latency_model.save(str(self.cache_dir / "latency_model.npz"))

        if self.calibration:
            self.calibration.save(str(self.cache_dir / "calibration.npz"))

        logger.info(f"  Saved to: {self.cache_dir}")

    # =========================================================================
    # Warmup profiling with batched requests
    # =========================================================================

    def run_warmup_profiling(self, generate_func):
        """
        Run warmup profiling with varying batch sizes.

        Uses batched generate() calls to get measurements at different token counts.
        Downloads ShareGPT for diverse prompts.

        Args:
            generate_func: The Engine.generate() function
        """
        if not self.needs_profiling():
            return

        logger.info("=" * 60)
        logger.info("TileSpec: Running warmup profiling")
        logger.info("=" * 60)

        # Download ShareGPT prompts
        prompts = _download_sharegpt(self.cache_dir, limit=500)
        logger.info(f"  Loaded {len(prompts)} prompts")

        # Start profiling
        self.start_profiling()

        # Run with varying batch sizes to get different token counts
        # batch_size * draft_tokens = total tokens in verify()
        batch_sizes = [1, 2, 4, 8, 16, 32, 64]
        prompt_idx = 0

        for batch_size in batch_sizes:
            # Run multiple batches at each size
            num_batches = max(1, 100 // batch_size)

            for batch_idx in range(num_batches):
                # Get prompts for this batch
                batch_prompts = []
                for _ in range(batch_size):
                    batch_prompts.append(prompts[prompt_idx % len(prompts)])
                    prompt_idx += 1

                try:
                    # Send batch request
                    generate_func(
                        batch_prompts,
                        sampling_params={"temperature": 0, "max_new_tokens": 64},
                    )
                except Exception as e:
                    logger.warning(f"Batch request failed: {e}")

            logger.info(f"  Batch size {batch_size}: {num_batches} batches done")

        # Finish profiling
        self.finish_profiling()
