"""
Tile-Spec Online Profiler

Collects latency and calibration data during actual inference,
then fits models after enough samples are collected.

Cache structure: tile_spec/cache/{model}_{gpu}_{tp}/
"""

import json
import logging
import os
import re
import time
import threading
from pathlib import Path
from typing import List, Optional, Tuple
from dataclasses import dataclass, field

import numpy as np
import torch

from sglang.srt.speculative.tile_spec.core import Calibration, PiecewiseLinearLatency

logger = logging.getLogger(__name__)

# ShareGPT download URL
SHAREGPT_URL = "https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json"


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


def download_sharegpt(cache_dir: Path) -> Path:
    """Download ShareGPT dataset if not cached."""
    dataset_path = cache_dir / "sharegpt.json"
    if dataset_path.exists():
        return dataset_path

    import urllib.request

    logger.info(f"Downloading ShareGPT dataset to {dataset_path}...")
    cache_dir.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(SHAREGPT_URL, dataset_path)
    logger.info("Download complete.")
    return dataset_path


def load_sharegpt_prompts(dataset_path: Path, num_samples: int = 500) -> List[str]:
    """Load prompts from ShareGPT dataset."""
    with open(dataset_path) as f:
        dataset = json.load(f)

    # Filter conversations with at least 2 turns
    dataset = [
        data for data in dataset
        if len(data.get("conversations", data.get("conversation", []))) >= 2
    ]

    # Extract first turn (user prompt)
    prompts = [
        data.get("conversations", data.get("conversation", []))[0]["value"]
        for data in dataset[:num_samples]
    ]

    return prompts


@dataclass
class OnlineProfiler:
    """
    Collects latency and calibration samples during actual inference.

    Usage:
        profiler = OnlineProfiler(cache_dir, min_latency_samples=200)

        # During verification:
        profiler.record_latency(num_tokens, latency_ms)
        profiler.record_calibration(scores, accepted)

        # Check if ready:
        if profiler.is_ready():
            latency_model, calibration = profiler.fit_models()
    """
    cache_dir: Path
    min_latency_samples: int = 200
    min_calibration_samples: int = 500

    # Collected samples
    latency_tokens: List[int] = field(default_factory=list)
    latency_times: List[float] = field(default_factory=list)
    calibration_scores: List[float] = field(default_factory=list)
    calibration_accepted: List[int] = field(default_factory=list)

    # State
    _fitted: bool = False
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def record_latency(self, num_tokens: int, latency_ms: float):
        """Record a latency sample from verification."""
        with self._lock:
            self.latency_tokens.append(num_tokens)
            self.latency_times.append(latency_ms)

    def record_calibration(self, scores: List[float], accepted: List[int]):
        """Record calibration samples from verification."""
        with self._lock:
            self.calibration_scores.extend(scores)
            self.calibration_accepted.extend(accepted)

    def has_enough_latency_samples(self) -> bool:
        """Check if we have enough latency samples."""
        return len(self.latency_tokens) >= self.min_latency_samples

    def has_enough_calibration_samples(self) -> bool:
        """Check if we have enough calibration samples."""
        return len(self.calibration_scores) >= self.min_calibration_samples

    def is_ready(self) -> bool:
        """Check if profiling is complete."""
        return self.has_enough_latency_samples() and self.has_enough_calibration_samples()

    def fit_models(self) -> Tuple[PiecewiseLinearLatency, Calibration]:
        """Fit latency and calibration models from collected data."""
        with self._lock:
            # Fit latency model
            latency_model = PiecewiseLinearLatency()
            latency_model.fit(self.latency_tokens, self.latency_times)

            # Fit calibration model
            calibration = Calibration()
            if self.calibration_scores:
                scores_arr = np.array(self.calibration_scores)
                accepted_arr = np.array(self.calibration_accepted)
                calibration.fit(scores_arr, accepted_arr)

            # Save to cache
            self._save_models(latency_model, calibration)

            self._fitted = True
            return latency_model, calibration

    def _save_models(self, latency_model: PiecewiseLinearLatency, calibration: Calibration):
        """Save fitted models to cache."""
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        latency_path = self.cache_dir / "latency_model.npz"
        calibration_path = self.cache_dir / "calibration.npz"

        latency_model.save(str(latency_path))
        calibration.save(str(calibration_path))

        # Also save raw data for debugging
        raw_path = self.cache_dir / "profiling_data.npz"
        np.savez(
            raw_path,
            latency_tokens=self.latency_tokens,
            latency_times=self.latency_times,
            calibration_scores=self.calibration_scores,
            calibration_accepted=self.calibration_accepted,
        )

        logger.info(f"Saved tile-spec models to {self.cache_dir}")


class TileSpecProfiler:
    """
    Main profiler class - manages online profiling and caching.
    """

    def __init__(self, server_args):
        self.server_args = server_args
        self.gpu_name = torch.cuda.get_device_name(0)
        self.cache_dir = get_cache_dir(
            server_args.model_path,
            self.gpu_name,
            server_args.tp_size
        )
        logger.info(f"Tile-spec cache dir: {self.cache_dir}")

        # Check for cached models
        self.latency_model = None
        self.calibration = None
        self._load_cached_models()

        # Online profiler (used if no cache)
        self.online_profiler: Optional[OnlineProfiler] = None
        if self.latency_model is None:
            self.online_profiler = OnlineProfiler(self.cache_dir)
            logger.info("No cached models found - will profile online during inference")

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

                logger.info(f"Loaded tile-spec models from cache: {self.cache_dir}")
                logger.info(f"  Latency boundaries: {self.latency_model.boundaries}")
                return True
            except Exception as e:
                logger.warning(f"Failed to load cached models: {e}")
                self.latency_model = None
                self.calibration = None

        return False

    def is_profiling(self) -> bool:
        """Check if we're still collecting profiling data."""
        return self.online_profiler is not None and not self.online_profiler._fitted

    def record_verification(
        self,
        num_tokens: int,
        latency_ms: float,
        scores: Optional[List[float]] = None,
        accepted: Optional[List[int]] = None,
    ):
        """
        Record verification data for profiling.
        Call this from eagle_worker.verify() during inference.
        """
        if self.online_profiler is None:
            return

        self.online_profiler.record_latency(num_tokens, latency_ms)

        if scores is not None and accepted is not None:
            self.online_profiler.record_calibration(scores, accepted)

        # Check if ready to fit
        if self.online_profiler.is_ready() and not self.online_profiler._fitted:
            logger.info("Collected enough samples - fitting tile-spec models...")
            self.latency_model, self.calibration = self.online_profiler.fit_models()
            logger.info(f"Latency boundaries: {self.latency_model.boundaries}")
            logger.info(f"Optimal k candidates: {self.latency_model.get_optimal_k_candidates()}")

    def get_models(self) -> Tuple[Optional[PiecewiseLinearLatency], Optional[Calibration]]:
        """Get fitted models (or None if not ready yet)."""
        return self.latency_model, self.calibration
