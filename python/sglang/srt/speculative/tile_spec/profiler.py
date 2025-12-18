"""
Tile-Spec Profiler - Online profiling during actual speculation path.

Profiles latency and calibration by recording data during real verify() calls.
Uses ShareGPT dataset for warmup traffic.

Cache structure: tile_spec/cache/{model}_{gpu}_{tp}/
"""

import json
import logging
import re
import urllib.request
from collections import defaultdict
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch

from sglang.srt.speculative.tile_spec.core import Calibration, PiecewiseLinearLatency

logger = logging.getLogger(__name__)

SHAREGPT_URL = "https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json"


def download_sharegpt(cache_dir: Path) -> Path:
    """Download ShareGPT dataset if not cached."""
    dataset_path = cache_dir / "sharegpt.json"
    if dataset_path.exists():
        return dataset_path

    cache_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Downloading ShareGPT dataset to {dataset_path}...")

    try:
        urllib.request.urlretrieve(SHAREGPT_URL, dataset_path)
        logger.info("Download complete.")
    except Exception as e:
        logger.error(f"Failed to download ShareGPT: {e}")
        raise

    return dataset_path


def load_sharegpt_prompts(dataset_path: Path, num_prompts: int = 200) -> List[str]:
    """Load prompts from ShareGPT dataset."""
    with open(dataset_path) as f:
        data = json.load(f)

    prompts = []
    for item in data:
        if "conversations" in item:
            for conv in item["conversations"]:
                if conv.get("from") == "human" and conv.get("value"):
                    text = conv["value"].strip()
                    # Filter reasonable length prompts
                    if 50 < len(text) < 2000:
                        prompts.append(text)
                        if len(prompts) >= num_prompts:
                            return prompts
    return prompts


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


class OnlineProfiler:
    """Collects latency and calibration data during actual verify() calls."""

    def __init__(self, min_samples: int = 100):
        self.min_samples = min_samples
        self.latency_data: List[Tuple[int, float]] = []  # (num_tokens, latency_ms)
        self.calibration_data: List[Tuple[float, bool]] = []  # (score, accepted)
        self._profiling = True

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

        self.latency_data.append((num_tokens, latency_ms))

        # Record calibration data if provided
        if scores is not None and accepted is not None:
            scores_np = scores.detach().cpu().numpy().flatten()
            accepted_np = accepted.detach().cpu().numpy().flatten()
            for s, a in zip(scores_np, accepted_np):
                self.calibration_data.append((float(s), bool(a)))

        # Check if we have enough samples
        if len(self.latency_data) >= self.min_samples:
            self._profiling = False
            logger.info(
                f"Profiling complete: {len(self.latency_data)} latency samples, "
                f"{len(self.calibration_data)} calibration samples"
            )

    def is_profiling(self) -> bool:
        return self._profiling

    def fit_models(self) -> Tuple[PiecewiseLinearLatency, Calibration]:
        """Fit latency and calibration models from collected data."""
        # Aggregate latency data by token count (take median)
        latency_by_tokens = defaultdict(list)
        for num_tokens, latency_ms in self.latency_data:
            latency_by_tokens[num_tokens].append(latency_ms)

        token_counts = sorted(latency_by_tokens.keys())
        latencies = [np.median(latency_by_tokens[t]) for t in token_counts]

        logger.info(f"Fitting latency model from {len(token_counts)} unique token counts")

        # Fit latency model
        latency_model = PiecewiseLinearLatency()
        latency_model.fit(token_counts, latencies)

        # Fit calibration model
        calibration = Calibration()
        if self.calibration_data:
            scores = np.array([s for s, _ in self.calibration_data])
            accepted = np.array([a for _, a in self.calibration_data])
            calibration.fit(scores, accepted)
            logger.info(f"Fitted calibration from {len(scores)} samples")
        else:
            logger.info("No calibration data - using default calibration")

        return latency_model, calibration


class TileSpecProfiler:
    """Manages tile-spec profiling and cache."""

    def __init__(self, server_args):
        self.server_args = server_args
        self.gpu_name = torch.cuda.get_device_name(0)
        self.cache_dir = get_cache_dir(
            server_args.model_path, self.gpu_name, server_args.tp_size
        )
        logger.info(f"Tile-spec cache dir: {self.cache_dir}")

        # Models (loaded from cache or fitted from profiling)
        self.latency_model: Optional[PiecewiseLinearLatency] = None
        self.calibration: Optional[Calibration] = None

        # Online profiler (used during warmup)
        self.online_profiler: Optional[OnlineProfiler] = None

        # Try to load from cache
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

                logger.info("Loaded tile-spec models from cache")
                logger.info(f"  Latency boundaries: {self.latency_model.boundaries}")
                return True
            except Exception as e:
                logger.warning(f"Failed to load cached models: {e}")
                self.latency_model = None
                self.calibration = None

        return False

    def needs_profiling(self) -> bool:
        """Check if profiling is needed (no cache)."""
        return self.latency_model is None

    def start_profiling(self, min_samples: int = 100):
        """Start online profiling mode."""
        self.online_profiler = OnlineProfiler(min_samples=min_samples)
        logger.info(f"Started tile-spec profiling (need {min_samples} samples)")

    def is_profiling(self) -> bool:
        """Check if currently profiling."""
        return self.online_profiler is not None and self.online_profiler.is_profiling()

    def record_verification(
        self,
        num_tokens: int,
        latency_ms: float,
        scores: Optional[torch.Tensor] = None,
        accepted: Optional[torch.Tensor] = None,
    ):
        """Record verification data during profiling."""
        if self.online_profiler:
            self.online_profiler.record(num_tokens, latency_ms, scores, accepted)

    def finish_profiling(self):
        """Finish profiling, fit models, and save to cache."""
        if not self.online_profiler:
            return

        # Fit models
        self.latency_model, self.calibration = self.online_profiler.fit_models()

        # Save to cache
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.latency_model.save(str(self.cache_dir / "latency_model.npz"))
        self.calibration.save(str(self.cache_dir / "calibration.npz"))

        logger.info(f"Saved tile-spec models to {self.cache_dir}")
        logger.info(f"  Latency boundaries: {self.latency_model.boundaries}")

        self.online_profiler = None

    def get_models(self) -> Tuple[Optional[PiecewiseLinearLatency], Optional[Calibration]]:
        """Get fitted models (or None if not available)."""
        return self.latency_model, self.calibration
