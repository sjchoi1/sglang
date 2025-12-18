"""
Tile-Spec Profiler - Automatic profiling at engine initialization.

Profiles latency by recording actual verify() calls during warmup.
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


def get_cache_dir(model_path: str, gpu_name: str, tp_size: int) -> Path:
    """Get cache directory path."""
    model_name = re.sub(r"[^a-z0-9-]", "", model_path.split("/")[-1].lower())
    gpu_short = re.sub(r"[^a-z0-9]", "", gpu_name.lower().replace("nvidia ", ""))
    cache_name = f"{model_name}_{gpu_short}_tp{tp_size}"

    # Try project root first
    current = Path(__file__).parent
    while current != current.parent:
        if (current / "tile_spec").exists():
            return current / "tile_spec" / "cache" / cache_name
        current = current.parent
    return Path.home() / ".cache" / "sglang" / "tile_spec" / cache_name


def _load_prompts(cache_dir: Path, limit: int = 100) -> List[str]:
    """Load ShareGPT prompts, with synthetic fallback."""
    sharegpt_path = cache_dir / "sharegpt.json"

    if not sharegpt_path.exists():
        cache_dir.mkdir(parents=True, exist_ok=True)
        try:
            logger.info("Downloading ShareGPT for profiling...")
            urllib.request.urlretrieve(SHAREGPT_URL, sharegpt_path)
        except Exception as e:
            logger.warning(f"ShareGPT download failed: {e}")
            return [f"Write a story about topic {i}." for i in range(limit)]

    try:
        with open(sharegpt_path) as f:
            data = json.load(f)
        prompts = []
        for item in data:
            for conv in item.get("conversations", []):
                if conv.get("from") == "human":
                    text = conv.get("value", "").strip()
                    if 50 < len(text) < 2000:
                        prompts.append(text)
                        if len(prompts) >= limit:
                            return prompts
        return prompts or [f"Write a story about topic {i}." for i in range(limit)]
    except Exception:
        return [f"Write a story about topic {i}." for i in range(limit)]


class TileSpecProfiler:
    """Profiles tile-spec latency using actual sglang runs."""

    def __init__(self, server_args, min_samples: int = 50):
        self.server_args = server_args
        self.cache_dir = get_cache_dir(
            server_args.model_path,
            torch.cuda.get_device_name(0),
            server_args.tp_size,
        )
        self.min_samples = min_samples
        self._profiling = False
        self._latency_data: List[Tuple[int, float]] = []
        self._calibration_data: List[Tuple[float, bool]] = []

        self.latency_model: Optional[PiecewiseLinearLatency] = None
        self.calibration: Optional[Calibration] = None
        self._load_cache()

    def _load_cache(self) -> bool:
        """Load models from cache."""
        latency_path = self.cache_dir / "latency_model.npz"
        if not latency_path.exists():
            return False
        try:
            self.latency_model = PiecewiseLinearLatency()
            self.latency_model.load(str(latency_path))
            self.calibration = Calibration()
            calib_path = self.cache_dir / "calibration.npz"
            if calib_path.exists():
                self.calibration.load(str(calib_path))
            logger.info(f"TileSpec: Loaded from cache, boundaries={self.latency_model.boundaries}")
            return True
        except Exception as e:
            logger.warning(f"Cache load failed: {e}")
            self.latency_model = None
            self.calibration = None
            return False

    def needs_profiling(self) -> bool:
        return self.latency_model is None

    def is_profiling(self) -> bool:
        return self._profiling

    def get_models(self) -> Tuple[Optional[PiecewiseLinearLatency], Optional[Calibration]]:
        return self.latency_model, self.calibration

    def start_profiling(self):
        """Start collecting latency data."""
        if not self.needs_profiling():
            return
        self._profiling = True
        self._latency_data.clear()
        self._calibration_data.clear()
        logger.info("TileSpec: Started profiling")

    def record(self, num_tokens: int, latency_ms: float, scores=None, accepted=None):
        """Record latency and calibration data from verify() call."""
        if not self._profiling:
            return
        self._latency_data.append((num_tokens, latency_ms))

        # Collect calibration data (score, accepted) pairs
        if scores is not None and accepted is not None:
            scores_np = scores.detach().cpu().numpy().flatten()
            accepted_np = accepted.detach().cpu().numpy().flatten()
            for s, a in zip(scores_np, accepted_np):
                self._calibration_data.append((float(s), bool(a)))

        # Auto-finish when enough diverse samples
        if len(self._latency_data) >= self.min_samples:
            unique = len(set(t for t, _ in self._latency_data))
            if unique >= 4:
                self.finish_profiling()

    def finish_profiling(self):
        """Fit latency model from collected data."""
        if not self._profiling:
            return
        self._profiling = False

        if len(self._latency_data) < 10:
            logger.warning(f"TileSpec: Only {len(self._latency_data)} samples")
            return

        # Aggregate by token count
        by_tokens = defaultdict(list)
        for n, lat in self._latency_data:
            by_tokens[n].append(lat)

        token_counts = sorted(by_tokens.keys())
        latencies = [np.median(by_tokens[t]) for t in token_counts]

        logger.info(f"TileSpec: Fitting from {len(self._latency_data)} samples")
        logger.info(f"  Token counts: {token_counts}")

        self.latency_model = PiecewiseLinearLatency()
        self.latency_model.fit(token_counts, latencies)

        # Fit calibration from collected data
        self.calibration = Calibration()
        if self._calibration_data:
            cal_scores = np.array([s for s, _ in self._calibration_data])
            cal_accepted = np.array([a for _, a in self._calibration_data])
            self.calibration.fit(cal_scores, cal_accepted)
            logger.info(f"  Calibration: {len(self._calibration_data)} samples")

        # Save cache
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.latency_model.save(str(self.cache_dir / "latency_model.npz"))
        self.calibration.save(str(self.cache_dir / "calibration.npz"))

        logger.info(f"TileSpec: Complete, boundaries={self.latency_model.boundaries}")
        self._latency_data.clear()
        self._calibration_data.clear()

    def run_warmup(self, generate_func):
        """Run warmup profiling with varying batch sizes."""
        if not self.needs_profiling():
            return

        logger.info("TileSpec: Running warmup profiling...")
        prompts = _load_prompts(self.cache_dir, limit=100)
        self.start_profiling()

        # Fewer batches: [1, 4, 16, 64] x 8 each = 32 batches total
        batch_sizes = [1, 4, 16, 64]
        idx = 0
        for bs in batch_sizes:
            for _ in range(8):
                batch = [prompts[(idx + j) % len(prompts)] for j in range(bs)]
                idx += bs
                try:
                    generate_func(batch, sampling_params={"temperature": 0, "max_new_tokens": 32})
                except Exception as e:
                    logger.warning(f"Warmup failed: {e}")

        self.finish_profiling()
