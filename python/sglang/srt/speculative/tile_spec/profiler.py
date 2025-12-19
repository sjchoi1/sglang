"""
Tile-Spec Profiler - Automatic profiling during warmup.

Profiles latency by recording actual verify() calls.
Cache structure: ~/.cache/sglang/tile_spec/{model}_{gpu}_tp{N}/
"""

import json
import logging
import re
import time
import urllib.request
from collections import defaultdict
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import numpy as np
import torch

from sglang.srt.speculative.tile_spec.core import Calibration, PiecewiseLinearLatency

logger = logging.getLogger(__name__)

SHAREGPT_URL = "https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json"
WARMUP_BATCH_SIZES = [1, 4, 16, 64]

# Cache location: ~/.cache/sglang/tile_spec/
TILE_SPEC_CACHE_ROOT = Path.home() / ".cache" / "sglang" / "tile_spec"


def get_cache_dir(model_path: str, gpu_name: str, tp_size: int) -> Path:
    """Get cache directory path for model-specific profiling data."""
    model_name = re.sub(r"[^a-z0-9-]", "", model_path.split("/")[-1].lower())
    gpu_short = re.sub(r"[^a-z0-9]", "", gpu_name.lower().replace("nvidia ", ""))
    cache_name = f"{model_name}_{gpu_short}_tp{tp_size}"
    return TILE_SPEC_CACHE_ROOT / cache_name


def _load_prompts(limit: int = 100) -> List[str]:
    """Load ShareGPT prompts for profiling warmup."""
    sharegpt_path = TILE_SPEC_CACHE_ROOT / "sharegpt.json"

    if not sharegpt_path.exists():
        TILE_SPEC_CACHE_ROOT.mkdir(parents=True, exist_ok=True)
        try:
            logger.info("TileSpec: Downloading ShareGPT dataset...")
            urllib.request.urlretrieve(SHAREGPT_URL, sharegpt_path)
        except Exception as e:
            logger.warning(f"TileSpec: ShareGPT download failed: {e}")
            return [f"Write a detailed story about topic number {i}." for i in range(limit)]

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
        return prompts or [f"Write a detailed story about topic number {i}." for i in range(limit)]
    except Exception:
        return [f"Write a detailed story about topic number {i}." for i in range(limit)]


def tile_spec_warmup(
    server_args,
    generate_fn: Callable[[List[str]], None],
    check_ready_fn: Callable[[], bool],
    max_wait: int = 60,
):
    """
    Run TileSpec warmup profiling through actual verify() path.

    Args:
        server_args: Server args with model_path and tp_size
        generate_fn: Function to send generation requests, signature: (prompts: List[str]) -> None
        check_ready_fn: Function that returns True when profiling is complete
        max_wait: Maximum seconds to wait for profiling to complete
    """
    if not getattr(server_args, 'tile_spec', False):
        return

    # Check if already profiled (cached)
    if check_ready_fn():
        return

    logger.info("TileSpec: Running warmup profiling...")

    # Load prompts from shared cache
    prompts = _load_prompts(limit=100)

    # Run warmup with varying batch sizes
    idx = 0
    for batch_size in WARMUP_BATCH_SIZES:
        batch = prompts[idx:idx + batch_size]
        idx += batch_size
        if not batch:
            break
        try:
            generate_fn(batch)
            logger.info(f"TileSpec: Warmup batch_size={batch_size} done")
        except Exception as e:
            logger.warning(f"TileSpec: Warmup batch failed: {e}")

    # Wait for profiling to complete (auto-finishes when enough samples)
    for _ in range(max_wait):
        if check_ready_fn():
            logger.info("TileSpec: Profiling complete")
            return
        time.sleep(1.0)

    logger.warning("TileSpec: Profiling timeout, continuing anyway")


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
