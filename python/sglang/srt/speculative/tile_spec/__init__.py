"""
Tile-aware dynamic speculation for speculative decoding.

This module optimizes draft token count based on GPU tile boundaries
and acceptance probability to maximize throughput.
"""

from sglang.srt.speculative.tile_spec.core import (
    Calibration,
    PiecewiseLinearLatency,
    compute_optimal_k,
)
from sglang.srt.speculative.tile_spec.profiler import (
    TileSpecProfiler,
    download_sharegpt,
    get_cache_dir,
    load_sharegpt_prompts,
)

__all__ = [
    "Calibration",
    "PiecewiseLinearLatency",
    "compute_optimal_k",
    "TileSpecProfiler",
    "download_sharegpt",
    "get_cache_dir",
    "load_sharegpt_prompts",
]
