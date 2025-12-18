"""
Tile-aware dynamic speculation for speculative decoding.

Optimizes draft token count based on GPU tile boundaries
and acceptance probability to maximize throughput.
"""

from sglang.srt.speculative.tile_spec.core import (
    Calibration,
    PiecewiseLinearLatency,
    compute_optimal_k,
)
from sglang.srt.speculative.tile_spec.profiler import (
    TileSpecProfiler,
    get_cache_dir,
)

__all__ = [
    "Calibration",
    "PiecewiseLinearLatency",
    "compute_optimal_k",
    "TileSpecProfiler",
    "get_cache_dir",
]
