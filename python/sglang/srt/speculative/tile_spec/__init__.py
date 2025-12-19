"""
Tile-aware dynamic speculation for speculative decoding.

Optimizes draft token count based on GPU tile boundaries
and acceptance probability to maximize throughput.
"""

from sglang.srt.speculative.tile_spec.core import (
    Calibration,
    PiecewiseLinearLatency,
    compute_optimal_k,
    find_draft_cutoffs,
)
from sglang.srt.speculative.tile_spec.profiler import (
    TileSpecProfiler,
    get_cache_dir,
    tile_spec_warmup,
)

__all__ = [
    "Calibration",
    "PiecewiseLinearLatency",
    "compute_optimal_k",
    "find_draft_cutoffs",
    "TileSpecProfiler",
    "get_cache_dir",
    "tile_spec_warmup",
]
