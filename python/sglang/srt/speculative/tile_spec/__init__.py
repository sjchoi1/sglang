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

__all__ = [
    "Calibration",
    "PiecewiseLinearLatency",
    "compute_optimal_k",
]
