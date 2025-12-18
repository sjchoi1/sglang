"""
Tile-aware dynamic speculation for speculative decoding.

This module provides:
- Calibration: maps cumulative draft scores to acceptance probabilities
- PiecewiseLinearLatency: models verification latency with tile boundaries
- compute_optimal_k: finds optimal draft token count maximizing E/Latency
"""

from typing import List, Tuple, Optional
import torch
import numpy as np


class Calibration:
    """Maps cumulative draft confidence to acceptance probability."""

    def __init__(self, num_bins: int = 50):
        self.num_bins = num_bins
        self.bin_edges = np.linspace(0, 1, num_bins + 1)
        self.bin_probs = np.ones(num_bins) * 0.5  # default
        self._tensor_probs = None
        self._device = None

    def fit(self, scores: np.ndarray, accepted: np.ndarray):
        """Fit calibration from collected data."""
        for i in range(self.num_bins):
            lo, hi = self.bin_edges[i], self.bin_edges[i + 1]
            mask = (scores >= lo) & (scores < hi)
            if mask.sum() > 0:
                self.bin_probs[i] = accepted[mask].mean()
        self._tensor_probs = None  # invalidate cache

    def predict(self, scores: torch.Tensor) -> torch.Tensor:
        """Map scores to acceptance probabilities."""
        device = scores.device

        # Cache tensor on device
        if self._tensor_probs is None or self._device != device:
            self._tensor_probs = torch.from_numpy(self.bin_probs).float().to(device)
            self._device = device

        # Clamp and bucketize
        scores_clamped = scores.clamp(0, 1 - 1e-6)
        bin_idx = (scores_clamped * self.num_bins).long().clamp(0, self.num_bins - 1)
        return self._tensor_probs[bin_idx]

    def save(self, path: str):
        np.savez(path, bin_edges=self.bin_edges, bin_probs=self.bin_probs)

    def load(self, path: str):
        data = np.load(path)
        self.bin_edges = data['bin_edges']
        self.bin_probs = data['bin_probs']
        self.num_bins = len(self.bin_probs)
        self._tensor_probs = None


class PiecewiseLinearLatency:
    """
    Latency model with automatic boundary detection.

    Supports two modes:
    - Piecewise linear: fits linear regression per segment (default)
    - LUT: stores exact latency values for all measured token counts

    When use_lut=True, predict() returns exact measured values when available,
    and falls back to piecewise linear interpolation for unmeasured counts.
    """

    def __init__(self, use_lut: bool = False):
        """
        Args:
            use_lut: If True, store exact latencies and use them for prediction.
                     Falls back to piecewise linear for unmeasured token counts.
        """
        self.use_lut = use_lut
        self.boundaries: List[int] = []
        self.slopes: List[float] = []
        self.intercepts: List[float] = []
        # LUT: exact latencies at measured token counts
        self.lut: dict = {}  # token_count -> latency
        # Store boundary latencies separately for fast lookup
        self.boundary_latencies: dict = {}  # boundary -> latency

    def fit(
        self,
        token_counts: List[int],
        latencies: List[float],
        jump_threshold: float = 0.15,
    ):
        """
        Fit piecewise linear model from measurements.

        Args:
            token_counts: list of token counts profiled
            latencies: corresponding latencies (ms)
            jump_threshold: relative jump to detect boundary (0.15 = 15%)
        """
        # Sort by token count
        sorted_pairs = sorted(zip(token_counts, latencies))
        tokens = np.array([p[0] for p in sorted_pairs])
        lats = np.array([p[1] for p in sorted_pairs])

        # Store LUT if enabled
        if self.use_lut:
            self.lut = {int(t): float(l) for t, l in zip(tokens, lats)}

        # Detect boundaries
        self.boundaries = [int(tokens[0])]
        for i in range(1, len(tokens)):
            if lats[i-1] > 0 and (lats[i] - lats[i-1]) / lats[i-1] > jump_threshold:
                self.boundaries.append(int(tokens[i]))
        self.boundaries.append(int(tokens[-1]) + 1)

        # Store boundary latencies (optimal k values = boundaries[:-1])
        # For each boundary, store the latency at that exact token count
        for b in self.boundaries[:-1]:
            idx = np.where(tokens == b)[0]
            if len(idx) > 0:
                self.boundary_latencies[b] = float(lats[idx[0]])
            elif b > 0:
                # Find closest measured point at or below boundary
                below = tokens[tokens <= b]
                if len(below) > 0:
                    closest_idx = np.argmax(below)
                    self.boundary_latencies[b] = float(lats[closest_idx])

        # Fit linear regression per segment
        self.slopes = []
        self.intercepts = []

        for i in range(len(self.boundaries) - 1):
            lo, hi = self.boundaries[i], self.boundaries[i + 1]
            mask = (tokens >= lo) & (tokens < hi)

            if mask.sum() >= 2:
                X, y = tokens[mask], lats[mask]
                slope = np.cov(X, y)[0, 1] / (np.var(X) + 1e-9)
                intercept = y.mean() - slope * X.mean()
            elif mask.sum() == 1:
                slope, intercept = 0.0, lats[mask][0]
            else:
                slope, intercept = 0.0, 0.0

            self.slopes.append(float(slope))
            self.intercepts.append(float(intercept))

    def predict(self, n: int) -> float:
        """
        Predict latency for n tokens.

        If use_lut=True and n is in LUT, returns exact value.
        Otherwise uses piecewise linear interpolation.
        """
        # Check boundary latencies first (fast path for optimal k search)
        if n in self.boundary_latencies:
            return self.boundary_latencies[n]

        # Check LUT if enabled
        if self.use_lut and n in self.lut:
            return self.lut[n]

        # Fall back to piecewise linear
        for i in range(len(self.boundaries) - 1):
            if self.boundaries[i] <= n < self.boundaries[i + 1]:
                return self.slopes[i] * n + self.intercepts[i]

        # Extrapolate from last segment
        if self.slopes:
            return self.slopes[-1] * n + self.intercepts[-1]
        return 1.0  # fallback

    def get_boundaries(self) -> List[int]:
        """Return tile boundaries (segment start points)."""
        return self.boundaries[:-1]

    def get_optimal_k_candidates(self) -> List[int]:
        """
        Return optimal k values for searching.

        These are the END of each segment minus 1, which gives
        maximum tokens before the next latency jump.

        Example: boundaries = [1, 65, 129, 193, 257, 385, 513]
        Returns: [64, 128, 192, 256, 384, 512]
        """
        # Skip the first boundary (usually 1) and return boundary[i] - 1
        candidates = [b - 1 for b in self.boundaries[1:]]
        return candidates

    def save(self, path: str):
        np.savez(
            path,
            boundaries=self.boundaries,
            slopes=self.slopes,
            intercepts=self.intercepts,
            use_lut=self.use_lut,
            lut_keys=list(self.lut.keys()) if self.lut else [],
            lut_values=list(self.lut.values()) if self.lut else [],
            boundary_latencies_keys=list(self.boundary_latencies.keys()),
            boundary_latencies_values=list(self.boundary_latencies.values()),
        )

    def load(self, path: str):
        data = np.load(path)
        self.boundaries = data['boundaries'].tolist()
        self.slopes = data['slopes'].tolist()
        self.intercepts = data['intercepts'].tolist()
        self.use_lut = bool(data.get('use_lut', False))

        # Load LUT if present
        lut_keys = data.get('lut_keys', [])
        lut_values = data.get('lut_values', [])
        if len(lut_keys) > 0:
            self.lut = {int(k): float(v) for k, v in zip(lut_keys, lut_values)}

        # Load boundary latencies
        bl_keys = data.get('boundary_latencies_keys', [])
        bl_values = data.get('boundary_latencies_values', [])
        if len(bl_keys) > 0:
            self.boundary_latencies = {int(k): float(v) for k, v in zip(bl_keys, bl_values)}


def compute_optimal_k(
    score_list: torch.Tensor,
    calibration: Calibration,
    latency_model: PiecewiseLinearLatency,
    prefill_tokens: int = 0,
    max_k: int = 256,
) -> int:
    """
    Find optimal draft token count maximizing E[accepted] / Latency.

    Args:
        score_list: [bs, n_candidates] cumulative draft scores
        calibration: maps scores to acceptance probability
        latency_model: piecewise linear latency predictor
        prefill_tokens: tokens from prefill (for mixed batches)
        max_k: maximum draft tokens allowed

    Returns:
        Optimal total draft tokens for the batch
    """
    bs, n_cand = score_list.shape

    # Calibrate scores to acceptance probabilities
    probs = calibration.predict(score_list)

    # Sort globally by probability (descending)
    flat_probs = probs.flatten()
    sorted_probs, _ = torch.sort(flat_probs, descending=True)

    # Cumulative expected accepted tokens
    cum_E = torch.cumsum(sorted_probs, dim=0)

    # Search over optimal k candidates (end of each tile segment)
    candidates = latency_model.get_optimal_k_candidates()
    max_tokens = min(max_k, len(sorted_probs))

    best_k = min(candidates[0], max_tokens) if candidates else 32
    best_ratio = 0.0

    for k in candidates:
        if k <= 0 or k > max_tokens:
            continue

        # E = expected accepted drafts + bonus tokens (1 per request)
        E_total = cum_E[k - 1].item() + bs

        # Latency for total tokens through model
        L = latency_model.predict(prefill_tokens + k)

        if L > 0:
            ratio = E_total / L
            if ratio > best_ratio:
                best_ratio = ratio
                best_k = k

    return max(8, best_k)
