"""
Core tile-aware speculation algorithms.

- Calibration: maps cumulative draft scores to acceptance probabilities (linear regression)
- PiecewiseLinearLatency: models verification latency with tile boundaries
- find_optimal_cutoff: finds optimal draft token count maximizing E[accepted]/Latency
"""

from typing import List, Tuple

import numpy as np
import torch


class Calibration:
    """Maps cumulative draft scores to acceptance probability using linear regression."""

    def __init__(self):
        # Linear regression: P(accept) = slope * score + intercept
        self.slope = 1.0  # default: identity-ish mapping
        self.intercept = 0.0
        self._slope_tensor = None
        self._intercept_tensor = None
        self._device = None

    def fit(self, scores: np.ndarray, accepted: np.ndarray):
        """Fit linear regression from collected (cumulative score, accepted) pairs."""
        if len(scores) < 2:
            return

        # Simple linear regression: y = a*x + b
        x_mean = scores.mean()
        y_mean = accepted.mean()

        # Compute slope: cov(x,y) / var(x)
        numerator = ((scores - x_mean) * (accepted - y_mean)).sum()
        denominator = ((scores - x_mean) ** 2).sum()

        if denominator > 1e-9:
            self.slope = float(numerator / denominator)
            self.intercept = float(y_mean - self.slope * x_mean)
        else:
            # Fallback if no variance in scores
            self.slope = 0.0
            self.intercept = float(y_mean)

        # Invalidate cached tensors
        self._slope_tensor = None
        self._intercept_tensor = None

    def predict(self, scores: torch.Tensor) -> torch.Tensor:
        """Map cumulative scores to acceptance probabilities."""
        device = scores.device

        # Cache tensors on device
        if self._slope_tensor is None or self._device != device:
            self._slope_tensor = torch.tensor(self.slope, dtype=torch.float32, device=device)
            self._intercept_tensor = torch.tensor(self.intercept, dtype=torch.float32, device=device)
            self._device = device

        # Linear prediction with clamp to [0.01, 0.99]
        probs = self._slope_tensor * scores + self._intercept_tensor
        return probs.clamp(0.01, 0.99)

    def save(self, path: str):
        np.savez(path, slope=self.slope, intercept=self.intercept)

    def load(self, path: str):
        data = np.load(path)
        self.slope = float(data["slope"])
        self.intercept = float(data["intercept"])
        self._slope_tensor = None
        self._intercept_tensor = None


class PiecewiseLinearLatency:
    """
    Latency model with automatic boundary detection.

    Detects tile boundaries via latency jumps and fits linear regression
    per segment for interpolation.
    """

    def __init__(self):
        self.boundaries: List[int] = []
        self.slopes: List[float] = []
        self.intercepts: List[float] = []
        self.boundary_latencies: dict = {}  # boundary -> latency
        # Cached tensors for vectorized predict_batch
        self._latency_cache: torch.Tensor = None
        self._cache_device: torch.device = None
        self._cache_size: int = 0

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

        # Detect boundaries
        self.boundaries = [int(tokens[0])]
        for i in range(1, len(tokens)):
            if lats[i - 1] > 0 and (lats[i] - lats[i - 1]) / lats[i - 1] > jump_threshold:
                self.boundaries.append(int(tokens[i]))
        self.boundaries.append(int(tokens[-1]) + 1)

        # Store boundary latencies for fast lookup
        for b in self.boundaries[:-1]:
            idx = np.where(tokens == b)[0]
            if len(idx) > 0:
                self.boundary_latencies[b] = float(lats[idx[0]])
            elif b > 0:
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
        """Predict latency for n tokens."""
        # Check boundary latencies first (fast path)
        if n in self.boundary_latencies:
            return self.boundary_latencies[n]

        # Fall back to piecewise linear
        for i in range(len(self.boundaries) - 1):
            if self.boundaries[i] <= n < self.boundaries[i + 1]:
                return self.slopes[i] * n + self.intercepts[i]

        # Extrapolate from last segment
        if self.slopes:
            return self.slopes[-1] * n + self.intercepts[-1]
        return 1.0  # fallback

    def predict_batch(self, max_tokens: int, device: torch.device) -> torch.Tensor:
        """
        Get latencies for token counts 1..max_tokens as a tensor (cached).

        Returns:
            latencies: [max_tokens] tensor where latencies[i] = latency for (i+1) tokens
        """
        # Return cached if available
        if (self._latency_cache is not None and
            self._cache_device == device and
            self._cache_size >= max_tokens):
            return self._latency_cache[:max_tokens]

        # Build latency lookup table
        latencies = torch.empty(max_tokens, dtype=torch.float32, device=device)
        for i in range(1, max_tokens + 1):
            latencies[i - 1] = self.predict(i)

        # Cache for reuse
        self._latency_cache = latencies
        self._cache_device = device
        self._cache_size = max_tokens

        return latencies

    def save(self, path: str):
        np.savez(
            path,
            boundaries=self.boundaries,
            slopes=self.slopes,
            intercepts=self.intercepts,
            boundary_latencies_keys=list(self.boundary_latencies.keys()),
            boundary_latencies_values=list(self.boundary_latencies.values()),
        )

    def load(self, path: str):
        data = np.load(path)
        self.boundaries = data["boundaries"].tolist()
        self.slopes = data["slopes"].tolist()
        self.intercepts = data["intercepts"].tolist()

        # Load boundary latencies
        bl_keys = data.get("boundary_latencies_keys", [])
        bl_values = data.get("boundary_latencies_values", [])
        if len(bl_keys) > 0:
            self.boundary_latencies = {int(k): float(v) for k, v in zip(bl_keys, bl_values)}


def find_optimal_cutoff(
    scores: torch.Tensor,
    calibration: Calibration,
    latency_model: PiecewiseLinearLatency,
    prefill_tokens: int = 0,
) -> Tuple[int, torch.Tensor]:
    """
    Find optimal draft token cutoff maximizing E[accepted] / Latency.

    Fully vectorized implementation:
    1. Calibrate cumulative scores to acceptance probabilities
    2. Sort globally by probability
    3. Compute E/L ratio for all values in parallel
    4. Find argmax and compute per-request allocation

    Args:
        scores: [bs, n_candidates] cumulative draft scores
        calibration: maps cumulative scores to acceptance probability
        latency_model: piecewise linear latency predictor
        prefill_tokens: tokens from prefill (for mixed batches)

    Returns:
        total_draft_tokens: optimal total draft tokens for the batch
        per_request_draft_tokens: [bs] number of draft tokens per request
    """
    bs, n_cand = scores.shape
    device = scores.device
    max_tokens = bs * n_cand

    # 1. Calibrate cumulative scores to acceptance probabilities (vectorized)
    probs = calibration.predict(scores)  # [bs, n_cand]

    # 2. Flatten and sort globally by probability (vectorized)
    flat_probs = probs.flatten()  # [bs * n_cand]
    sorted_probs, sorted_indices = torch.sort(flat_probs, descending=True)

    # 3. Compute cumulative expected value (vectorized)
    cum_E = torch.cumsum(sorted_probs, dim=0)  # [max_tokens]

    # 4. Get latencies for all token counts (cached tensor, single call)
    if prefill_tokens > 0:
        latencies = latency_model.predict_batch(prefill_tokens + max_tokens, device)
        latencies = latencies[prefill_tokens:prefill_tokens + max_tokens]
    else:
        latencies = latency_model.predict_batch(max_tokens, device)

    # 5. Compute E/L ratio for all counts (vectorized) - add bs bonus tokens
    ratios = (cum_E + bs) / latencies  # [max_tokens]

    # 6. Find best count (vectorized argmax)
    best_idx = ratios.argmax().item()
    total_draft_tokens = best_idx + 1  # 1-indexed
    total_draft_tokens = max(bs, total_draft_tokens)  # at least 1 per request

    # 7. Compute per-request allocation (vectorized with bincount)
    selected_indices = sorted_indices[:total_draft_tokens]
    request_ids = selected_indices // n_cand  # which request each token belongs to
    per_request_draft_tokens = torch.bincount(request_ids, minlength=bs)

    return total_draft_tokens, per_request_draft_tokens
