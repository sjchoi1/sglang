"""
Core tile-aware speculation algorithms.

- Calibration: maps cumulative draft scores to acceptance probabilities (linear regression)
- PiecewiseLinearLatency: models verification latency with tile boundaries
"""

from typing import List

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
        assert len(scores) == len(accepted), f"Mismatched lengths: {len(scores)} vs {len(accepted)}"
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
        assert len(token_counts) == len(latencies), f"Mismatched lengths: {len(token_counts)} vs {len(latencies)}"
        if len(token_counts) == 0:
            return

        # Sort by token count
        sorted_pairs = sorted(zip(token_counts, latencies))
        tokens = np.array([p[0] for p in sorted_pairs])
        lats = np.array([p[1] for p in sorted_pairs])

        # Detect boundaries (where latency jumps)
        self.boundaries = [int(tokens[0])]
        for i in range(1, len(tokens)):
            if lats[i - 1] > 0 and (lats[i] - lats[i - 1]) / lats[i - 1] > jump_threshold:
                self.boundaries.append(int(tokens[i]))
        self.boundaries.append(int(tokens[-1]) + 1)

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
        """Predict latency for n tokens using piecewise linear model."""
        # Find segment and use linear regression
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
        )

    def load(self, path: str):
        data = np.load(path)
        self.boundaries = data["boundaries"].tolist()
        self.slopes = data["slopes"].tolist()
        self.intercepts = data["intercepts"].tolist()
