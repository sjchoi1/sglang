"""
TileSpec: Tile-aware dynamic speculation optimization.

Optimizes draft token count based on GPU tile boundaries and acceptance
probability to maximize throughput. Includes automatic profiling, calibration,
and latency modeling.

Classes:
    Calibration: Maps cumulative draft scores to acceptance probabilities
    PiecewiseLinearLatency: Models verification latency with tile boundaries
    TileSpecProfiler: Automatic profiling during warmup with caching
"""

import csv
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

logger = logging.getLogger(__name__)

# Try to import tqdm, fallback to no-op if not available
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable

SHAREGPT_URL = "https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json"

# Comprehensive batch size sweep for profiling - covers full range 1-64
WARMUP_BATCH_SIZES = list(range(1, 65))

# Store original log levels for restoration
_original_log_levels = {}


# ==============================================================================
# Core Algorithms
# ==============================================================================

class Calibration:
    """
    Maps cumulative draft scores to acceptance probability using linear regression.

    Note: EAGLE/EAGLE3 don't need calibration (score ≈ P(accept) directly).
    This class is kept for Draft LM and other methods that may need calibration.
    """

    def __init__(self):
        # Linear regression: P(accept) = slope * score + intercept
        self.slope = 1.0  # default: identity mapping
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
        # Precomputed tensors (device-specific caches)
        self._boundaries_tensors: dict = {}
        self._slopes_tensors: dict = {}
        self._intercepts_tensors: dict = {}

    def fit(
        self,
        token_counts: List[int],
        latencies: List[float],
        jump_threshold: float = 0.1,
    ):
        """
        Fit piecewise linear model from measurements.

        Args:
            token_counts: list of token counts profiled
            latencies: corresponding latencies (ms)
            jump_threshold: relative jump to detect boundary (0.05 = 5%)
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

    def _get_tensors_for_device(self, device: torch.device):
        """Get or create precomputed tensors for the given device."""
        device_key = str(device)
        if device_key not in self._boundaries_tensors:
            # Precompute tensors for this device
            self._boundaries_tensors[device_key] = torch.tensor(
                self.boundaries[1:], dtype=torch.float32, device=device
            )
            self._slopes_tensors[device_key] = torch.tensor(
                self.slopes, dtype=torch.float32, device=device
            )
            self._intercepts_tensors[device_key] = torch.tensor(
                self.intercepts, dtype=torch.float32, device=device
            )
        return (
            self._boundaries_tensors[device_key],
            self._slopes_tensors[device_key],
            self._intercepts_tensors[device_key],
        )

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

        # Edge case: no boundaries fitted
        if not self.boundaries or not self.slopes:
            latencies = torch.ones(max_tokens, dtype=torch.float32, device=device)
            self._latency_cache = latencies
            self._cache_device = device
            self._cache_size = max_tokens
            return latencies

        # Get precomputed tensors for this device
        boundaries_tensor, slopes_tensor, intercepts_tensor = self._get_tensors_for_device(device)

        # Vectorized latency computation
        token_range = torch.arange(1, max_tokens + 1, dtype=torch.float32, device=device)

        # Find segment index for each token using searchsorted
        segment_indices = torch.searchsorted(boundaries_tensor, token_range, right=False)
        segment_indices = segment_indices.clamp(0, len(self.slopes) - 1)

        # Vectorized linear interpolation: latency = slope * tokens + intercept
        latencies = slopes_tensor[segment_indices] * token_range + intercepts_tensor[segment_indices]

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
        # Clear tensor caches (will be recreated on demand)
        self._boundaries_tensors.clear()
        self._slopes_tensors.clear()
        self._intercepts_tensors.clear()
        self._latency_cache = None
        self._cache_device = None
        self._cache_size = 0


# ==============================================================================
# Profiler
# ==============================================================================

class TileSpecProfiler:
    """Profiles tile-spec latency using actual sglang runs."""

    def __init__(self, server_args):
        self.server_args = server_args
        self.cache_dir = get_cache_dir(
            server_args.model_path,
            torch.cuda.get_device_name(0),
            server_args.tp_size,
        )
        self._profiling = False
        self._latency_data: List[Tuple[int, float]] = []
        self._calibration_data: List[Tuple[float, bool]] = []

        self.latency_model: Optional[PiecewiseLinearLatency] = None
        self._load_cache()

    def _load_cache(self) -> bool:
        """Load latency model from cache."""
        latency_path = self.cache_dir / "latency_model.npz"
        if not latency_path.exists():
            return False
        try:
            self.latency_model = PiecewiseLinearLatency()
            self.latency_model.load(str(latency_path))
            logger.info(f"TileSpec: Loaded from cache, boundaries={self.latency_model.boundaries}")
            return True
        except Exception as e:
            logger.warning(f"Cache load failed: {e}")
            self.latency_model = None
            return False

    def needs_profiling(self) -> bool:
        return self.latency_model is None

    def is_profiling(self) -> bool:
        return self._profiling

    def get_latency_model(self) -> Optional[PiecewiseLinearLatency]:
        """Get the latency model (calibration not needed - scores are used directly)."""
        return self.latency_model

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

    def finish_profiling(self):
        """Fit latency model from collected data."""
        if not self._profiling:
            return
        self._profiling = False

        # Aggregate by token count with outlier removal
        by_tokens = defaultdict(list)
        for n, lat in self._latency_data:
            by_tokens[n].append(lat)

        # Remove outliers using IQR method per token count
        by_tokens_clean = {}
        outliers = {}
        for token_count, lats in by_tokens.items():
            if len(lats) >= 3:
                lats_arr = np.array(lats)
                q1, q3 = np.percentile(lats_arr, [25, 75])
                iqr = q3 - q1
                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr
                mask = (lats_arr >= lower_bound) & (lats_arr <= upper_bound)
                by_tokens_clean[token_count] = lats_arr[mask].tolist()
                outliers[token_count] = lats_arr[~mask].tolist()
            else:
                by_tokens_clean[token_count] = lats
                outliers[token_count] = []

        token_counts = sorted(by_tokens_clean.keys())
        latencies = [np.median(by_tokens_clean[t]) for t in token_counts]

        total_outliers = sum(len(o) for o in outliers.values())
        logger.info(f"TileSpec: Fitting from {len(self._latency_data)} samples ({total_outliers} outliers removed)")
        logger.info(f"  Token counts: min={min(token_counts)}, max={max(token_counts)}, unique={len(token_counts)}")
        logger.info(f"  Latency range: {min(latencies):.2f}ms - {max(latencies):.2f}ms")

        self.latency_model = PiecewiseLinearLatency()
        self.latency_model.fit(token_counts, latencies)

        # Note: Calibration is not needed - scores directly represent P(accept)
        # We still collect data for visualization to prove this relationship
        if self._calibration_data:
            logger.info(f"  Calibration data: {len(self._calibration_data)} samples (used for visualization only)")

        # Save cache
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.latency_model.save(str(self.cache_dir / "latency_model.npz"))

        # Save raw data to CSV files
        self._save_csv_data(by_tokens_clean, outliers)

        # Generate and save visualizations
        self._save_visualizations(token_counts, latencies, by_tokens_clean, outliers)

        logger.info(f"TileSpec: Complete, boundaries={self.latency_model.boundaries}")
        self._latency_data.clear()
        self._calibration_data.clear()

    def _save_csv_data(self, by_tokens_clean, outliers):
        """Save raw profiling data to CSV files."""
        csv_dir = self.cache_dir / "csv"
        csv_dir.mkdir(parents=True, exist_ok=True)

        # 1. Save latency data (clean samples)
        latency_csv = csv_dir / "latency_data.csv"
        with open(latency_csv, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['token_count', 'latency_ms'])
            for token_count in sorted(by_tokens_clean.keys()):
                for latency in by_tokens_clean[token_count]:
                    writer.writerow([token_count, latency])

        # 2. Save outliers if any exist
        total_outliers = sum(len(o) for o in outliers.values())
        if total_outliers > 0:
            outliers_csv = csv_dir / "latency_outliers.csv"
            with open(outliers_csv, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['token_count', 'latency_ms'])
                for token_count in sorted(outliers.keys()):
                    for latency in outliers[token_count]:
                        writer.writerow([token_count, latency])

        # 3. Save calibration data
        if self._calibration_data:
            cal_csv = csv_dir / "calibration_data.csv"
            with open(cal_csv, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['cumulative_score', 'accepted'])
                for score, accepted in self._calibration_data:
                    writer.writerow([score, int(accepted)])
            logger.info(f"TileSpec: Saved {len(self._calibration_data)} calibration samples to CSV")
        else:
            logger.warning("TileSpec: No calibration data collected!")

        logger.info(f"TileSpec: Saved CSV data to {csv_dir}")

    def _save_visualizations(self, token_counts, latencies, by_tokens_clean, outliers):
        """Generate and save profiling visualizations as PNG."""
        try:
            import matplotlib
            matplotlib.use('Agg')  # Non-interactive backend
            import matplotlib.pyplot as plt
        except ImportError:
            logger.warning("matplotlib not available, skipping visualizations")
            return

        plots_dir = self.cache_dir / "plots"
        plots_dir.mkdir(parents=True, exist_ok=True)

        # 1. Latency Model Plot
        self._plot_latency_model(token_counts, latencies, by_tokens_clean, outliers, plots_dir)

        # 2. Score vs Acceptance Plot (proves score ≈ P(accept), no calibration needed)
        self._plot_calibration(plots_dir)

        logger.info(f"TileSpec: Saved visualizations to {plots_dir}")

    def _plot_latency_model(self, token_counts, latencies, by_tokens_clean, outliers, plots_dir):
        """Plot latency model with raw samples and fitted model."""
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(10, 6))

        # Plot clean samples (scatter)
        all_tokens = []
        all_lats = []
        for n, lat in by_tokens_clean.items():
            all_tokens.extend([n] * len(lat))
            all_lats.extend(lat)
        ax.scatter(all_tokens, all_lats, alpha=0.3, s=20, label='Raw samples')

        # Plot fitted piecewise model
        if self.latency_model:
            max_tokens = max(token_counts)
            fit_x = np.arange(1, max_tokens + 1)
            fit_y = [self.latency_model.predict(n) for n in fit_x]
            ax.plot(fit_x, fit_y, 'r-', linewidth=2, label='Fitted model')

            # Mark tile boundaries
            for boundary in self.latency_model.boundaries:
                if boundary <= max_tokens:
                    ax.axvline(boundary, color='red', linestyle=':', alpha=0.5, linewidth=1)

        ax.set_xlabel('Total Tokens', fontsize=12)
        ax.set_ylabel('Latency (ms)', fontsize=12)
        ax.set_title('TileSpec Latency Model', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(plots_dir / 'latency_model.png', dpi=150)
        plt.close()

    def _plot_calibration(self, plots_dir):
        """Plot score vs P(accept) with identity line and distribution."""
        import matplotlib.pyplot as plt

        fig, ax1 = plt.subplots(figsize=(10, 6))

        # Bin scores and compute acceptance rate
        cal_scores = np.array([s for s, _ in self._calibration_data])
        cal_accepted = np.array([a for _, a in self._calibration_data])

        # Create 20 bins of 0.05 width each (0 to 1)
        n_bins = 20
        bin_width = 0.05
        bins = np.linspace(0, 1, n_bins + 1)
        bin_indices = np.digitize(cal_scores, bins)

        bin_centers = []
        acceptance_rates = []
        bin_counts = []
        for i in range(1, len(bins)):
            mask = bin_indices == i
            bin_centers.append((bins[i-1] + bins[i]) / 2)
            bin_counts.append(mask.sum())
            if mask.sum() > 0:
                acceptance_rates.append(cal_accepted[mask].mean())
            else:
                acceptance_rates.append(0)

        # Plot empirical acceptance rate as bar graph
        ax1.bar(bin_centers, acceptance_rates, width=bin_width * 0.8, alpha=0.7,
               color='steelblue', edgecolor='black', linewidth=0.5, label='P(Accept)')

        # Plot identity line (0,0) to (1,1) - proves score ≈ P(accept)
        ax1.plot([0, 1], [0, 1], 'r-', linewidth=2, label='Identity (y=x)')

        ax1.set_xlabel('Cumulative Draft Score', fontsize=12)
        ax1.set_ylabel('P(Accept)', fontsize=12, color='steelblue')
        ax1.set_xlim(-0.05, 1.05)
        ax1.set_ylim(0, 1.05)
        ax1.tick_params(axis='y', labelcolor='steelblue')

        # Secondary y-axis for distribution (normalized to sum=1)
        ax2 = ax1.twinx()
        total_samples = sum(bin_counts)
        bin_distribution = [c / total_samples if total_samples > 0 else 0 for c in bin_counts]
        ax2.bar(bin_centers, bin_distribution, width=bin_width * 0.4, alpha=0.5,
               color='green', edgecolor='darkgreen', linewidth=0.5, label='Distribution')
        ax2.set_ylabel('Distribution', fontsize=12, color='green')
        ax2.tick_params(axis='y', labelcolor='green')

        # Combined legend
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

        ax1.set_title('Draft Confidence vs Acceptance Probability', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        plt.savefig(plots_dir / 'score_acceptance.png', dpi=150)
        plt.close()

# ==============================================================================
# Helper Functions
# ==============================================================================

def _suppress_profiling_logs():
    """Temporarily suppress verbose logs during profiling."""
    global _original_log_levels

    # Create flag file to signal scheduler subprocess to suppress logs
    flag_file = Path("/tmp/.sglang_tile_spec_profiling")
    flag_file.touch()

    # Suppress uvicorn access logs (HTTP 200 OK messages)
    uvicorn_logger = logging.getLogger("uvicorn.access")
    _original_log_levels["uvicorn"] = uvicorn_logger.level
    uvicorn_logger.setLevel(logging.WARNING)


def _restore_profiling_logs():
    """Restore original logging levels after profiling."""
    global _original_log_levels

    # Remove flag file
    flag_file = Path("/tmp/.sglang_tile_spec_profiling")
    flag_file.unlink(missing_ok=True)

    if "uvicorn" in _original_log_levels:
        logging.getLogger("uvicorn.access").setLevel(_original_log_levels["uvicorn"])
    _original_log_levels.clear()


def _get_tile_spec_cache_root() -> Path:
    """Get cache root at project root level."""
    # Start from current file and go up to find project root
    current = Path(__file__).resolve()
    # Go up from: .../sglang/python/sglang/srt/speculative/tilespec.py
    # To: .../sglang/
    project_root = current.parent.parent.parent.parent.parent
    return project_root / "tile_spec" / "cache"


TILE_SPEC_CACHE_ROOT = _get_tile_spec_cache_root()


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
    finish_fn: Callable[[], None],
):
    """
    Run TileSpec warmup profiling through actual verify() path.

    Draft token variation is handled internally via random truncation in organize_draft_results().

    Args:
        server_args: Server args with model_path and tp_size
        generate_fn: Function to send generation requests, signature: (prompts: List[str]) -> None
        check_ready_fn: Function that returns True when profiling is complete
        finish_fn: Function to signal profiling should finish
    """
    logger.info(f"TileSpec: warmup called, tile_spec={getattr(server_args, 'tile_spec', False)}")
    if not getattr(server_args, 'tile_spec', False):
        logger.info("TileSpec: Skipping warmup (not enabled)")
        return

    # Check if cache exists (direct filesystem check to avoid race conditions)
    cache_dir = get_cache_dir(
        server_args.model_path,
        torch.cuda.get_device_name(0),
        server_args.tp_size,
    )
    latency_cache = cache_dir / "latency_model.npz"
    if latency_cache.exists():
        logger.info(f"TileSpec: Cache exists at {latency_cache}, skipping warmup")
        return

    logger.info("TileSpec: Running warmup profiling...")
    logger.info(f"  Batch sizes: 1 to {len(WARMUP_BATCH_SIZES)} (full sweep)")
    logger.info(f"  Draft tokens: random truncation per request (handled internally)")
    logger.info(f"  Total configurations: {len(WARMUP_BATCH_SIZES)}")

    # Load prompts from shared cache
    prompts = _load_prompts(limit=200)

    # Suppress verbose logs during profiling
    _suppress_profiling_logs()

    try:
        # Run warmup with varying batch sizes
        # Draft token variation is handled internally via random truncation
        idx = 0
        total_runs = 0

        pbar = tqdm(total=len(WARMUP_BATCH_SIZES), desc="TileSpec Profiling", unit="run", ncols=80)

        for batch_size in WARMUP_BATCH_SIZES:
            batch = prompts[idx:idx + batch_size]
            idx = (idx + batch_size) % len(prompts)  # Wrap around if needed
            if not batch:
                break
            try:
                generate_fn(batch)
                total_runs += 1
                pbar.update(1)
            except Exception as e:
                logger.warning(f"TileSpec: Warmup batch failed: {e}")
                pbar.update(1)

        pbar.close()
    finally:
        # Restore logging levels even if profiling fails
        _restore_profiling_logs()

    logger.info(f"TileSpec: Completed {total_runs} warmup runs")

    # Signal profiling to finish
    logger.info("TileSpec: Signaling profiling to finish...")
    try:
        finish_fn()
    except Exception as e:
        logger.warning(f"TileSpec: finish_fn() failed: {e}")

    # Wait for profiling to complete via polling
    logger.info("TileSpec: Waiting for profiling completion...")
    while True:
        try:
            if check_ready_fn():
                logger.info("TileSpec: Profiling complete")
                return
        except Exception as e:
            logger.warning(f"TileSpec: check_ready_fn() failed: {e}")
        time.sleep(1.0)
