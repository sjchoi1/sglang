"""
Tile-Spec Profiler - Automatic profiling during warmup.

Profiles latency by recording actual verify() calls.
Cache structure: tile_spec/cache/{model}_{gpu}_tp{N}/
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

# Comprehensive batch size sweep for profiling - covers full range 1-128
# With K values 1-8, this explores all token counts from 1 to ~1024
WARMUP_BATCH_SIZES = list(range(1, 129))

# Cache location: Find project root to avoid path nesting issues
def _get_tile_spec_cache_root() -> Path:
    """Get cache root at project root level."""
    # Start from current file and go up to find project root
    current = Path(__file__).resolve()
    # Go up from: .../sglang/python/sglang/srt/speculative/tile_spec/profiler.py
    # To: .../sglang/
    project_root = current.parent.parent.parent.parent.parent.parent
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
    logger.info(f"TileSpec: warmup called, tile_spec={getattr(server_args, 'tile_spec', False)}")
    if not getattr(server_args, 'tile_spec', False):
        logger.info("TileSpec: Skipping warmup (not enabled)")
        return

    # Check if already profiled (cached)
    try:
        is_ready = check_ready_fn()
        logger.info(f"TileSpec: check_ready_fn() = {is_ready}")
    except Exception as e:
        logger.warning(f"TileSpec: check_ready_fn() failed: {e}")
        is_ready = False

    if is_ready:
        logger.info("TileSpec: Already profiled, skipping warmup")
        return

    logger.info("TileSpec: Running warmup profiling...")
    logger.info(f"  Batch sizes: 1 to {len(WARMUP_BATCH_SIZES)} (full sweep)")
    logger.info(f"  Running single iteration per batch size for comprehensive coverage")

    # Load prompts from shared cache
    prompts = _load_prompts(limit=200)

    # Run warmup with varying batch sizes (single iteration for each size)
    idx = 0
    total_runs = 0

    for batch_size in WARMUP_BATCH_SIZES:
        batch = prompts[idx:idx + batch_size]
        idx = (idx + batch_size) % len(prompts)  # Wrap around if needed
        if not batch:
            break
        try:
            generate_fn(batch)
            total_runs += 1
            if batch_size % 16 == 0 or batch_size <= 10:  # Log less frequently
                logger.info(f"TileSpec: Warmup bs={batch_size} done (total: {total_runs})")
        except Exception as e:
            logger.warning(f"TileSpec: Warmup batch failed: {e}")

    logger.info(f"TileSpec: Completed {total_runs} warmup runs")

    # Wait for profiling to complete (auto-finishes when enough samples)
    logger.info(f"TileSpec: Waiting for profiling to complete (max {max_wait}s)...")
    for i in range(max_wait):
        try:
            is_ready = check_ready_fn()
            if is_ready:
                logger.info(f"TileSpec: Profiling complete (checked after {i+1}s)")
                return
        except Exception as e:
            logger.warning(f"TileSpec: check_ready_fn() failed in loop: {e}")
        time.sleep(1.0)

    logger.warning("TileSpec: Profiling timeout, continuing anyway")


class TileSpecProfiler:
    """Profiles tile-spec latency using actual sglang runs."""

    def __init__(self, server_args, min_samples: int = 70):
        self.server_args = server_args
        self.cache_dir = get_cache_dir(
            server_args.model_path,
            torch.cuda.get_device_name(0),
            server_args.tp_size,
        )
        self.min_samples = min_samples
        self.min_unique_counts = 6  # Require at least 6 different token counts
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
            if unique >= self.min_unique_counts:
                logger.info(f"TileSpec: Auto-finishing profiling ({len(self._latency_data)} samples, {unique} unique token counts)")
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

        # Generate and save visualizations
        self._save_visualizations(token_counts, latencies, by_tokens)

        logger.info(f"TileSpec: Complete, boundaries={self.latency_model.boundaries}")
        self._latency_data.clear()
        self._calibration_data.clear()

    def _save_visualizations(self, token_counts, latencies, by_tokens):
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
        self._plot_latency_model(token_counts, latencies, by_tokens, plots_dir)

        # 2. Calibration Plot
        if self._calibration_data:
            self._plot_calibration(plots_dir)

        # 3. Token Distribution Plot
        self._plot_token_distribution(by_tokens, plots_dir)

        logger.info(f"TileSpec: Saved visualizations to {plots_dir}")

    def _plot_latency_model(self, token_counts, latencies, by_tokens, plots_dir):
        """Plot latency model with raw samples, fitted model, and boundaries."""
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(10, 6))

        # Plot all raw samples (scatter)
        all_tokens = []
        all_lats = []
        for n, lat in by_tokens.items():
            all_tokens.extend([n] * len(lat))
            all_lats.extend(lat)
        ax.scatter(all_tokens, all_lats, alpha=0.3, s=20, label='Raw samples')

        # Plot aggregated medians
        ax.plot(token_counts, latencies, 'o-', linewidth=2, markersize=8, label='Median latency')

        # Plot fitted piecewise model
        if self.latency_model:
            max_tokens = max(token_counts)
            fit_x = np.arange(1, max_tokens + 1)
            fit_y = [self.latency_model.predict(n) for n in fit_x]
            ax.plot(fit_x, fit_y, '--', linewidth=1.5, alpha=0.7, label='Fitted model')

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
        """Plot calibration curve: score vs P(accept)."""
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(10, 6))

        # Bin scores and compute acceptance rate
        cal_scores = np.array([s for s, _ in self._calibration_data])
        cal_accepted = np.array([a for _, a in self._calibration_data])

        # Create bins
        n_bins = 50
        bins = np.linspace(cal_scores.min(), cal_scores.max(), n_bins)
        bin_indices = np.digitize(cal_scores, bins)

        bin_centers = []
        acceptance_rates = []
        for i in range(1, len(bins)):
            mask = bin_indices == i
            if mask.sum() > 0:
                bin_centers.append((bins[i-1] + bins[i]) / 2)
                acceptance_rates.append(cal_accepted[mask].mean())

        # Plot empirical acceptance rate
        ax.scatter(bin_centers, acceptance_rates, alpha=0.6, s=30, label='Empirical')

        # Plot fitted calibration curve
        if self.calibration:
            score_range = np.linspace(cal_scores.min(), cal_scores.max(), 200)
            score_tensor = torch.tensor(score_range, dtype=torch.float32)
            fitted_probs = self.calibration.predict(score_tensor).cpu().numpy()
            ax.plot(score_range, fitted_probs, '-', linewidth=2, label='Fitted model')

        ax.set_xlabel('Cumulative Draft Score', fontsize=12)
        ax.set_ylabel('P(Accept)', fontsize=12)
        ax.set_title('TileSpec Calibration Curve', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(plots_dir / 'calibration.png', dpi=150)
        plt.close()

    def _plot_token_distribution(self, by_tokens, plots_dir):
        """Plot distribution of token counts profiled."""
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(10, 6))

        token_counts = sorted(by_tokens.keys())
        sample_counts = [len(by_tokens[t]) for t in token_counts]

        ax.bar(token_counts, sample_counts, alpha=0.7, edgecolor='black')
        ax.set_xlabel('Total Tokens', fontsize=12)
        ax.set_ylabel('Number of Samples', fontsize=12)
        ax.set_title('Profiling Coverage', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        plt.savefig(plots_dir / 'token_distribution.png', dpi=150)
        plt.close()
