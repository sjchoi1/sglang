"""
Tile-Spec Auto Profiler

Handles automatic latency profiling and calibration for tile-spec optimization.
Uses the exact SGLang forward path for profiling.

Cache structure: tile_spec/cache/{model}_{gpu}_{tp}/
"""

import json
import logging
import os
import re
import time
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch

from sglang.srt.speculative.tile_spec.core import Calibration, PiecewiseLinearLatency

logger = logging.getLogger(__name__)

# ShareGPT download URL
SHAREGPT_URL = "https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json"


def get_cache_dir(model_path: str, gpu_name: str, tp_size: int) -> Path:
    """Get cache directory path based on model, GPU, and TP configuration."""
    # Normalize model name (e.g., "meta-llama/Llama-3.1-8B-Instruct" -> "llama-3.1-8b-instruct")
    model_name = model_path.split("/")[-1].lower()
    model_name = re.sub(r"[^a-z0-9-]", "", model_name)

    # Normalize GPU name (e.g., "NVIDIA A100-SXM4-80GB" -> "a100")
    gpu_short = gpu_name.lower()
    for pattern in ["nvidia ", "geforce ", "tesla ", "-sxm4", "-pcie", "-80gb", "-40gb"]:
        gpu_short = gpu_short.replace(pattern, "")
    gpu_short = re.sub(r"[^a-z0-9]", "", gpu_short)

    cache_name = f"{model_name}_{gpu_short}_tp{tp_size}"

    # Cache in project root tile_spec/cache/
    current = Path(__file__).parent
    while current != current.parent:
        if (current / "tile_spec").exists() and (current / "python").exists():
            return current / "tile_spec" / "cache" / cache_name
        current = current.parent

    # Fallback to ~/.cache/sglang/tile_spec/
    return Path.home() / ".cache" / "sglang" / "tile_spec" / cache_name


def download_sharegpt(cache_dir: Path) -> Path:
    """Download ShareGPT dataset if not cached."""
    dataset_path = cache_dir / "sharegpt.json"
    if dataset_path.exists():
        return dataset_path

    import urllib.request

    logger.info(f"Downloading ShareGPT dataset to {dataset_path}...")
    cache_dir.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(SHAREGPT_URL, dataset_path)
    logger.info("Download complete.")
    return dataset_path


def load_sharegpt_prompts(dataset_path: Path, num_samples: int = 500) -> List[str]:
    """Load prompts from ShareGPT dataset."""
    with open(dataset_path) as f:
        dataset = json.load(f)

    # Filter conversations with at least 2 turns
    dataset = [
        data for data in dataset
        if len(data.get("conversations", data.get("conversation", []))) >= 2
    ]

    # Extract first turn (user prompt)
    prompts = [
        data.get("conversations", data.get("conversation", []))[0]["value"]
        for data in dataset[:num_samples]
    ]

    return prompts


def save_latency_plot(tokens: list, latencies: list, boundaries: list, output_path: Path):
    """Generate and save latency visualization."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not available, skipping visualization")
        return

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

    # Plot 1: Raw latency with boundaries
    ax1.plot(tokens, latencies, "b-", linewidth=1, alpha=0.7, label="Measured Latency")
    for b in boundaries[1:-1]:
        ax1.axvline(x=b, color="r", linestyle="--", alpha=0.7, linewidth=1.5)
        ax1.annotate(f"{b}", xy=(b, max(latencies) * 0.95), ha="center", fontsize=9, color="red")

    ax1.set_xlabel("Token Count", fontsize=12)
    ax1.set_ylabel("Latency (ms)", fontsize=12)
    ax1.set_title("Verification Latency vs Token Count (Tile Boundaries in Red)", fontsize=14)
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # Plot 2: Latency jump percentage
    jumps, jump_tokens = [], []
    for i in range(1, len(tokens)):
        if latencies[i - 1] > 0:
            jump = (latencies[i] - latencies[i - 1]) / latencies[i - 1] * 100
            jumps.append(jump)
            jump_tokens.append(tokens[i])

    colors = ["red" if j > 15 else "steelblue" for j in jumps]
    ax2.bar(jump_tokens, jumps, width=1, color=colors, alpha=0.7)
    ax2.axhline(y=15, color="r", linestyle="--", linewidth=1.5, label="15% threshold")
    ax2.axhline(y=0, color="black", linewidth=0.5)
    ax2.set_xlabel("Token Count", fontsize=12)
    ax2.set_ylabel("Latency Jump (%)", fontsize=12)
    ax2.set_title("Latency Jump Between Consecutive Token Counts", fontsize=14)
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    ax2.set_ylim(-10, max(50, max(jumps) * 1.1) if jumps else 50)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    logger.info(f"Saved latency plot to: {output_path}")


def save_calibration_plot(calibration: Calibration, output_path: Path):
    """Generate and save calibration visualization."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not available, skipping visualization")
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    bin_centers = (calibration.bin_edges[:-1] + calibration.bin_edges[1:]) / 2
    ax.bar(bin_centers, calibration.bin_probs, width=np.diff(calibration.bin_edges), alpha=0.7)
    ax.set_xlabel("Cumulative Score", fontsize=12)
    ax.set_ylabel("Acceptance Probability", fontsize=12)
    ax.set_title("Calibration: Score to Acceptance Probability", fontsize=14)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    logger.info(f"Saved calibration plot to: {output_path}")


class TileSpecProfiler:
    """
    Manages tile-spec profiling and caching.

    Usage:
        profiler = TileSpecProfiler(server_args)
        profiler.run_profiling(target_worker, draft_worker, tokenizer)
    """

    def __init__(self, server_args):
        from sglang.srt.server_args import ServerArgs
        self.server_args: ServerArgs = server_args

        if torch.cuda.is_available():
            self.gpu_name = torch.cuda.get_device_name(0)
        else:
            self.gpu_name = "unknown"

        self.cache_dir = get_cache_dir(
            server_args.model_path,
            self.gpu_name,
            server_args.tp_size
        )
        self.latency_path = self.cache_dir / "latency.npz"
        self.calibration_path = self.cache_dir / "calibration.npz"
        self.latency_plot_path = self.cache_dir / "latency.png"
        self.calibration_plot_path = self.cache_dir / "calibration.png"

        logger.info(f"Tile-spec cache dir: {self.cache_dir}")

    def has_latency_cache(self) -> bool:
        return self.latency_path.exists()

    def has_calibration_cache(self) -> bool:
        return self.calibration_path.exists()

    def profile_latency(
        self,
        target_worker,
        max_tokens: int = 512,
        num_warmup: int = 3,
        num_runs: int = 10,
    ) -> Tuple[List[int], List[float]]:
        """
        Profile verification latency using actual model forward pass.

        Uses the same ForwardBatch infrastructure as CUDA graph capture.
        """
        from sglang.srt.model_executor.forward_batch_info import (
            ForwardBatch,
            ForwardMode,
            CaptureHiddenMode,
        )

        logger.info("Profiling verification latency...")

        model_runner = target_worker.model_runner
        device = model_runner.device

        # Get model config
        vocab_size = model_runner.model_config.vocab_size

        token_counts = list(range(1, max_tokens + 1))
        latencies = []

        for num_tokens in token_counts:
            # Create dummy inputs similar to CUDA graph capture
            input_ids = torch.randint(0, vocab_size, (num_tokens,), device=device)
            req_pool_indices = torch.zeros(1, dtype=torch.int32, device=device)
            seq_lens = torch.tensor([num_tokens], dtype=torch.int32, device=device)
            out_cache_loc = torch.zeros(num_tokens, dtype=torch.int64, device=device)
            positions = torch.arange(num_tokens, dtype=torch.int64, device=device)
            extend_prefix_lens = torch.zeros(1, dtype=torch.int32, device=device)
            extend_seq_lens = torch.tensor([num_tokens], dtype=torch.int32, device=device)

            # Create ForwardBatch for decode (verification uses decode path)
            forward_batch = ForwardBatch(
                forward_mode=ForwardMode.DECODE,
                batch_size=1,
                input_ids=input_ids,
                req_pool_indices=req_pool_indices,
                seq_lens=seq_lens,
                out_cache_loc=out_cache_loc,
                seq_lens_sum=num_tokens,
                req_to_token_pool=model_runner.req_to_token_pool,
                token_to_kv_pool=model_runner.token_to_kv_pool,
                attn_backend=model_runner.attn_backend,
                return_logprob=False,
                capture_hidden_mode=CaptureHiddenMode.NULL,
                positions=positions,
                extend_prefix_lens=extend_prefix_lens,
                extend_seq_lens=extend_seq_lens,
            )

            # Warmup
            for _ in range(num_warmup):
                with torch.no_grad():
                    model_runner.forward(forward_batch)
            torch.cuda.synchronize()

            # Measure
            times = []
            for _ in range(num_runs):
                torch.cuda.synchronize()
                start = time.perf_counter()
                with torch.no_grad():
                    model_runner.forward(forward_batch)
                torch.cuda.synchronize()
                times.append(time.perf_counter() - start)

            latency_ms = np.median(times) * 1000
            latencies.append(latency_ms)

            if num_tokens % 64 == 0:
                logger.info(f"  n={num_tokens}: {latency_ms:.3f} ms")

        return token_counts, latencies

    def run_calibration(
        self,
        draft_worker,
        tokenizer,
        num_samples: int = 500,
    ) -> Calibration:
        """
        Run calibration using ShareGPT dataset.

        Collects (score, accepted) pairs by running speculative decoding
        on ShareGPT prompts.
        """
        logger.info("Running calibration with ShareGPT...")

        # Download ShareGPT if needed
        dataset_path = download_sharegpt(self.cache_dir)
        prompts = load_sharegpt_prompts(dataset_path, num_samples)

        scores_all = []
        accepted_all = []

        # TODO: Implement actual speculative decoding loop
        # For now, this is a placeholder that will be filled in
        # when we integrate with the scheduler

        # The actual implementation will:
        # 1. Tokenize prompts
        # 2. Run draft_forward to get scores
        # 3. Run verification to get acceptance
        # 4. Collect (score, accepted) pairs

        logger.warning("Calibration with ShareGPT not yet implemented - using default calibration")

        # Return default calibration
        calibration = Calibration()
        return calibration

    def run_profiling(
        self,
        target_worker,
        draft_worker=None,
        tokenizer=None,
    ) -> Tuple[PiecewiseLinearLatency, Calibration]:
        """
        Run full profiling: latency + calibration.

        Returns (latency_model, calibration).
        """
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Latency profiling
        if self.has_latency_cache():
            logger.info(f"Loading latency model from cache: {self.latency_path}")
            latency_model = PiecewiseLinearLatency()
            latency_model.load(str(self.latency_path))
        else:
            logger.info("Profiling latency...")
            tokens, latencies = self.profile_latency(target_worker)

            latency_model = PiecewiseLinearLatency()
            latency_model.fit(tokens, latencies, jump_threshold=0.15)

            latency_model.save(str(self.latency_path))
            logger.info(f"Saved latency model to: {self.latency_path}")
            logger.info(f"Detected boundaries: {latency_model.get_boundaries()}")
            logger.info(f"Optimal k candidates: {latency_model.get_optimal_k_candidates()}")

            save_latency_plot(tokens, latencies, latency_model.boundaries, self.latency_plot_path)

        # Calibration
        if self.has_calibration_cache():
            logger.info(f"Loading calibration from cache: {self.calibration_path}")
            calibration = Calibration()
            calibration.load(str(self.calibration_path))
        else:
            logger.info("Running calibration...")
            calibration = self.run_calibration(draft_worker, tokenizer)

            if calibration.bin_probs is not None:
                calibration.save(str(self.calibration_path))
                logger.info(f"Saved calibration to: {self.calibration_path}")
                save_calibration_plot(calibration, self.calibration_plot_path)

        return latency_model, calibration
