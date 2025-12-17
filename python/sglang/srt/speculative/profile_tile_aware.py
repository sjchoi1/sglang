"""
Profiling script for tile-aware dynamic speculation.

Usage:
    python -m sglang.srt.speculative.profile_tile_aware \
        --model-path meta-llama/Llama-3.1-8B \
        --draft-model-path meta-llama/Llama-3.1-8B-Instruct \
        --output-dir ./tile_aware_profiles

This script:
1. Profiles verification latency at different token counts
2. Fits a piecewise linear latency model
3. Collects (score, accepted) pairs for calibration
4. Fits the calibration model
"""

import argparse
import time
from typing import List, Tuple

import numpy as np
import torch

from sglang.srt.speculative.tile_aware import Calibration, PiecewiseLinearLatency


def profile_verification_latency(
    run_verify_fn,
    batch_size: int,
    token_counts: List[int],
    num_warmup: int = 5,
    num_runs: int = 20,
) -> List[Tuple[int, float]]:
    """
    Profile verification latency by running actual verification.

    Args:
        run_verify_fn: Function that runs verification with (batch_size, n_tokens)
                       and returns latency in ms
        batch_size: Batch size to profile
        token_counts: List of token counts to profile
        num_warmup: Warmup iterations
        num_runs: Measurement iterations

    Returns:
        List of (token_count, latency_ms) tuples
    """
    measurements = []

    for n_tokens in token_counts:
        # Warmup
        for _ in range(num_warmup):
            run_verify_fn(batch_size, n_tokens)
        torch.cuda.synchronize()

        # Measure
        times = []
        for _ in range(num_runs):
            torch.cuda.synchronize()
            start = time.perf_counter()
            run_verify_fn(batch_size, n_tokens)
            torch.cuda.synchronize()
            times.append(time.perf_counter() - start)

        latency_ms = np.median(times) * 1000
        measurements.append((n_tokens, latency_ms))
        print(f"  n={n_tokens:4d}: {latency_ms:.3f} ms")

    return measurements


def collect_calibration_data(
    run_draft_verify_fn,
    num_samples: int = 1000,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Collect (score, accepted) pairs for calibration.

    Args:
        run_draft_verify_fn: Function that runs draft+verify and returns
                             (scores, accepted_mask) tensors
        num_samples: Number of samples to collect

    Returns:
        scores: [N] array of cumulative draft scores
        accepted: [N] array of 0/1 acceptance indicators
    """
    all_scores = []
    all_accepted = []

    for i in range(num_samples):
        scores, accepted = run_draft_verify_fn()
        all_scores.append(scores.cpu().numpy().flatten())
        all_accepted.append(accepted.cpu().numpy().flatten())

        if (i + 1) % 100 == 0:
            print(f"  Collected {i + 1}/{num_samples} samples")

    return np.concatenate(all_scores), np.concatenate(all_accepted)


def main():
    parser = argparse.ArgumentParser(description="Profile tile-aware speculation")
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument("--min-tokens", type=int, default=16)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--token-step", type=int, default=16)
    parser.add_argument("--num-calibration-samples", type=int, default=500)
    args = parser.parse_args()

    import os
    os.makedirs(args.output_dir, exist_ok=True)

    # Token counts to profile
    token_counts = list(range(args.min_tokens, args.max_tokens + 1, args.token_step))

    print("=" * 60)
    print("Tile-Aware Speculation Profiling")
    print("=" * 60)

    # TODO: Initialize model and create run_verify_fn
    # This is a placeholder - actual implementation needs model setup

    print("\n[1] Profiling verification latency...")
    print("-" * 40)

    # Placeholder: Replace with actual verification function
    def dummy_run_verify(batch_size, n_tokens):
        # Simulate verification
        x = torch.randn(batch_size * n_tokens, 4096, device="cuda")
        y = torch.nn.functional.linear(x, torch.randn(4096, 4096, device="cuda"))
        return y

    all_measurements = []
    for bs in args.batch_sizes:
        print(f"\nBatch size = {bs}")
        measurements = profile_verification_latency(
            dummy_run_verify, bs, token_counts
        )
        for n, lat in measurements:
            all_measurements.append((n, lat))

    # Fit latency model
    print("\n[2] Fitting piecewise linear latency model...")
    print("-" * 40)

    latency_model = PiecewiseLinearLatency()
    tokens, lats = zip(*all_measurements)
    latency_model.fit(list(tokens), list(lats))

    print(f"Detected boundaries: {latency_model.get_boundaries()}")
    for i, (slope, intercept) in enumerate(zip(latency_model.slopes, latency_model.intercepts)):
        lo, hi = latency_model.boundaries[i], latency_model.boundaries[i + 1]
        print(f"  [{lo:3d}, {hi:3d}): slope={slope:.4f}, intercept={intercept:.2f}")

    latency_path = os.path.join(args.output_dir, "latency_model.npz")
    latency_model.save(latency_path)
    print(f"Saved latency model to {latency_path}")

    print("\n[3] Collecting calibration data...")
    print("-" * 40)

    # Placeholder: Replace with actual draft+verify function
    def dummy_draft_verify():
        scores = torch.rand(64, device="cuda")
        accepted = (torch.rand(64, device="cuda") < scores).float()
        return scores, accepted

    scores, accepted = collect_calibration_data(
        dummy_draft_verify, args.num_calibration_samples
    )

    print(f"Collected {len(scores)} (score, accepted) pairs")
    print(f"Overall acceptance rate: {accepted.mean():.3f}")

    # Fit calibration model
    print("\n[4] Fitting calibration model...")
    print("-" * 40)

    calibration = Calibration(num_bins=50)
    calibration.fit(scores, accepted)

    calib_path = os.path.join(args.output_dir, "calibration.npz")
    calibration.save(calib_path)
    print(f"Saved calibration model to {calib_path}")

    # Print calibration summary
    print("\nCalibration bins (score range -> acceptance prob):")
    for i in range(0, calibration.num_bins, 10):
        lo = calibration.bin_edges[i]
        hi = calibration.bin_edges[i + 1]
        prob = calibration.bin_probs[i]
        print(f"  [{lo:.2f}, {hi:.2f}): {prob:.3f}")

    print("\n" + "=" * 60)
    print("Profiling complete!")
    print(f"Output files in: {args.output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
