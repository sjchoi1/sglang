"""
Test 6: Throughput Benchmark (Simulated)
========================================
Simulates the throughput impact of tile-aware speculation
compared to fixed k baseline.

Usage:
    python tilespec/test_06_benchmark.py

This is a simulation-based benchmark. For real throughput testing,
run the actual SGLang server with and without --speculative-tile-aware.
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "python"))

import torch
from sglang.srt.speculative.tile_aware import (
    Calibration,
    PiecewiseLinearLatency,
    compute_optimal_k,
)


def simulate_speculative_step(
    scores: torch.Tensor,
    k: int,
    calibration: Calibration,
    latency_model: PiecewiseLinearLatency,
) -> dict:
    """
    Simulate one speculative decoding step.

    Returns:
        dict with expected_accepted, latency, throughput
    """
    bs = scores.shape[0]

    # Get acceptance probabilities
    probs = calibration.predict(scores[:, :k])
    flat_probs = probs.flatten()

    # Expected accepted tokens (probabilistic)
    expected_accepted = flat_probs.sum().item()

    # Add bonus tokens (1 per request - the verified token)
    total_tokens = expected_accepted + bs

    # Latency for verification
    latency = latency_model.predict(k)

    # Throughput = tokens / time
    throughput = total_tokens / latency if latency > 0 else 0

    return {
        "k": k,
        "expected_accepted": expected_accepted,
        "bonus_tokens": bs,
        "total_tokens": total_tokens,
        "latency_ms": latency,
        "throughput": throughput,
    }


def run_benchmark(
    calibration: Calibration,
    latency_model: PiecewiseLinearLatency,
    num_iterations: int = 100,
    batch_size: int = 4,
    max_k: int = 256,
):
    """Run simulated benchmark comparing fixed k vs tile-aware."""

    print(f"Running {num_iterations} iterations with batch_size={batch_size}")
    print("-" * 70)

    # Fixed k baselines
    fixed_k_values = [32, 64, 96, 128, 192, 256]

    results = {
        "tile_aware": {"throughputs": [], "k_values": []},
    }
    for k in fixed_k_values:
        if k <= max_k:
            results[f"fixed_{k}"] = {"throughputs": [], "k_values": []}

    for i in range(num_iterations):
        # Generate random scores for this iteration
        # Simulate varying quality - sometimes high, sometimes low
        quality = np.random.beta(2, 2)  # Random quality between 0 and 1
        scores = torch.rand(batch_size, max_k) * 0.4 + quality * 0.5

        # Tile-aware: compute optimal k
        tile_aware_k = compute_optimal_k(
            scores, calibration, latency_model, prefill_tokens=0, max_k=max_k
        )
        result = simulate_speculative_step(scores, tile_aware_k, calibration, latency_model)
        results["tile_aware"]["throughputs"].append(result["throughput"])
        results["tile_aware"]["k_values"].append(tile_aware_k)

        # Fixed k baselines
        for k in fixed_k_values:
            if k <= max_k:
                result = simulate_speculative_step(scores, k, calibration, latency_model)
                results[f"fixed_{k}"]["throughputs"].append(result["throughput"])
                results[f"fixed_{k}"]["k_values"].append(k)

    return results


def print_results(results: dict):
    """Print benchmark results."""

    print()
    print("=" * 70)
    print("Benchmark Results")
    print("=" * 70)
    print()

    print(f"{'Method':<20} {'Mean Throughput':>15} {'Std':>10} {'Improvement':>12}")
    print("-" * 70)

    # Get baseline (fixed_64 as reference)
    baseline_key = "fixed_64"
    if baseline_key not in results:
        baseline_key = list(results.keys())[1]  # First fixed k

    baseline_throughput = np.mean(results[baseline_key]["throughputs"])

    sorted_methods = sorted(
        results.items(),
        key=lambda x: np.mean(x[1]["throughputs"]),
        reverse=True
    )

    for method, data in sorted_methods:
        mean_tp = np.mean(data["throughputs"])
        std_tp = np.std(data["throughputs"])
        improvement = (mean_tp / baseline_throughput - 1) * 100

        if method == "tile_aware":
            method_name = "Tile-Aware (dynamic)"
        else:
            method_name = f"Fixed k={method.split('_')[1]}"

        marker = " ←" if method == "tile_aware" else ""
        print(f"{method_name:<20} {mean_tp:>15.2f} {std_tp:>10.2f} {improvement:>+11.1f}%{marker}")

    print()

    # Tile-aware specific analysis
    tile_k_values = results["tile_aware"]["k_values"]
    print("Tile-Aware k Selection Distribution:")
    print("-" * 70)

    unique_k, counts = np.unique(tile_k_values, return_counts=True)
    for k, count in sorted(zip(unique_k, counts), key=lambda x: -x[1]):
        pct = count / len(tile_k_values) * 100
        print(f"  k={k:3d}: {count:4d} times ({pct:5.1f}%)")

    print()


def main():
    print("=" * 70)
    print("TileSpec Test 6: Throughput Benchmark (Simulated)")
    print("=" * 70)
    print()

    # Load models
    output_dir = os.path.dirname(os.path.abspath(__file__))
    latency_path = os.path.join(output_dir, "latency_model.npz")
    calibration_path = os.path.join(output_dir, "calibration.npz")

    if not os.path.exists(latency_path):
        print(f"ERROR: {latency_path} not found. Run test_02 first.")
        return

    if not os.path.exists(calibration_path):
        print(f"ERROR: {calibration_path} not found. Run test_04 first.")
        return

    latency_model = PiecewiseLinearLatency()
    latency_model.load(latency_path)
    print(f"Loaded latency model: boundaries = {latency_model.get_boundaries()}")

    calibration = Calibration()
    calibration.load(calibration_path)
    print(f"Loaded calibration: {calibration.num_bins} bins")
    print()

    # Run benchmark
    results = run_benchmark(
        calibration,
        latency_model,
        num_iterations=500,
        batch_size=4,
        max_k=256,
    )

    print_results(results)

    print("=" * 70)
    print("Test 6 Complete!")
    print("=" * 70)
    print()
    print("Note: This is a SIMULATED benchmark based on the calibration model.")
    print("For real throughput measurements, run SGLang server with:")
    print()
    print("  # Baseline:")
    print("  python -m sglang.launch_server --model-path ... --speculative-algorithm EAGLE3")
    print()
    print("  # Tile-aware:")
    print("  python -m sglang.launch_server --model-path ... --speculative-algorithm EAGLE3 \\")
    print("      --speculative-tile-aware \\")
    print("      --speculative-calibration-path tilespec/calibration.npz \\")
    print("      --speculative-latency-path tilespec/latency_model.npz")


if __name__ == "__main__":
    main()
