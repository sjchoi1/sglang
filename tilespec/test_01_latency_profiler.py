"""
Test 1: Latency Profiler
========================
Measures MLP forward pass latency at different token counts to detect tile boundaries.

Usage:
    python tilespec/test_01_latency_profiler.py

Expected output:
    - Latency measurements for various token counts
    - Detected tile boundaries (where latency jumps >15%)
    - Saved latency data to tilespec/latency_raw.npz
"""

import os
import sys
import time
import numpy as np

# Add parent to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch


def measure_mlp_latency(
    total_tokens: int,
    hidden_size: int = 4096,
    intermediate_size: int = 14336,
    num_warmup: int = 5,
    num_runs: int = 20,
) -> float:
    """
    Measure MLP forward pass latency for given token count.

    Uses Llama-style MLP dimensions by default (8B model).
    """
    device = "cuda"

    # Create tensors
    x = torch.randn(total_tokens, hidden_size, device=device, dtype=torch.float16)
    w_gate = torch.randn(intermediate_size, hidden_size, device=device, dtype=torch.float16)
    w_up = torch.randn(intermediate_size, hidden_size, device=device, dtype=torch.float16)
    w_down = torch.randn(hidden_size, intermediate_size, device=device, dtype=torch.float16)

    def forward():
        # Llama-style SwiGLU MLP
        gate = torch.nn.functional.linear(x, w_gate)
        up = torch.nn.functional.linear(x, w_up)
        hidden = torch.nn.functional.silu(gate) * up
        out = torch.nn.functional.linear(hidden, w_down)
        return out

    # Warmup
    for _ in range(num_warmup):
        forward()
    torch.cuda.synchronize()

    # Measure
    times = []
    for _ in range(num_runs):
        torch.cuda.synchronize()
        start = time.perf_counter()
        forward()
        torch.cuda.synchronize()
        times.append(time.perf_counter() - start)

    return np.median(times) * 1000  # ms


def main():
    print("=" * 60)
    print("TileSpec Test 1: Latency Profiler")
    print("=" * 60)

    # Check CUDA
    if not torch.cuda.is_available():
        print("ERROR: CUDA not available")
        return

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print()

    # Configuration
    min_tokens = 1
    max_tokens = 512
    step = 1  # Fine granularity to detect exact boundaries

    token_counts = list(range(min_tokens, max_tokens + 1, step))

    print(f"Profiling {len(token_counts)} token counts from {min_tokens} to {max_tokens}")
    print("-" * 60)

    # Profile
    results = []
    for n in token_counts:
        lat = measure_mlp_latency(n)
        results.append((n, lat))
        print(f"n={n:4d}: {lat:.4f} ms")

    # Detect boundaries
    print()
    print("-" * 60)
    print("Detected Tile Boundaries (>15% latency jump):")
    print("-" * 60)

    boundaries = [results[0][0]]  # Start with first token count
    for i in range(1, len(results)):
        prev_n, prev_lat = results[i-1]
        curr_n, curr_lat = results[i]
        if prev_lat > 0:
            jump = (curr_lat - prev_lat) / prev_lat
            if jump > 0.15:
                boundaries.append(curr_n)
                print(f"  {prev_n:4d} -> {curr_n:4d}: +{jump*100:.1f}% ({prev_lat:.4f} -> {curr_lat:.4f} ms)")

    boundaries.append(max_tokens + step)  # End boundary

    print()
    print(f"Boundaries: {boundaries}")

    # Save raw data
    output_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(output_dir, "latency_raw.npz")

    tokens_arr = np.array([r[0] for r in results])
    latencies_arr = np.array([r[1] for r in results])

    np.savez(output_path, tokens=tokens_arr, latencies=latencies_arr, boundaries=boundaries)
    print(f"Saved raw data to: {output_path}")

    print()
    print("=" * 60)
    print("Test 1 Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
