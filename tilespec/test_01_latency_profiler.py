"""
Test 1: Latency Profiler with Visualization
============================================
Measures MLP forward pass latency at different token counts to detect tile boundaries,
then generates a visualization.

Usage:
    python tilespec/test_01_latency_profiler.py

Output:
    - tilespec/latency_raw.npz (raw data)
    - tilespec/latency_plot.png (visualization)
"""

import os
import sys
import time
import numpy as np

# Add parent to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


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


def visualize(tokens, latencies, boundaries, output_dir):
    """Generate visualization of latency data."""
    global HAS_MATPLOTLIB
    if not HAS_MATPLOTLIB:
        print("Installing matplotlib...")
        import subprocess
        subprocess.check_call(["pip", "install", "matplotlib"])
        HAS_MATPLOTLIB = True

    import matplotlib.pyplot as plt

    # Create figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

    # Plot 1: Raw latency
    ax1.plot(tokens, latencies, 'b-', linewidth=1, alpha=0.7, label='Measured Latency')

    # Mark tile boundaries
    for b in boundaries[1:-1]:
        ax1.axvline(x=b, color='r', linestyle='--', alpha=0.7, linewidth=1.5)
        ax1.annotate(f'{b}', xy=(b, max(latencies) * 0.95), ha='center', fontsize=9, color='red')

    ax1.set_xlabel('Token Count', fontsize=12)
    ax1.set_ylabel('Latency (ms)', fontsize=12)
    ax1.set_title('MLP Latency vs Token Count (Tile Boundaries in Red)', fontsize=14)
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # Plot 2: Latency jump percentage
    jumps = []
    jump_tokens = []
    for i in range(1, len(tokens)):
        if latencies[i-1] > 0:
            jump = (latencies[i] - latencies[i-1]) / latencies[i-1] * 100
            jumps.append(jump)
            jump_tokens.append(tokens[i])

    colors = ['red' if j > 15 else 'steelblue' for j in jumps]
    ax2.bar(jump_tokens, jumps, width=1, color=colors, alpha=0.7)
    ax2.axhline(y=15, color='r', linestyle='--', linewidth=1.5, label='15% threshold')
    ax2.axhline(y=0, color='black', linewidth=0.5)

    ax2.set_xlabel('Token Count', fontsize=12)
    ax2.set_ylabel('Latency Jump (%)', fontsize=12)
    ax2.set_title('Latency Jump Between Consecutive Token Counts (Red = >15%)', fontsize=14)
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    ax2.set_ylim(-10, max(jumps) * 1.1 if jumps else 50)

    plt.tight_layout()

    # Save
    output_path = os.path.join(output_dir, "latency_plot.png")
    plt.savefig(output_path, dpi=150)
    print(f"Saved plot to: {output_path}")
    plt.close()


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
    for i, n in enumerate(token_counts):
        lat = measure_mlp_latency(n)
        results.append((n, lat))
        if n % 50 == 0 or n <= 10:
            print(f"n={n:4d}: {lat:.4f} ms")

    tokens = np.array([r[0] for r in results])
    latencies = np.array([r[1] for r in results])

    # Detect boundaries
    print()
    print("-" * 60)
    print("Detected Tile Boundaries (>15% latency jump):")
    print("-" * 60)

    boundaries = [results[0][0]]
    for i in range(1, len(results)):
        prev_n, prev_lat = results[i-1]
        curr_n, curr_lat = results[i]
        if prev_lat > 0:
            jump = (curr_lat - prev_lat) / prev_lat
            if jump > 0.15:
                boundaries.append(curr_n)
                print(f"  {prev_n:4d} -> {curr_n:4d}: +{jump*100:.1f}% ({prev_lat:.4f} -> {curr_lat:.4f} ms)")

    boundaries.append(max_tokens + step)

    print()
    print(f"Boundaries: {boundaries}")

    # Save raw data
    output_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(output_dir, "latency_raw.npz")
    np.savez(output_path, tokens=tokens, latencies=latencies, boundaries=boundaries)
    print(f"Saved raw data to: {output_path}")

    # Generate visualization
    print()
    print("Generating visualization...")
    visualize(tokens, latencies, boundaries, output_dir)

    print()
    print("=" * 60)
    print("Test 1 Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
