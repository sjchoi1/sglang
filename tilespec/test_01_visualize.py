"""
Test 1 Visualization: Latency vs Token Count
=============================================
Plots the latency data from Test 1 to visualize tile boundaries.

Usage:
    python tilespec/test_01_visualize.py

Output:
    tilespec/latency_plot.png
"""

import os
import numpy as np

try:
    import matplotlib.pyplot as plt
except ImportError:
    print("Installing matplotlib...")
    import subprocess
    subprocess.check_call(["pip", "install", "matplotlib"])
    import matplotlib.pyplot as plt


def main():
    # Load raw data from Test 1
    data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "latency_raw.npz")
    if not os.path.exists(data_path):
        print(f"ERROR: {data_path} not found. Run test_01_latency_profiler.py first.")
        return

    data = np.load(data_path)
    tokens = data["tokens"]
    latencies = data["latencies"]
    boundaries = data["boundaries"].tolist()

    # Create figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

    # Plot 1: Raw latency
    ax1.plot(tokens, latencies, 'b-', linewidth=1.5, marker='o', markersize=3, label='Measured Latency')

    # Mark tile boundaries
    for b in boundaries[1:-1]:  # Skip first and last
        ax1.axvline(x=b, color='r', linestyle='--', alpha=0.7, linewidth=1)

    # Add boundary labels
    for i, b in enumerate(boundaries[1:-1]):
        ax1.annotate(f'{b}', xy=(b, ax1.get_ylim()[1]),
                     xytext=(b, ax1.get_ylim()[1] * 0.95),
                     ha='center', fontsize=8, color='red')

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

    ax2.bar(jump_tokens, jumps, width=6, color='steelblue', alpha=0.7)
    ax2.axhline(y=15, color='r', linestyle='--', linewidth=1.5, label='15% threshold')

    # Highlight significant jumps
    for i, (t, j) in enumerate(zip(jump_tokens, jumps)):
        if j > 15:
            ax2.bar(t, j, width=6, color='red', alpha=0.8)

    ax2.set_xlabel('Token Count', fontsize=12)
    ax2.set_ylabel('Latency Jump (%)', fontsize=12)
    ax2.set_title('Latency Jump Between Consecutive Token Counts', fontsize=14)
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    plt.tight_layout()

    # Save
    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "latency_plot.png")
    plt.savefig(output_path, dpi=150)
    print(f"Saved plot to: {output_path}")

    # Also show summary stats
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"Token range: {tokens[0]} - {tokens[-1]}")
    print(f"Latency range: {latencies.min():.4f} - {latencies.max():.4f} ms")
    print(f"Detected boundaries: {boundaries}")
    print(f"Significant jumps (>15%):")
    for t, j in zip(jump_tokens, jumps):
        if j > 15:
            print(f"  At {t}: +{j:.1f}%")


if __name__ == "__main__":
    main()
