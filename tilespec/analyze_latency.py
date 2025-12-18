"""
Analyze latency data to find both jumps AND drops.
"""
import numpy as np
import os

data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "latency_raw.npz")
data = np.load(data_path)
tokens = data["tokens"]
latencies = data["latencies"]

print("=" * 70)
print("Latency Analysis: Jumps and Drops")
print("=" * 70)

print("\n### Significant JUMPS (>15% increase):")
print("-" * 70)
for i in range(1, len(tokens)):
    if latencies[i-1] > 0:
        change = (latencies[i] - latencies[i-1]) / latencies[i-1] * 100
        if change > 15:
            print(f"  {tokens[i-1]:4d} -> {tokens[i]:4d}: +{change:5.1f}%  ({latencies[i-1]:.4f} -> {latencies[i]:.4f} ms)")

print("\n### Significant DROPS (>10% decrease):")
print("-" * 70)
for i in range(1, len(tokens)):
    if latencies[i-1] > 0:
        change = (latencies[i] - latencies[i-1]) / latencies[i-1] * 100
        if change < -10:
            print(f"  {tokens[i-1]:4d} -> {tokens[i]:4d}: {change:5.1f}%  ({latencies[i-1]:.4f} -> {latencies[i]:.4f} ms)")

print("\n### Latency at key points (multiples of 64):")
print("-" * 70)
for n in [1, 32, 64, 65, 96, 128, 129, 192, 193, 256, 257, 320, 384, 385, 448, 512]:
    if n <= len(tokens):
        idx = n - 1
        print(f"  n={n:4d}: {latencies[idx]:.4f} ms")

print("\n### Statistics per segment:")
print("-" * 70)
boundaries = [1, 65, 129, 193, 257, 385, 513]
for i in range(len(boundaries) - 1):
    lo, hi = boundaries[i], boundaries[i+1]
    mask = (tokens >= lo) & (tokens < hi)
    segment_lats = latencies[mask]
    if len(segment_lats) > 0:
        print(f"  [{lo:3d}-{hi:3d}): mean={segment_lats.mean():.4f}, std={segment_lats.std():.4f}, min={segment_lats.min():.4f}, max={segment_lats.max():.4f}")
