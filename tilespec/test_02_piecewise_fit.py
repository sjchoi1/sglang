"""
Test 2: Piecewise Linear Fit Verification
==========================================
Loads latency data from Test 1 and verifies our PiecewiseLinearLatency model
correctly detects boundaries and fits the data.

Usage:
    python tilespec/test_02_piecewise_fit.py

Prerequisites:
    - Run test_01_latency_profiler.py first to generate latency_raw.npz
"""

import os
import sys

import numpy as np

# Add parent to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "python"))

from sglang.srt.speculative.tile_aware import PiecewiseLinearLatency


def main():
    print("=" * 60)
    print("TileSpec Test 2: Piecewise Linear Fit Verification")
    print("=" * 60)

    # Load raw data from Test 1
    data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "latency_raw.npz")
    if not os.path.exists(data_path):
        print(f"ERROR: {data_path} not found. Run test_01_latency_profiler.py first.")
        return

    data = np.load(data_path)
    tokens = data["tokens"].tolist()
    latencies = data["latencies"].tolist()

    print(f"Loaded {len(tokens)} data points")
    print()

    # Fit our model
    print("Fitting PiecewiseLinearLatency model...")
    print("-" * 60)

    model = PiecewiseLinearLatency()
    model.fit(tokens, latencies, jump_threshold=0.15)

    print(f"Detected boundaries: {model.boundaries}")
    print(f"Segments: {len(model.slopes)}")
    print()

    # Show segment details
    print("Segment Details:")
    print("-" * 60)
    for i in range(len(model.slopes)):
        lo = model.boundaries[i]
        hi = model.boundaries[i + 1] if i + 1 < len(model.boundaries) else "∞"
        print(f"  [{lo:4} - {hi:>4}): slope={model.slopes[i]:.6f}, intercept={model.intercepts[i]:.4f}")

    print()

    # Test predictions vs actual
    print("Prediction Accuracy:")
    print("-" * 60)
    print(f"{'Tokens':>8} {'Actual':>10} {'Predicted':>10} {'Error':>10}")

    errors = []
    for t, actual in zip(tokens, latencies):
        predicted = model.predict(t)
        error = abs(predicted - actual) / actual * 100
        errors.append(error)
        if t % 64 == 0 or t in model.boundaries:  # Show key points
            print(f"{t:>8} {actual:>10.4f} {predicted:>10.4f} {error:>9.1f}%")

    print()
    print(f"Mean absolute error: {np.mean(errors):.2f}%")
    print(f"Max absolute error:  {np.max(errors):.2f}%")

    # Save fitted model
    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "latency_model.npz")
    model.save(output_path)
    print()
    print(f"Saved fitted model to: {output_path}")

    # Verify reload works
    print()
    print("Verifying model save/load...")
    model2 = PiecewiseLinearLatency()
    model2.load(output_path)
    assert model2.boundaries == model.boundaries, "Boundaries mismatch after reload"
    assert model2.slopes == model.slopes, "Slopes mismatch after reload"
    print("✓ Model save/load verified")

    # Show optimal k values (the tile boundaries)
    print()
    print("=" * 60)
    print("Optimal k values (tile boundaries):")
    print("=" * 60)
    optimal_k = model.get_boundaries()
    print(f"  {optimal_k}")
    print()
    print("These are the token counts where you get maximum efficiency.")
    print("Going above these values incurs a latency jump with minimal benefit.")

    print()
    print("=" * 60)
    print("Test 2 Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
