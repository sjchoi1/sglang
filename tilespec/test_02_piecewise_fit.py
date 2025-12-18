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


def test_model(model, tokens, latencies, name):
    """Test a model and return error statistics."""
    errors = []
    boundary_errors = []

    for t, actual in zip(tokens, latencies):
        predicted = model.predict(t)
        error = abs(predicted - actual) / actual * 100
        errors.append(error)
        if t in model.boundaries:
            boundary_errors.append(error)

    return {
        "name": name,
        "mean_error": np.mean(errors),
        "max_error": np.max(errors),
        "boundary_mean_error": np.mean(boundary_errors) if boundary_errors else 0,
    }


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

    # Fit piecewise linear model (default)
    print("Fitting PiecewiseLinearLatency model (piecewise linear mode)...")
    print("-" * 60)

    model = PiecewiseLinearLatency(use_lut=False)
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
    print("Prediction Accuracy (Piecewise Linear):")
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

    # Now test LUT mode
    print()
    print("=" * 60)
    print("Comparing Piecewise Linear vs LUT Mode")
    print("=" * 60)

    model_lut = PiecewiseLinearLatency(use_lut=True)
    model_lut.fit(tokens, latencies, jump_threshold=0.15)

    # Compare at boundary points (where optimal k search happens)
    print()
    print("Boundary Latency Comparison:")
    print("-" * 60)
    print(f"{'Boundary':>10} {'Actual':>10} {'Piecewise':>10} {'LUT':>10} {'PW Err':>8} {'LUT Err':>8}")

    for b in model.get_boundaries():
        idx = tokens.index(b) if b in tokens else -1
        actual = latencies[idx] if idx >= 0 else 0

        pw_pred = model.predict(b)
        lut_pred = model_lut.predict(b)

        pw_err = abs(pw_pred - actual) / actual * 100 if actual > 0 else 0
        lut_err = abs(lut_pred - actual) / actual * 100 if actual > 0 else 0

        print(f"{b:>10} {actual:>10.4f} {pw_pred:>10.4f} {lut_pred:>10.4f} {pw_err:>7.2f}% {lut_err:>7.2f}%")

    # Summary statistics
    pw_stats = test_model(model, tokens, latencies, "Piecewise Linear")
    lut_stats = test_model(model_lut, tokens, latencies, "LUT")

    print()
    print("Overall Comparison:")
    print("-" * 60)
    print(f"{'Model':<20} {'Mean Err':>10} {'Max Err':>10} {'Boundary Err':>12}")
    print(f"{'Piecewise Linear':<20} {pw_stats['mean_error']:>9.2f}% {pw_stats['max_error']:>9.2f}% {pw_stats['boundary_mean_error']:>11.2f}%")
    print(f"{'LUT (exact)':<20} {lut_stats['mean_error']:>9.2f}% {lut_stats['max_error']:>9.2f}% {lut_stats['boundary_mean_error']:>11.2f}%")

    # Save LUT model too
    lut_output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "latency_model_lut.npz")
    model_lut.save(lut_output_path)
    print()
    print(f"Saved LUT model to: {lut_output_path}")

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
    print("Note: The algorithm searches ONLY at these boundary points,")
    print("so LUT vs Piecewise Linear mainly differs at boundary accuracy.")

    print()
    print("=" * 60)
    print("Test 2 Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
