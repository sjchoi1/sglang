"""
Test 4: Fit Calibration Model
=============================
Loads calibration data from Test 3 and fits the Calibration model
to map draft scores to acceptance probabilities.

Usage:
    python tilespec/test_04_fit_calibration.py

Prerequisites:
    - Run test_03_collect_calibration.py first to generate calibration_raw.npz

Output:
    - tilespec/calibration.npz (fitted calibration model)
"""

import os
import sys
import numpy as np

# Add parent to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "python"))

from sglang.srt.speculative.tile_aware import Calibration

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


def visualize_calibration(scores, accepted, calibration, output_dir):
    """Visualize calibration fit."""
    global HAS_MATPLOTLIB
    if not HAS_MATPLOTLIB:
        print("Installing matplotlib...")
        import subprocess
        subprocess.check_call(["pip", "install", "matplotlib"])
        HAS_MATPLOTLIB = True

    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Binned empirical vs fitted
    num_bins = 20
    bin_edges = np.linspace(0, 1, num_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    empirical_probs = []
    fitted_probs = []
    bin_counts = []

    for i in range(num_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        mask = (scores >= lo) & (scores < hi)
        if mask.sum() > 0:
            empirical_probs.append(accepted[mask].mean())
            bin_counts.append(mask.sum())
        else:
            empirical_probs.append(np.nan)
            bin_counts.append(0)

        # Get fitted probability for bin center (lookup by value, not index)
        bin_center = (lo + hi) / 2
        cal_bin_idx = min(int(bin_center * calibration.num_bins), calibration.num_bins - 1)
        fitted_probs.append(calibration.bin_probs[cal_bin_idx])

    # Bar chart comparison
    x = np.arange(num_bins)
    width = 0.35

    ax1.bar(x - width/2, empirical_probs, width, label='Empirical', alpha=0.7)
    ax1.bar(x + width/2, fitted_probs, width, label='Fitted', alpha=0.7)
    ax1.set_xlabel('Score Bin')
    ax1.set_ylabel('Acceptance Probability')
    ax1.set_title('Empirical vs Fitted Acceptance Probability')
    ax1.set_xticks(x[::2])
    ax1.set_xticklabels([f'{b:.1f}' for b in bin_centers[::2]])
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Calibration curve
    score_range = np.linspace(0, 1, 100)
    fitted_curve = []
    for s in score_range:
        bin_idx = min(int(s * calibration.num_bins), calibration.num_bins - 1)
        fitted_curve.append(calibration.bin_probs[bin_idx])

    ax2.plot(score_range, fitted_curve, 'b-', linewidth=2, label='Fitted Calibration')
    ax2.scatter(scores[::max(1, len(scores)//500)],
                accepted[::max(1, len(scores)//500)],
                alpha=0.1, s=5, c='gray', label='Data (sampled)')

    ax2.set_xlabel('Draft Score')
    ax2.set_ylabel('Acceptance Probability')
    ax2.set_title('Calibration Curve')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, 1)
    ax2.set_ylim(-0.05, 1.05)

    plt.tight_layout()

    output_path = os.path.join(output_dir, "calibration_plot.png")
    plt.savefig(output_path, dpi=150)
    print(f"Saved calibration plot to: {output_path}")
    plt.close()


def main():
    print("=" * 60)
    print("TileSpec Test 4: Fit Calibration Model")
    print("=" * 60)

    # Load raw data from Test 3
    output_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(output_dir, "calibration_raw.npz")

    if not os.path.exists(data_path):
        print(f"ERROR: {data_path} not found. Run test_03_collect_calibration.py first.")
        return

    data = np.load(data_path)
    scores = data["scores"]
    accepted = data["accepted"]

    print(f"Loaded {len(scores)} calibration samples")
    print()

    # Fit calibration model
    print("Fitting Calibration model...")
    print("-" * 60)

    calibration = Calibration(num_bins=50)
    calibration.fit(scores, accepted)

    print(f"Number of bins: {calibration.num_bins}")
    print()

    # Show bin probabilities
    print("Bin Probabilities (first 10 and last 10):")
    print("-" * 60)
    print("Low scores (0.0 - 0.2):")
    for i in range(10):
        lo = calibration.bin_edges[i]
        hi = calibration.bin_edges[i + 1]
        print(f"  [{lo:.2f}, {hi:.2f}): {calibration.bin_probs[i]:.4f}")

    print("\nHigh scores (0.8 - 1.0):")
    for i in range(40, 50):
        lo = calibration.bin_edges[i]
        hi = calibration.bin_edges[i + 1]
        print(f"  [{lo:.2f}, {hi:.2f}): {calibration.bin_probs[i]:.4f}")

    print()

    # Evaluate fit quality
    print("Fit Quality Evaluation:")
    print("-" * 60)

    # Compute calibration error
    predicted_probs = []
    for s in scores:
        bin_idx = min(int(s * calibration.num_bins), calibration.num_bins - 1)
        predicted_probs.append(calibration.bin_probs[bin_idx])
    predicted_probs = np.array(predicted_probs)

    # Brier score (lower is better)
    brier_score = np.mean((predicted_probs - accepted) ** 2)
    print(f"Brier Score: {brier_score:.4f} (lower is better, 0 = perfect)")

    # Expected Calibration Error (ECE)
    ece = 0.0
    num_bins = 10
    for i in range(num_bins):
        lo, hi = i / num_bins, (i + 1) / num_bins
        mask = (predicted_probs >= lo) & (predicted_probs < hi)
        if mask.sum() > 0:
            bin_acc = accepted[mask].mean()
            bin_conf = predicted_probs[mask].mean()
            ece += mask.sum() / len(scores) * abs(bin_acc - bin_conf)

    print(f"Expected Calibration Error (ECE): {ece:.4f} (lower is better)")

    # Log loss
    eps = 1e-7
    log_loss = -np.mean(
        accepted * np.log(np.clip(predicted_probs, eps, 1 - eps)) +
        (1 - accepted) * np.log(np.clip(1 - predicted_probs, eps, 1 - eps))
    )
    print(f"Log Loss: {log_loss:.4f}")

    print()

    # Save fitted model
    output_path = os.path.join(output_dir, "calibration.npz")
    calibration.save(output_path)
    print(f"Saved fitted calibration to: {output_path}")

    # Verify reload
    print()
    print("Verifying model save/load...")
    calibration2 = Calibration()
    calibration2.load(output_path)
    assert np.allclose(calibration2.bin_probs, calibration.bin_probs), "Mismatch after reload"
    print("✓ Model save/load verified")

    # Generate visualization
    print()
    print("Generating visualization...")
    visualize_calibration(scores, accepted, calibration, output_dir)

    print()
    print("=" * 60)
    print("Test 4 Complete!")
    print("=" * 60)
    print()
    print("Calibration model fitted and saved.")
    print("Next step: Run test_05_end_to_end.py for end-to-end testing")


if __name__ == "__main__":
    main()
