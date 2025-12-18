"""
Fit Calibration Model for Tile-Spec
====================================
Fits calibration model from collected (score, accepted) data.

Usage:
    # First, collect calibration data by running inference with:
    SGLANG_TILE_SPEC_CALIBRATE=1 python -m sglang.launch_server ...

    # Then fit the calibration model:
    python tile_spec/fit_calibration.py --input calibration_raw.npz --output calibration.npz
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "python"))

from sglang.srt.speculative.tile_spec import Calibration


def main():
    parser = argparse.ArgumentParser(description="Fit tile-spec calibration model")
    parser.add_argument("--input", type=str, required=True, help="Input .npz with scores and accepted arrays")
    parser.add_argument("--output", type=str, default=None, help="Output .npz for fitted model")
    parser.add_argument("--num-bins", type=int, default=50, help="Number of calibration bins")
    args = parser.parse_args()

    if args.output is None:
        args.output = os.path.join(os.path.dirname(args.input), "calibration.npz")

    print("=" * 60)
    print("Tile-Spec Calibration Fitting")
    print("=" * 60)

    # Load data
    data = np.load(args.input)
    scores = data["scores"]
    accepted = data["accepted"]

    print(f"Loaded {len(scores)} samples")
    print(f"Score range: [{scores.min():.4f}, {scores.max():.4f}]")
    print(f"Acceptance rate: {accepted.mean():.2%}")
    print()

    # Fit calibration
    calibration = Calibration(num_bins=args.num_bins)
    calibration.fit(scores, accepted)

    # Show bin statistics
    print("Calibration bins (score range -> P(accepted)):")
    print("-" * 60)
    for i in range(0, calibration.num_bins, 5):
        lo = calibration.bin_edges[i]
        hi = calibration.bin_edges[i + 1]
        prob = calibration.bin_probs[i]
        print(f"  [{lo:.2f}, {hi:.2f}): {prob:.3f}")

    # Save
    calibration.save(args.output)
    print()
    print(f"Saved calibration model to: {args.output}")

    print()
    print("=" * 60)
    print("Done!")
    print("=" * 60)


if __name__ == "__main__":
    main()
