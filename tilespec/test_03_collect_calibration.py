"""
Test 3: Collect Calibration Data from EAGLE3 Inference
=======================================================
This script enables calibration data collection and runs EAGLE3 inference
to collect (score, accepted) pairs for calibration fitting.

Two modes:
1. Simulated mode (default): Generate synthetic data to test the pipeline
2. Server mode: Collect real data from running SGLang server

Usage:
    # Simulated mode (no GPU needed)
    python tilespec/test_03_collect_calibration.py --simulate

    # Server mode (requires running SGLang server with EAGLE3)
    # First, start server with calibration collection enabled:
    #   SGLANG_COLLECT_CALIBRATION=1 python -m sglang.launch_server ...
    # Then run:
    python tilespec/test_03_collect_calibration.py --server-url http://localhost:30000

Output:
    - tilespec/calibration_raw.npz (score, accepted pairs)
"""

import argparse
import os
import sys
import numpy as np

# Add parent to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "python"))


def generate_simulated_calibration_data(
    num_samples: int = 10000,
    acceptance_curve: str = "sigmoid",
) -> tuple:
    """
    Generate simulated calibration data for testing.

    The relationship between draft score and acceptance probability
    is modeled as a sigmoid curve (realistic for neural draft models).

    Args:
        num_samples: Number of (score, accepted) pairs to generate
        acceptance_curve: "sigmoid" or "linear"

    Returns:
        (scores, accepted) tuple of numpy arrays
    """
    # Generate scores uniformly in [0, 1]
    scores = np.random.uniform(0, 1, num_samples).astype(np.float32)

    if acceptance_curve == "sigmoid":
        # Sigmoid acceptance probability: higher score = higher acceptance
        # P(accept) = 1 / (1 + exp(-k*(score - 0.5)))
        k = 10  # Steepness
        probs = 1 / (1 + np.exp(-k * (scores - 0.5)))
    elif acceptance_curve == "linear":
        # Linear: P(accept) = score
        probs = scores
    else:
        raise ValueError(f"Unknown curve: {acceptance_curve}")

    # Sample accepted based on probability
    accepted = (np.random.uniform(0, 1, num_samples) < probs).astype(np.float32)

    return scores, accepted


def collect_from_server(server_url: str, num_requests: int = 100):
    """
    Collect calibration data from a running SGLang server.

    Requires server started with SGLANG_COLLECT_CALIBRATION=1 environment variable.
    After sending requests, fetches collected data via special endpoint.
    """
    try:
        import requests
    except ImportError:
        print("Installing requests...")
        import subprocess
        subprocess.check_call(["pip", "install", "requests"])
        import requests

    # Sample prompts for calibration
    prompts = [
        "Explain the theory of relativity in simple terms.",
        "Write a Python function to sort a list.",
        "What are the benefits of exercise?",
        "Describe the process of photosynthesis.",
        "How do computers store information?",
        "What is machine learning?",
        "Explain how a car engine works.",
        "What causes the seasons to change?",
        "Describe the water cycle.",
        "How does the internet work?",
    ]

    print(f"Sending {num_requests} requests to {server_url}...")

    # Send requests to generate calibration data
    for i in range(num_requests):
        prompt = prompts[i % len(prompts)]
        try:
            response = requests.post(
                f"{server_url}/generate",
                json={
                    "text": prompt,
                    "sampling_params": {
                        "max_new_tokens": 64,
                        "temperature": 0.7,
                    }
                },
                timeout=30,
            )
            if response.status_code != 200:
                print(f"Request {i} failed: {response.status_code}")
        except Exception as e:
            print(f"Request {i} error: {e}")

        if (i + 1) % 10 == 0:
            print(f"  Completed {i + 1}/{num_requests} requests")

    # Fetch collected calibration data
    print("Fetching calibration data from server...")
    try:
        response = requests.get(f"{server_url}/get_calibration_data", timeout=30)
        if response.status_code == 200:
            data = response.json()
            scores = np.array(data["scores"], dtype=np.float32)
            accepted = np.array(data["accepted"], dtype=np.float32)
            print(f"Collected {len(scores)} calibration samples")
            return scores, accepted
        else:
            print(f"Failed to fetch calibration data: {response.status_code}")
            print("Note: Server must be started with SGLANG_COLLECT_CALIBRATION=1")
            return None, None
    except Exception as e:
        print(f"Error fetching calibration data: {e}")
        return None, None


def analyze_calibration_data(scores: np.ndarray, accepted: np.ndarray):
    """Analyze and visualize the calibration data."""
    print()
    print("=" * 60)
    print("Calibration Data Analysis")
    print("=" * 60)

    print(f"Total samples: {len(scores)}")
    print(f"Acceptance rate: {accepted.mean():.2%}")
    print(f"Score range: [{scores.min():.4f}, {scores.max():.4f}]")
    print(f"Score mean: {scores.mean():.4f}")
    print()

    # Bin analysis
    num_bins = 10
    bin_edges = np.linspace(0, 1, num_bins + 1)

    print("Score -> Acceptance Rate (binned):")
    print("-" * 60)
    print(f"{'Bin Range':>15} {'Count':>10} {'Acceptance':>12}")

    for i in range(num_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        mask = (scores >= lo) & (scores < hi)
        count = mask.sum()
        if count > 0:
            acc_rate = accepted[mask].mean()
            print(f"  [{lo:.2f}, {hi:.2f})  {count:>8}    {acc_rate:>10.2%}")
        else:
            print(f"  [{lo:.2f}, {hi:.2f})  {count:>8}    {'N/A':>10}")

    # Check monotonicity (higher score should mean higher acceptance)
    print()
    print("Correlation check:")
    correlation = np.corrcoef(scores, accepted)[0, 1]
    print(f"  Pearson correlation: {correlation:.4f}")
    if correlation > 0.3:
        print("  ✓ Positive correlation (good for calibration)")
    elif correlation > 0:
        print("  ⚠ Weak positive correlation (calibration may help)")
    else:
        print("  ✗ Negative/no correlation (calibration unlikely to help)")


def main():
    parser = argparse.ArgumentParser(description="Collect EAGLE3 calibration data")
    parser.add_argument("--simulate", action="store_true", help="Use simulated data")
    parser.add_argument("--server-url", type=str, default=None, help="SGLang server URL")
    parser.add_argument("--num-samples", type=int, default=10000, help="Number of samples (simulated mode)")
    parser.add_argument("--num-requests", type=int, default=100, help="Number of requests (server mode)")
    parser.add_argument("--curve", type=str, default="sigmoid", choices=["sigmoid", "linear"])
    args = parser.parse_args()

    print("=" * 60)
    print("TileSpec Test 3: Collect Calibration Data")
    print("=" * 60)

    output_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(output_dir, "calibration_raw.npz")

    if args.simulate:
        print()
        print("Mode: Simulated data generation")
        print(f"Generating {args.num_samples} samples with {args.curve} curve...")
        scores, accepted = generate_simulated_calibration_data(
            num_samples=args.num_samples,
            acceptance_curve=args.curve,
        )
    elif args.server_url:
        print()
        print(f"Mode: Collect from server at {args.server_url}")
        scores, accepted = collect_from_server(args.server_url, args.num_requests)
        if scores is None:
            print("Failed to collect data. Falling back to simulated data.")
            scores, accepted = generate_simulated_calibration_data()
    else:
        print()
        print("No mode specified. Use --simulate or --server-url")
        print("Defaulting to simulated data...")
        scores, accepted = generate_simulated_calibration_data()

    # Analyze
    analyze_calibration_data(scores, accepted)

    # Save
    np.savez(output_path, scores=scores, accepted=accepted)
    print()
    print(f"Saved calibration data to: {output_path}")
    print(f"  - scores: shape {scores.shape}")
    print(f"  - accepted: shape {accepted.shape}")

    print()
    print("=" * 60)
    print("Test 3 Complete!")
    print("=" * 60)
    print()
    print("Next step: Run test_04_fit_calibration.py to fit the calibration model")


if __name__ == "__main__":
    main()
