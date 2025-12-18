"""
Latency Profiler for Tile-Spec
==============================
Profiles real model forward pass latency to detect tile boundaries.

Usage:
    python tile_spec/latency_profiler.py --model-path meta-llama/Llama-3.1-8B-Instruct

Output:
    - tile_spec/latency_model.npz (fitted model)
    - tile_spec/latency_plot.png (visualization)
"""

import argparse
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "python"))

from sglang.srt.speculative.tile_spec import PiecewiseLinearLatency


def profile_model_forward(
    model,
    token_counts: list,
    hidden_size: int,
    num_warmup: int = 3,
    num_runs: int = 10,
) -> list:
    """Profile model forward pass at different token counts."""
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype

    results = []
    for n_tokens in token_counts:
        # Create dummy hidden states (simulating verification input)
        hidden_states = torch.randn(n_tokens, hidden_size, device=device, dtype=dtype)

        # Warmup
        for _ in range(num_warmup):
            with torch.no_grad():
                # Run through model layers (simplified - just measures compute)
                for layer in model.model.layers:
                    hidden_states = layer(hidden_states, position_ids=torch.zeros(n_tokens, dtype=torch.long, device=device))[0]
        torch.cuda.synchronize()

        # Measure
        times = []
        for _ in range(num_runs):
            hidden_states = torch.randn(n_tokens, hidden_size, device=device, dtype=dtype)
            torch.cuda.synchronize()
            start = time.perf_counter()
            with torch.no_grad():
                for layer in model.model.layers:
                    hidden_states = layer(hidden_states, position_ids=torch.zeros(n_tokens, dtype=torch.long, device=device))[0]
            torch.cuda.synchronize()
            times.append(time.perf_counter() - start)

        latency_ms = np.median(times) * 1000
        results.append((n_tokens, latency_ms))

        if n_tokens % 64 == 0 or n_tokens <= 16:
            print(f"  n={n_tokens:4d}: {latency_ms:.3f} ms")

    return results


def profile_mlp_latency(
    hidden_size: int = 4096,
    intermediate_size: int = 14336,
    num_layers: int = 32,
    token_counts: list = None,
    num_warmup: int = 5,
    num_runs: int = 20,
) -> list:
    """
    Profile MLP forward pass latency (fallback when model not available).
    Simulates full model by multiplying single layer latency.
    """
    device = "cuda"

    if token_counts is None:
        token_counts = list(range(1, 513))

    # Create MLP weights
    w_gate = torch.randn(intermediate_size, hidden_size, device=device, dtype=torch.float16)
    w_up = torch.randn(intermediate_size, hidden_size, device=device, dtype=torch.float16)
    w_down = torch.randn(hidden_size, intermediate_size, device=device, dtype=torch.float16)

    results = []
    for n_tokens in token_counts:
        x = torch.randn(n_tokens, hidden_size, device=device, dtype=torch.float16)

        def forward():
            gate = torch.nn.functional.linear(x, w_gate)
            up = torch.nn.functional.linear(x, w_up)
            hidden = torch.nn.functional.silu(gate) * up
            return torch.nn.functional.linear(hidden, w_down)

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

        # Scale by number of layers (approximate full model)
        latency_ms = np.median(times) * 1000 * num_layers
        results.append((n_tokens, latency_ms))

        if n_tokens % 64 == 0 or n_tokens <= 8:
            print(f"  n={n_tokens:4d}: {latency_ms:.3f} ms")

    return results


def visualize(tokens, latencies, model, output_dir):
    """Generate visualization."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, skipping visualization")
        return

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

    # Plot 1: Raw latency with boundaries
    ax1.plot(tokens, latencies, "b-", linewidth=1, alpha=0.7, label="Measured Latency")
    for b in model.boundaries[1:-1]:
        ax1.axvline(x=b, color="r", linestyle="--", alpha=0.7, linewidth=1.5)
        ax1.annotate(f"{b}", xy=(b, max(latencies) * 0.95), ha="center", fontsize=9, color="red")

    ax1.set_xlabel("Token Count", fontsize=12)
    ax1.set_ylabel("Latency (ms)", fontsize=12)
    ax1.set_title("Model Forward Latency vs Token Count (Tile Boundaries in Red)", fontsize=14)
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # Plot 2: Latency jump percentage
    jumps, jump_tokens = [], []
    for i in range(1, len(tokens)):
        if latencies[i - 1] > 0:
            jump = (latencies[i] - latencies[i - 1]) / latencies[i - 1] * 100
            jumps.append(jump)
            jump_tokens.append(tokens[i])

    colors = ["red" if j > 15 else "steelblue" for j in jumps]
    ax2.bar(jump_tokens, jumps, width=1, color=colors, alpha=0.7)
    ax2.axhline(y=15, color="r", linestyle="--", linewidth=1.5, label="15% threshold")
    ax2.axhline(y=0, color="black", linewidth=0.5)
    ax2.set_xlabel("Token Count", fontsize=12)
    ax2.set_ylabel("Latency Jump (%)", fontsize=12)
    ax2.set_title("Latency Jump Between Consecutive Token Counts", fontsize=14)
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    ax2.set_ylim(-10, max(50, max(jumps) * 1.1) if jumps else 50)

    plt.tight_layout()
    output_path = os.path.join(output_dir, "latency_plot.png")
    plt.savefig(output_path, dpi=150)
    print(f"Saved plot to: {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Profile model latency for tile-spec")
    parser.add_argument("--model-path", type=str, default=None, help="HuggingFace model path (optional, uses MLP proxy if not provided)")
    parser.add_argument("--hidden-size", type=int, default=4096, help="Hidden size (for MLP mode)")
    parser.add_argument("--intermediate-size", type=int, default=14336, help="Intermediate size (for MLP mode)")
    parser.add_argument("--num-layers", type=int, default=32, help="Number of layers (for MLP mode)")
    parser.add_argument("--max-tokens", type=int, default=512, help="Max tokens to profile")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory")
    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = os.path.dirname(os.path.abspath(__file__))

    print("=" * 60)
    print("Tile-Spec Latency Profiler")
    print("=" * 60)

    if not torch.cuda.is_available():
        print("ERROR: CUDA not available")
        return

    print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Token counts to profile
    token_counts = list(range(1, args.max_tokens + 1))

    if args.model_path:
        # Load real model
        print(f"\nLoading model: {args.model_path}")
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            torch_dtype=torch.float16,
            device_map="cuda",
            trust_remote_code=True,
        )
        model.eval()
        hidden_size = model.config.hidden_size
        print(f"Hidden size: {hidden_size}")
        print(f"\nProfiling model forward pass...")
        print("-" * 60)
        results = profile_model_forward(model, token_counts, hidden_size)
    else:
        # Use MLP proxy
        print(f"\nNo model specified, using MLP proxy")
        print(f"Config: hidden={args.hidden_size}, intermediate={args.intermediate_size}, layers={args.num_layers}")
        print(f"\nProfiling MLP forward pass (scaled by {args.num_layers} layers)...")
        print("-" * 60)
        results = profile_mlp_latency(
            hidden_size=args.hidden_size,
            intermediate_size=args.intermediate_size,
            num_layers=args.num_layers,
            token_counts=token_counts,
        )

    tokens = [r[0] for r in results]
    latencies = [r[1] for r in results]

    # Fit latency model
    print("\n" + "-" * 60)
    print("Fitting PiecewiseLinearLatency model...")
    print("-" * 60)

    latency_model = PiecewiseLinearLatency()
    latency_model.fit(tokens, latencies, jump_threshold=0.15)

    print(f"Detected boundaries: {latency_model.get_boundaries()}")
    print(f"Optimal k candidates: {latency_model.get_optimal_k_candidates()}")

    # Save model
    output_path = os.path.join(args.output_dir, "latency_model.npz")
    latency_model.save(output_path)
    print(f"Saved model to: {output_path}")

    # Generate visualization
    print("\nGenerating visualization...")
    visualize(np.array(tokens), np.array(latencies), latency_model, args.output_dir)

    print("\n" + "=" * 60)
    print("Profiling Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
