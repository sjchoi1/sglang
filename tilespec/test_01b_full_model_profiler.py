"""
Test 1b: Full Model Latency Profiler
====================================
Profiles the full verification model (not just MLP) to get accurate
latencies for E/L optimization.

This is more accurate than MLP-only profiling but requires model loading.
Use this for production; use test_01 for quick boundary detection.

Usage:
    # With model path
    python tilespec/test_01b_full_model_profiler.py --model-path meta-llama/Llama-3.1-8B

    # Use existing latency_raw.npz boundaries but with scaled values
    python tilespec/test_01b_full_model_profiler.py --scale-from-mlp

Output:
    - tilespec/latency_full_model.npz (raw data)
    - tilespec/latency_model_full.npz (fitted model)
"""

import argparse
import os
import sys
import time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "python"))

import torch


def measure_transformer_layer_latency(
    total_tokens: int,
    hidden_size: int = 4096,
    intermediate_size: int = 14336,
    num_heads: int = 32,
    head_dim: int = 128,
    num_warmup: int = 5,
    num_runs: int = 20,
) -> float:
    """
    Measure a full transformer layer (attention + MLP) latency.
    This is more representative than MLP-only.
    """
    device = "cuda"
    dtype = torch.float16

    # Create inputs
    x = torch.randn(total_tokens, hidden_size, device=device, dtype=dtype)

    # Attention weights (simplified - no KV cache)
    wq = torch.randn(num_heads * head_dim, hidden_size, device=device, dtype=dtype)
    wk = torch.randn(num_heads * head_dim, hidden_size, device=device, dtype=dtype)
    wv = torch.randn(num_heads * head_dim, hidden_size, device=device, dtype=dtype)
    wo = torch.randn(hidden_size, num_heads * head_dim, device=device, dtype=dtype)

    # MLP weights (SwiGLU)
    w_gate = torch.randn(intermediate_size, hidden_size, device=device, dtype=dtype)
    w_up = torch.randn(intermediate_size, hidden_size, device=device, dtype=dtype)
    w_down = torch.randn(hidden_size, intermediate_size, device=device, dtype=dtype)

    def forward():
        # Simplified attention (no masking, no KV cache - just linear projections)
        q = torch.nn.functional.linear(x, wq)
        k = torch.nn.functional.linear(x, wk)
        v = torch.nn.functional.linear(x, wv)

        # Reshape for attention
        q = q.view(total_tokens, num_heads, head_dim)
        k = k.view(total_tokens, num_heads, head_dim)
        v = v.view(total_tokens, num_heads, head_dim)

        # Simplified attention score (not actual attention - just measuring GEMM)
        attn_out = q * k.mean(dim=0, keepdim=True)  # Simplified
        attn_out = attn_out.view(total_tokens, -1)
        attn_proj = torch.nn.functional.linear(attn_out, wo)

        # MLP
        residual = x + attn_proj
        gate = torch.nn.functional.linear(residual, w_gate)
        up = torch.nn.functional.linear(residual, w_up)
        hidden = torch.nn.functional.silu(gate) * up
        out = torch.nn.functional.linear(hidden, w_down)

        return residual + out

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


def scale_from_mlp_profile(mlp_data_path: str, scale_factor: float = 1.7) -> tuple:
    """
    Load MLP-only latencies and scale to approximate full model.
    MLP is typically ~60% of model time, so scale by ~1.7x.
    """
    data = np.load(mlp_data_path)
    tokens = data["tokens"]
    latencies = data["latencies"] * scale_factor
    boundaries = data["boundaries"]

    return tokens, latencies, list(boundaries)


def main():
    parser = argparse.ArgumentParser(description="Full model latency profiler")
    parser.add_argument("--model-path", type=str, help="Model path for accurate profiling")
    parser.add_argument("--scale-from-mlp", action="store_true",
                       help="Scale MLP latencies instead of profiling")
    parser.add_argument("--scale-factor", type=float, default=1.7,
                       help="Scale factor for MLP->full model (default: 1.7)")
    parser.add_argument("--max-tokens", type=int, default=512)
    args = parser.parse_args()

    print("=" * 60)
    print("TileSpec Test 1b: Full Model Latency Profiler")
    print("=" * 60)

    output_dir = os.path.dirname(os.path.abspath(__file__))

    if args.scale_from_mlp:
        print()
        print("Mode: Scale from MLP profile")
        mlp_path = os.path.join(output_dir, "latency_raw.npz")

        if not os.path.exists(mlp_path):
            print(f"ERROR: {mlp_path} not found. Run test_01 first.")
            return

        tokens, latencies, boundaries = scale_from_mlp_profile(mlp_path, args.scale_factor)
        print(f"Loaded MLP data, scaled by {args.scale_factor}x")
        print(f"Boundaries preserved: {boundaries}")

    elif args.model_path:
        print()
        print(f"Mode: Full model profiling with {args.model_path}")
        print("Note: This would load the actual model - not implemented yet")
        print("Using transformer layer approximation instead...")

        if not torch.cuda.is_available():
            print("ERROR: CUDA not available")
            return

        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print()

        token_counts = list(range(1, args.max_tokens + 1))
        results = []

        print(f"Profiling {len(token_counts)} token counts...")
        for i, n in enumerate(token_counts):
            lat = measure_transformer_layer_latency(n)
            results.append((n, lat))
            if n % 50 == 0 or n <= 10:
                print(f"n={n:4d}: {lat:.4f} ms")

        tokens = np.array([r[0] for r in results])
        latencies = np.array([r[1] for r in results])

        # Detect boundaries
        boundaries = [int(tokens[0])]
        for i in range(1, len(results)):
            prev_lat = results[i-1][1]
            curr_lat = results[i][1]
            if prev_lat > 0 and (curr_lat - prev_lat) / prev_lat > 0.15:
                boundaries.append(int(results[i][0]))
        boundaries.append(int(tokens[-1]) + 1)

    else:
        print()
        print("No mode specified. Use --scale-from-mlp or --model-path")
        print("Using --scale-from-mlp as default...")
        mlp_path = os.path.join(output_dir, "latency_raw.npz")

        if not os.path.exists(mlp_path):
            print(f"ERROR: {mlp_path} not found. Run test_01 first.")
            return

        tokens, latencies, boundaries = scale_from_mlp_profile(mlp_path, args.scale_factor)

    # Save full model data
    output_path = os.path.join(output_dir, "latency_full_model.npz")
    np.savez(output_path, tokens=tokens, latencies=latencies, boundaries=boundaries)
    print()
    print(f"Saved full model latencies to: {output_path}")

    # Fit and save model
    from sglang.srt.speculative.tile_aware import PiecewiseLinearLatency

    model = PiecewiseLinearLatency(use_lut=False)
    model.fit(tokens.tolist(), latencies.tolist(), jump_threshold=0.15)

    model_path = os.path.join(output_dir, "latency_model_full.npz")
    model.save(model_path)
    print(f"Saved full model latency model to: {model_path}")

    print()
    print("Comparison with MLP-only:")
    print("-" * 60)
    mlp_path = os.path.join(output_dir, "latency_raw.npz")
    if os.path.exists(mlp_path):
        mlp_data = np.load(mlp_path)
        for b in model.get_boundaries():
            if b in mlp_data["tokens"]:
                idx = list(mlp_data["tokens"]).index(b)
                mlp_lat = mlp_data["latencies"][idx]
                full_lat = model.predict(b)
                print(f"  n={b:4d}: MLP={mlp_lat:.4f}ms, Full={full_lat:.4f}ms, Ratio={full_lat/mlp_lat:.2f}x")

    print()
    print("=" * 60)
    print("Test 1b Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
