#!/usr/bin/env python3
"""Benchmark vectorized vs loop approach for draft token selection."""

import time
import torch
import numpy as np


def vectorized_selection(scores, tokens, draft_counts):
    """Current vectorized approach."""
    bs = len(draft_counts)
    device = draft_counts.device
    max_drafts = int(draft_counts.max().item())

    if max_drafts > 0:
        # One topk for all requests
        top_k_result = torch.topk(scores, max_drafts, dim=-1)
        top_indices = torch.sort(top_k_result.indices, dim=-1).values
        top_tokens = torch.gather(tokens, dim=1, index=top_indices)

        # Vectorized slicing
        total_drafts = int(draft_counts.sum().item())
        cumsum = draft_counts.cumsum(0)
        offsets = torch.cat([torch.zeros(1, device=device, dtype=torch.long), cumsum[:-1]])
        expanded_offsets = torch.repeat_interleave(offsets, draft_counts)
        local_indices = torch.arange(total_drafts, device=device, dtype=torch.long) - expanded_offsets
        request_indices = torch.repeat_interleave(torch.arange(bs, device=device), draft_counts)

        # Gather from topk results
        all_indices = top_indices[request_indices, local_indices]
        all_tokens = top_tokens[request_indices, local_indices]

        # Split into lists
        sizes = draft_counts.tolist()
        top_scores_index = list(torch.split(all_indices, sizes))
        draft_tokens = list(torch.split(all_tokens, sizes))
    else:
        top_scores_index = [torch.empty(0, dtype=torch.long, device=device) for _ in range(bs)]
        draft_tokens = [torch.empty(0, dtype=tokens.dtype, device=device) for _ in range(bs)]

    return top_scores_index, draft_tokens


def loop_selection(scores, tokens, draft_counts):
    """Simple loop approach."""
    bs = draft_counts.shape[0]
    device = draft_counts.device
    top_scores_index = []
    draft_tokens_list = []

    for i in range(bs):
        k = int(draft_counts[i].item())
        if k > 0:
            top_k = torch.topk(scores[i], k)
            indices = torch.sort(top_k.indices).values
            top_scores_index.append(indices)
            draft_tokens_list.append(torch.gather(tokens[i], dim=0, index=indices))
        else:
            top_scores_index.append(torch.empty(0, dtype=torch.long, device=device))
            draft_tokens_list.append(torch.empty(0, dtype=tokens.dtype, device=device))

    return top_scores_index, draft_tokens_list


def generate_draft_counts(bs, max_drafts, variance):
    """Generate draft counts with specified variance pattern."""
    if variance == "low":
        # Similar counts across all requests
        base = max_drafts // 2
        counts = torch.randint(max(1, base - 1), base + 2, (bs,), dtype=torch.long)
    elif variance == "medium":
        # Mix of low and medium counts
        counts = torch.randint(0, max_drafts + 1, (bs,), dtype=torch.long)
    elif variance == "high":
        # Concentrate in few requests
        counts = torch.zeros(bs, dtype=torch.long)
        # Put most drafts in first 25% of requests
        num_high = max(1, bs // 4)
        counts[:num_high] = torch.randint(max_drafts - 1, max_drafts + 1, (num_high,), dtype=torch.long)
    else:
        raise ValueError(f"Unknown variance: {variance}")

    return counts


def benchmark(bs, n_cand, max_drafts, variance, num_runs=100, device="cuda"):
    """Run benchmark for given configuration."""
    if not torch.cuda.is_available():
        device = "cpu"

    # Generate test data
    scores = torch.rand(bs, n_cand, device=device)
    tokens = torch.randint(0, 50000, (bs, n_cand), device=device)
    draft_counts = generate_draft_counts(bs, max_drafts, variance).to(device)

    # Warmup
    for _ in range(10):
        vectorized_selection(scores, tokens, draft_counts)
        loop_selection(scores, tokens, draft_counts)

    if device == "cuda":
        torch.cuda.synchronize()

    # Benchmark vectorized
    start = time.perf_counter()
    for _ in range(num_runs):
        vectorized_selection(scores, tokens, draft_counts)
    if device == "cuda":
        torch.cuda.synchronize()
    vectorized_time = (time.perf_counter() - start) / num_runs * 1000  # ms

    # Benchmark loop
    start = time.perf_counter()
    for _ in range(num_runs):
        loop_selection(scores, tokens, draft_counts)
    if device == "cuda":
        torch.cuda.synchronize()
    loop_time = (time.perf_counter() - start) / num_runs * 1000  # ms

    return {
        "vectorized_ms": vectorized_time,
        "loop_ms": loop_time,
        "speedup": vectorized_time / loop_time,
        "draft_counts": draft_counts.cpu().tolist(),
    }


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"{'='*80}")

    # Configuration
    batch_sizes = [1, 4, 8, 16, 32, 64]
    n_cand = 16  # Typical: 4 draft steps × 4 topk
    max_drafts = 4
    variances = ["low", "medium", "high"]
    num_runs = 1000 if device == "cuda" else 100

    results = []

    for bs in batch_sizes:
        for variance in variances:
            result = benchmark(bs, n_cand, max_drafts, variance, num_runs, device)
            results.append({
                "bs": bs,
                "variance": variance,
                **result
            })

            print(f"BS={bs:2d} variance={variance:6s} | "
                  f"Vectorized: {result['vectorized_ms']:.4f}ms | "
                  f"Loop: {result['loop_ms']:.4f}ms | "
                  f"Speedup: {result['speedup']:.2f}x | "
                  f"Counts: {result['draft_counts'][:5]}{'...' if bs > 5 else ''}")

    print(f"{'='*80}")
    print("\nSummary:")
    for variance in variances:
        var_results = [r for r in results if r["variance"] == variance]
        avg_speedup = np.mean([r["speedup"] for r in var_results])
        print(f"  {variance:6s} variance - Avg speedup: {avg_speedup:.2f}x "
              f"({'Loop' if avg_speedup > 1 else 'Vectorized'} is faster)")
