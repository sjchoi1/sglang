#!/usr/bin/env python3
"""Test vectorized slicing logic for mixed draft counts."""

import torch


def test_vectorized_slice(draft_counts, scores, tokens):
    """Test the vectorized slicing logic from organize_draft_results."""
    bs = len(draft_counts)
    device = draft_counts.device

    max_drafts = int(draft_counts.max().item())
    print(f"draft_counts: {draft_counts.tolist()}")
    print(f"max_drafts: {max_drafts}")

    if max_drafts > 0:
        # One topk for all requests
        top_k_result = torch.topk(scores, max_drafts, dim=-1)
        top_indices = torch.sort(top_k_result.indices, dim=-1).values
        top_tokens = torch.gather(tokens, dim=1, index=top_indices)

        print(f"top_indices:\n{top_indices}")
        print(f"top_tokens:\n{top_tokens}")

        # Vectorized slicing
        total_drafts = int(draft_counts.sum().item())
        cumsum = draft_counts.cumsum(0)
        offsets = torch.cat([torch.zeros(1, device=device, dtype=torch.long), cumsum[:-1]])

        print(f"total_drafts: {total_drafts}")
        print(f"cumsum: {cumsum.tolist()}")
        print(f"offsets: {offsets.tolist()}")

        expanded_offsets = torch.repeat_interleave(offsets, draft_counts)
        local_indices = torch.arange(total_drafts, device=device, dtype=torch.long) - expanded_offsets
        request_indices = torch.repeat_interleave(torch.arange(bs, device=device), draft_counts)

        print(f"expanded_offsets: {expanded_offsets.tolist()}")
        print(f"local_indices: {local_indices.tolist()}")
        print(f"request_indices: {request_indices.tolist()}")

        # Gather from topk results
        all_indices = top_indices[request_indices, local_indices]
        all_tokens = top_tokens[request_indices, local_indices]

        print(f"all_indices: {all_indices.tolist()}")
        print(f"all_tokens: {all_tokens.tolist()}")

        # Split into lists
        sizes = draft_counts.tolist()
        top_scores_index = list(torch.split(all_indices, sizes))
        draft_tokens = list(torch.split(all_tokens, sizes))

        print(f"\nFinal results:")
        for i, (indices, tokens) in enumerate(zip(top_scores_index, draft_tokens)):
            print(f"  Request {i}: indices={indices.tolist()}, tokens={tokens.tolist()}")
    else:
        print("All zeros - skipped")
        top_scores_index = [torch.empty(0, dtype=torch.long, device=device) for _ in range(bs)]
        draft_tokens = [torch.empty(0, dtype=torch.long, device=device) for _ in range(bs)]

    return top_scores_index, draft_tokens


if __name__ == "__main__":
    # Test case 1: Mixed (some zero, some non-zero)
    print("=" * 60)
    print("Test 1: Mixed draft counts [2, 0, 1]")
    print("=" * 60)
    draft_counts = torch.tensor([2, 0, 1], dtype=torch.long)
    scores = torch.tensor([
        [0.9, 0.8, 0.7, 0.6],  # Request 0: wants top 2
        [0.5, 0.4, 0.3, 0.2],  # Request 1: wants 0
        [0.95, 0.85, 0.75, 0.65],  # Request 2: wants top 1
    ], dtype=torch.float32)
    tokens = torch.tensor([
        [10, 20, 30, 40],
        [50, 60, 70, 80],
        [90, 100, 110, 120],
    ], dtype=torch.long)

    result = test_vectorized_slice(draft_counts, scores, tokens)
    print()

    # Test case 2: All zeros
    print("=" * 60)
    print("Test 2: All zeros [0, 0, 0]")
    print("=" * 60)
    draft_counts = torch.tensor([0, 0, 0], dtype=torch.long)
    result = test_vectorized_slice(draft_counts, scores, tokens)
    print()

    # Test case 3: All non-zero
    print("=" * 60)
    print("Test 3: All non-zero [2, 3, 1]")
    print("=" * 60)
    draft_counts = torch.tensor([2, 3, 1], dtype=torch.long)
    result = test_vectorized_slice(draft_counts, scores, tokens)
    print()

    # Test case 4: Edge - first is zero
    print("=" * 60)
    print("Test 4: First is zero [0, 2, 1]")
    print("=" * 60)
    draft_counts = torch.tensor([0, 2, 1], dtype=torch.long)
    result = test_vectorized_slice(draft_counts, scores, tokens)
