import math
from enum import IntEnum
from typing import List, Optional

import torch

from sglang.srt.utils import is_cuda, is_hip, is_npu

_is_cuda = is_cuda()
_is_hip = is_hip()
_is_npu = is_npu()

if _is_cuda or _is_hip:
    from sgl_kernel import (
        build_tree_kernel_efficient as sgl_build_tree_kernel_efficient,
    )


def organize_draft_results(
    score_list: List[torch.Tensor],
    token_list: List[torch.Tensor],
    parents_list: List[torch.Tensor],
    num_draft_token: int,
    latency_model=None,
    is_tilespec_profiling: bool = False,
):
    """
    Organize draft results and select tokens for verification.

    Args:
        score_list: List of score tensors from each draft step
        token_list: List of token tensors from each draft step
        parents_list: List of parent tensors from each draft step
        num_draft_token: Maximum draft tokens (always int)
        latency_model: TileSpec PiecewiseLinearLatency object (optional)
        is_tilespec_profiling: Whether TileSpec is in profiling mode

    Returns:
        parent_list: Parent indices for tree building
        top_scores_index: Selected token indices (tensor or list)
        draft_tokens: Selected tokens (tensor or list)
        per_request_draft_token_num: Per-request token counts (None if uniform)

    Note:
        Scores are used directly as P(accept) - no calibration needed.
        Empirically validated: cumulative draft score ≈ acceptance probability.
    """
    scores = torch.cat(score_list, dim=1).flatten(1)
    tokens = torch.cat(token_list, dim=1)
    bs, n_cand = scores.shape
    device = scores.device

    # Check if TileSpec is active
    tilespec_active = is_tilespec_profiling or (latency_model is not None)

    if tilespec_active:
        # Step 1: Determine draft counts per request
        if is_tilespec_profiling:
            # Profiling: mix random and full draft counts
            # - Random (50%): granular token counts for exact boundary detection
            # - Full (50%): high token count coverage for 192, 256+ boundaries
            if torch.rand(1, device=device).item() < 0.5:
                draft_counts = torch.randint(0, n_cand + 1, (bs,), device=device, dtype=torch.long)
            else:
                draft_counts = torch.full((bs,), n_cand, device=device, dtype=torch.long)
        else:
            # Runtime: optimal counts from global E/L optimization
            # Use scores directly as P(accept) - no calibration needed
            # (empirically validated: cumulative draft score ≈ acceptance probability)
            probs = scores
            sorted_probs, sorted_indices = torch.sort(probs.flatten(), descending=True)
            cum_E = torch.cumsum(sorted_probs, dim=0)

            # Get latencies for bs, bs+1, ..., bs+n_cand tokens
            max_total = bs + len(cum_E)
            latencies_all = latency_model.predict_batch(max_total, device)

            # For k drafts: total tokens = bs + (k+1), latency = latencies_all[bs + k]
            latencies = latencies_all[bs:bs + len(cum_E)]
            expected_throughput = (cum_E + bs) / latencies

            # Consider 0-draft option
            expected_throughput_zero = bs / latencies_all[bs - 1]  # bs tokens / latency(bs)

            if len(cum_E) == 0 or expected_throughput_zero >= expected_throughput.max():
                total = 0
            else:
                best_idx = expected_throughput.argmax().item()
                total = best_idx + 1

            # Compute draft counts
            if total > 0:
                request_ids = sorted_indices[:total] // n_cand
                draft_counts = torch.bincount(request_ids, minlength=bs)
            else:
                draft_counts = torch.zeros(bs, dtype=torch.long, device=device)

        # Step 2: Unified selection - single topk, vectorized split
        per_request_draft_token_num = draft_counts + 1  # +1 for verified token
        max_drafts = int(draft_counts.max().item())

        if max_drafts > 0:
            # One topk for all requests
            top_k_result = torch.topk(scores, max_drafts, dim=-1)
            top_indices = torch.sort(top_k_result.indices, dim=-1).values  # [bs, max_drafts]
            top_tokens = torch.gather(tokens, dim=1, index=top_indices)  # [bs, max_drafts]

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

            # Only compute selected_scores during profiling (for calibration data)
            if is_tilespec_profiling:
                top_scores_values = torch.gather(scores, dim=1, index=top_indices)
                all_scores = top_scores_values[request_indices, local_indices]
                selected_scores = list(torch.split(all_scores, sizes))
            else:
                selected_scores = None
        else:
            top_scores_index = [torch.empty(0, dtype=torch.long, device=device) for _ in range(bs)]
            draft_tokens = [torch.empty(0, dtype=tokens.dtype, device=device) for _ in range(bs)]
            selected_scores = None

        # Build parent_list for TileSpec case
        if len(parents_list) > 1:
            parent_list = torch.cat(parents_list[:-1], dim=1)
        else:
            parent_list = torch.empty(bs, 0, device=device)

        return parent_list, top_scores_index, draft_tokens, per_request_draft_token_num, selected_scores

    else:
        # Original EAGLE: uniform selection (returns 3 values for CUDA graph compatibility)
        top_scores = torch.topk(scores, num_draft_token - 1, dim=-1)
        top_scores_index = torch.sort(top_scores.indices).values
        draft_tokens = torch.gather(tokens, index=top_scores_index, dim=1)

        # Build parent_list
        if len(parents_list) > 1:
            parent_list = torch.cat(parents_list[:-1], dim=1)
        else:
            parent_list = torch.empty(bs, 0, device=device)

        return parent_list, top_scores_index, draft_tokens


class TreeMaskMode(IntEnum):
    FULL_MASK = 0
    QLEN_ONLY = 1
    QLEN_ONLY_BITPACKING = 2


def build_tree_kernel_efficient(
    verified_id: torch.Tensor,
    parent_list: List[torch.Tensor],
    top_scores_index: torch.Tensor,
    draft_tokens: torch.Tensor,
    seq_lens: torch.Tensor,
    seq_lens_sum: int,
    topk: int,
    spec_steps: int,
    num_verify_tokens: int,
    tree_mask_mode: TreeMaskMode = TreeMaskMode.FULL_MASK,
    tree_mask_buf: Optional[torch.Tensor] = None,
    position_buf: Optional[torch.Tensor] = None,
    per_request_draft_token_num: Optional[torch.Tensor] = None,
):
    # Handle both uniform (Tensor) and ragged (List[Tensor]) draft tokens
    if isinstance(draft_tokens, list):
        # Ragged case: per-request variable draft counts
        draft_tokens_with_verified = []
        for i, req_drafts in enumerate(draft_tokens):
            draft_tokens_with_verified.append(torch.cat([verified_id[i:i+1], req_drafts]))
        draft_tokens = torch.cat(draft_tokens_with_verified)

        # Also flatten top_scores_index (which is also a list in ragged case)
        top_scores_index = torch.cat(top_scores_index) if isinstance(top_scores_index, list) else top_scores_index
    else:
        # Uniform case: original logic
        draft_tokens = torch.cat((verified_id.unsqueeze(1), draft_tokens), dim=1).flatten()

    # seq_lens_sum == sum(seq_lens); seq_lens: sequence length without draft tokens
    bs = seq_lens.numel()
    device = seq_lens.device

    # Calculate actual verify token counts (uniform or per-request)
    if per_request_draft_token_num is not None:
        total_verify_tokens = per_request_draft_token_num.sum().item()  # Sum for flattened buffers
        max_verify_tokens = per_request_draft_token_num.max().item()    # Max for per-batch buffers
    else:
        total_verify_tokens = bs * num_verify_tokens
        max_verify_tokens = num_verify_tokens
    # e.g. for bs=1, tree_mask: num_draft_token, seq_lens_sum + num_draft_token (flattened)
    # where each row indicates the attending pattern of each draft token
    # if use_partial_packed_tree_mask is True, tree_mask: num_draft_token (flattened, packed)
    if tree_mask_buf is not None:
        tree_mask = tree_mask_buf
        if tree_mask_mode == TreeMaskMode.QLEN_ONLY:
            tree_mask.fill_(True)
        elif tree_mask_mode == TreeMaskMode.QLEN_ONLY_BITPACKING:
            tree_mask.fill_(0)
        elif tree_mask_mode == TreeMaskMode.FULL_MASK:
            tree_mask.fill_(True)
        else:
            raise NotImplementedError(f"Invalid tree mask: {tree_mask_mode=}")
    elif tree_mask_mode == TreeMaskMode.QLEN_ONLY:
        # QLEN_ONLY: num_verify_tokens^2 per request
        if per_request_draft_token_num is not None:
            mask_size = (per_request_draft_token_num ** 2).sum().item()
        else:
            mask_size = num_verify_tokens * bs * num_verify_tokens
        tree_mask = torch.full(
            (mask_size,),
            True,
            dtype=torch.bool,
            device=device,
        )
    elif tree_mask_mode == TreeMaskMode.QLEN_ONLY_BITPACKING:
        packed_dtypes = [torch.uint8, torch.uint16, torch.uint32]
        packed_dtype_idx = int(math.ceil(math.log2((num_verify_tokens + 7) // 8)))
        # BITPACKING: num_verify_tokens per request (packed)
        if per_request_draft_token_num is not None:
            mask_size = per_request_draft_token_num.sum().item()
        else:
            mask_size = num_verify_tokens * bs
        tree_mask = torch.zeros(
            (mask_size,),
            dtype=packed_dtypes[packed_dtype_idx],
            device=device,
        )
    elif tree_mask_mode == TreeMaskMode.FULL_MASK:
        # FULL_MASK: prompt→draft + draft→draft attention
        if per_request_draft_token_num is not None:
            total_verify_tokens = per_request_draft_token_num.sum().item()
            draft_draft_size = (per_request_draft_token_num ** 2).sum().item()
            mask_size = seq_lens_sum * total_verify_tokens + draft_draft_size
        else:
            mask_size = (
                seq_lens_sum * num_verify_tokens
                + num_verify_tokens * num_verify_tokens * bs
            )
        tree_mask = torch.full(
            (mask_size,),
            True,
            device=device,
        )
    else:
        raise NotImplementedError(f"Invalid tree mask: {tree_mask_mode=}")

    # TODO: make them torch.empty and fuse them into `sgl_build_tree_kernel`
    # retrive_buf: per-batch structure [3, bs, max_tokens_per_request]
    # Use max_verify_tokens to accommodate the request with most tokens
    retrive_buf = torch.full(
        (3, bs, max_verify_tokens), -1, device=device, dtype=torch.long
    )
    retrive_index, retrive_next_token, retrive_next_sibling = retrive_buf
    # position: where each token belongs to
    # e.g. if depth of each draft token is [0, 1, 1, 2] and the prompt length is 7
    # then, positions = [7, 8, 8, 9]
    # positions: flattened globally [total_tokens_across_all_requests]
    # Use total_verify_tokens (sum of per-request counts)
    if position_buf is not None:
        positions = position_buf
    else:
        positions = torch.empty(
            (total_verify_tokens,), device=device, dtype=torch.long
        )

    # TileSpec ragged case: loop over requests and call kernel with uniform count each time
    if per_request_draft_token_num is not None:
        # Calculate offsets into flattened buffers
        draft_offsets = torch.cat([
            torch.zeros(1, device=device, dtype=torch.long),
            per_request_draft_token_num.cumsum(0)
        ])

        # Calculate tree_mask offsets (depends on tree_mask_mode)
        if tree_mask_mode == TreeMaskMode.QLEN_ONLY:
            # mask_size per request = count^2
            mask_offsets = torch.cat([
                torch.zeros(1, device=device, dtype=torch.long),
                (per_request_draft_token_num ** 2).cumsum(0)
            ])
        elif tree_mask_mode == TreeMaskMode.QLEN_ONLY_BITPACKING:
            # mask_size per request = count
            mask_offsets = draft_offsets.clone()
        elif tree_mask_mode == TreeMaskMode.FULL_MASK:
            # mask_size per request = seq_len * count + count^2
            mask_sizes = seq_lens * per_request_draft_token_num + per_request_draft_token_num ** 2
            mask_offsets = torch.cat([
                torch.zeros(1, device=device, dtype=torch.long),
                mask_sizes.cumsum(0)
            ])
        else:
            raise NotImplementedError(f"Invalid tree mask mode: {tree_mask_mode}")

        # Convert to CPU lists once to avoid repeated GPU syncs in the loop
        counts_cpu = per_request_draft_token_num.tolist()
        draft_offsets_cpu = draft_offsets.tolist()
        mask_offsets_cpu = mask_offsets.tolist()

        # Loop over each request and call kernel
        for req_idx in range(bs):
            req_count = counts_cpu[req_idx]

            # Slice flattened draft data
            draft_start = draft_offsets_cpu[req_idx]
            draft_end = draft_offsets_cpu[req_idx + 1]

            # top_scores_index doesn't include verified token, so offset by 1 per request
            # In flattened layout: [v0, d0_0, ..., v1, d1_0, ...]
            # top_scores_index refers to draft tokens only: [d0_0, ..., d1_0, ...]
            scores_start = draft_start - req_idx if req_idx > 0 else 0
            scores_end = draft_end - (req_idx + 1)
            req_top_scores = top_scores_index[scores_start:scores_end] if scores_end > scores_start else torch.empty(0, dtype=torch.long, device=device)

            # Slice output buffers
            req_positions = positions[draft_start:draft_end]
            mask_start = mask_offsets_cpu[req_idx]
            mask_end = mask_offsets_cpu[req_idx + 1]
            req_tree_mask = tree_mask[mask_start:mask_end]

            # Call kernel with bs=1 and uniform count
            if _is_npu:
                torch.ops.npu.build_tree_kernel_efficient(
                    parent_list[req_idx:req_idx+1].to(dtype=torch.int64),
                    req_top_scores,
                    seq_lens[req_idx:req_idx+1],
                    req_tree_mask,
                    req_positions,
                    retrive_index[req_idx:req_idx+1, :req_count],
                    retrive_next_token[req_idx:req_idx+1, :req_count],
                    retrive_next_sibling[req_idx:req_idx+1, :req_count],
                    topk,
                    spec_steps,
                    req_count,
                    tree_mask_mode,
                )
            else:
                sgl_build_tree_kernel_efficient(
                    parent_list[req_idx:req_idx+1],
                    req_top_scores,
                    seq_lens[req_idx:req_idx+1],
                    req_tree_mask,
                    req_positions,
                    retrive_index[req_idx:req_idx+1, :req_count],
                    retrive_next_token[req_idx:req_idx+1, :req_count],
                    retrive_next_sibling[req_idx:req_idx+1, :req_count],
                    topk,
                    spec_steps,
                    req_count,
                    tree_mask_mode,
                )

            # TileSpec fix: kernel computes local indices (bid=0), convert to PADDED global
            # retrive_index values are used by verify_tree_greedy to access padded arrays
            # (candidates, target_predict are padded to [bs, max_verify_tokens])
            # Use padded offset (req_idx * max_verify_tokens), NOT ragged offset (draft_offsets)
            padded_offset = req_idx * max_verify_tokens
            if padded_offset > 0:
                retrive_index[req_idx, :req_count] += padded_offset
    else:
        # Uniform case: call kernel once with batched inputs (original behavior)
        if _is_npu:
            torch.ops.npu.build_tree_kernel_efficient(
                parent_list.to(dtype=torch.int64),
                top_scores_index,
                seq_lens,
                tree_mask,
                positions,
                retrive_index,
                retrive_next_token,
                retrive_next_sibling,
                topk,
                spec_steps,
                num_verify_tokens,
                tree_mask_mode,
            )
        else:
            sgl_build_tree_kernel_efficient(
                parent_list,
                top_scores_index,
                seq_lens,
                tree_mask,
                positions,
                retrive_index,
                retrive_next_token,
                retrive_next_sibling,
                topk,
                spec_steps,
                num_verify_tokens,
                tree_mask_mode,
            )
    return (
        tree_mask,
        positions,
        retrive_index,
        retrive_next_token,
        retrive_next_sibling,
        draft_tokens,
    )


def verify_tree_greedy_func(
    predicts: torch.Tensor,
    accept_index: torch.Tensor,
    accept_token_num: torch.Tensor,
    candidates: torch.Tensor,
    retrive_index: torch.Tensor,
    retrive_next_token: torch.Tensor,
    retrive_next_sibling: torch.Tensor,
    target_predict: torch.Tensor,
    topk: int = -1,
):
    if _is_cuda or _is_hip:
        from sgl_kernel import verify_tree_greedy

        verify_tree_greedy(
            predicts=predicts,  # mutable
            accept_index=accept_index,  # mutable
            accept_token_num=accept_token_num,  # mutable
            candidates=candidates,
            retrive_index=retrive_index,
            retrive_next_token=retrive_next_token,
            retrive_next_sibling=retrive_next_sibling,
            target_predict=target_predict,
        )

    elif _is_npu:
        from sgl_kernel_npu.sample.verify_tree_greedy import verify_tree_greedy

        verify_tree_greedy(
            predicts=predicts,
            accept_index=accept_index,
            accept_token_num=accept_token_num,
            candidates=candidates,
            retrive_index=retrive_index,
            retrive_next_token=retrive_next_token,
            retrive_next_sibling=retrive_next_sibling,
            target_predict=target_predict,
        )
    return predicts, accept_index, accept_token_num
