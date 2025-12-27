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
    # TileSpec ragged kernel - from tile_spec_kernels package
    # Install with: cd tile_spec/kernels && pip install . --no-build-isolation
    from tile_spec_kernels import build_tree_kernel_efficient_ragged as sgl_build_tree_kernel_ragged


import time
import os

_TILESPEC_DEBUG = os.environ.get("TILESPEC_DEBUG", "0") == "1"

def organize_draft_results(
    score_list: List[torch.Tensor],
    token_list: List[torch.Tensor],
    parents_list: List[torch.Tensor],
    num_draft_token: int,
    is_tilespec_profiling: bool = False,
    use_ragged_path: bool = False,
):
    """
    Organize draft results and select tokens for verification.

    Args:
        score_list: List of score tensors from each draft step
        token_list: List of token tensors from each draft step
        parents_list: List of parent tensors from each draft step
        num_draft_token: Maximum draft tokens (always int)
        is_tilespec_profiling: Whether TileSpec is in profiling mode
        use_ragged_path: Force ragged path with uniform counts (for overhead measurement)

    Returns (uniform path - 3 values):
        parent_list: Parent indices [bs, n_parents]
        top_scores_index: Selected token indices [bs, num_draft_token-1]
        draft_tokens: Selected tokens [bs, num_draft_token-1]

    Returns (ragged path - 6 values):
        parent_list: Parent indices [bs, n_parents]
        top_scores_index: Selected token indices [total_drafts] (1D flattened)
        draft_tokens: Selected tokens [total_drafts] (1D flattened)
        per_request_draft_token_num: Per-request counts including verified token [bs]
        sorted_request_ids: Request ID for each draft token [total_drafts]
        selected_scores: Scores for calibration [total_drafts] or None

    Note:
        Ragged path returns 1D flattened tensors + sorted_request_ids for near-zero
        overhead indexing (avoid repeat_interleave).
    """
    scores = torch.cat(score_list, dim=1).flatten(1)
    tokens = torch.cat(token_list, dim=1)
    bs, n_cand = scores.shape
    device = scores.device

    # Check if TileSpec/ragged path is active
    tilespec_active = is_tilespec_profiling or use_ragged_path

    if tilespec_active:
        if _TILESPEC_DEBUG:
            torch.cuda.synchronize()
            _t0 = time.perf_counter()

        # Step 1: Global selection - select best tokens across all requests
        # Total tokens = bs * (num_draft_token - 1), same as uniform approach
        total_drafts = bs * (num_draft_token - 1)

        if total_drafts > 0:
            # Use topk instead of full sort - O(n log k) vs O(n log n)
            flat_scores = scores.flatten()
            _, top_global_indices = torch.topk(flat_scores, total_drafts, largest=True, sorted=False)

            # Convert global indices to (request_id, col_index)
            request_ids = top_global_indices // n_cand
            col_indices = top_global_indices % n_cand

            # Compute per-request counts from actual selection
            draft_counts = torch.bincount(request_ids, minlength=bs).to(torch.long)
            per_request_draft_token_num = draft_counts + 1  # +1 for verified token

            # Sort by (request_id, col_index) to group by request and order within each
            composite_key = request_ids * n_cand + col_indices
            sort_order = torch.argsort(composite_key)

            sorted_request_ids = request_ids[sort_order]
            sorted_col_indices = col_indices[sort_order]

            # Return 1D flattened tensors (no padding!)
            # top_scores_index: column indices for each selected draft token
            top_scores_index = sorted_col_indices  # [total_drafts]
            # draft_tokens: actual token values
            draft_tokens = tokens[sorted_request_ids, sorted_col_indices]  # [total_drafts]

            # Only compute selected_scores during profiling (for calibration data)
            if is_tilespec_profiling:
                selected_scores = scores[sorted_request_ids, sorted_col_indices]  # [total_drafts]
            else:
                selected_scores = None
        else:
            per_request_draft_token_num = torch.ones(bs, dtype=torch.long, device=device)  # Just verified token
            top_scores_index = torch.empty(0, dtype=torch.long, device=device)
            draft_tokens = torch.empty(0, dtype=tokens.dtype, device=device)
            sorted_request_ids = torch.empty(0, dtype=torch.long, device=device)
            selected_scores = None

        # Build parent_list for TileSpec case
        if len(parents_list) > 1:
            parent_list = torch.cat(parents_list[:-1], dim=1)
        else:
            parent_list = torch.empty(bs, 0, device=device)

        if _TILESPEC_DEBUG:
            torch.cuda.synchronize()
            _t1 = time.perf_counter()
            print(f"[TileSpec] organize_draft_results ragged: {(_t1-_t0)*1000:.3f}ms (bs={bs})")

        return parent_list, top_scores_index, draft_tokens, per_request_draft_token_num, sorted_request_ids, selected_scores

    else:
        # Original EAGLE: uniform selection (returns 3 values for CUDA graph compatibility)
        if _TILESPEC_DEBUG:
            torch.cuda.synchronize()
            _t0 = time.perf_counter()

        top_scores = torch.topk(scores, num_draft_token - 1, dim=-1)
        top_scores_index = torch.sort(top_scores.indices).values
        draft_tokens = torch.gather(tokens, index=top_scores_index, dim=1)

        # Build parent_list
        if len(parents_list) > 1:
            parent_list = torch.cat(parents_list[:-1], dim=1)
        else:
            parent_list = torch.empty(bs, 0, device=device)

        if _TILESPEC_DEBUG:
            torch.cuda.synchronize()
            _t1 = time.perf_counter()
            print(f"[EAGLE] organize_draft_results uniform: {(_t1-_t0)*1000:.3f}ms (bs={bs})")

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
    sorted_request_ids: Optional[torch.Tensor] = None,
):
    # Handle uniform (2D Tensor) vs ragged (1D Tensor with sorted_request_ids)
    if per_request_draft_token_num is not None and sorted_request_ids is not None:
        # Ragged case: 1D flattened draft_tokens with sorted_request_ids
        # Build draft_tokens_with_verified by interleaving verified tokens
        # Layout: [v0, d0_0, d0_1, ..., v1, d1_0, d1_1, ..., v2, ...]
        device = draft_tokens.device
        bs = verified_id.shape[0]
        total_tokens = per_request_draft_token_num.sum().item()

        # Compute offsets for output positions
        token_offsets = torch.cat([
            torch.zeros(1, device=device, dtype=torch.long),
            per_request_draft_token_num.cumsum(0)[:-1]
        ])  # [bs]
        draft_offsets = torch.cat([
            torch.zeros(1, device=device, dtype=torch.long),
            (per_request_draft_token_num - 1).cumsum(0)[:-1]
        ])  # [bs] - offsets into draft_tokens (without verified)

        # Build output tensor (use torch.long as required by verify_tree_greedy kernel)
        draft_tokens_with_verified = torch.empty(total_tokens, dtype=torch.long, device=device)

        # Place verified tokens at start of each request's segment
        draft_tokens_with_verified[token_offsets] = verified_id.to(torch.long)

        # Place draft tokens using sorted_request_ids indexing (near-zero overhead!)
        if draft_tokens.numel() > 0:
            # Output positions: token_offsets[req_id] + 1 + local_position
            # local_position = global_draft_idx - draft_offsets[req_id]
            global_draft_idx = torch.arange(draft_tokens.numel(), device=device)
            local_pos = global_draft_idx - draft_offsets[sorted_request_ids]  # Indexing! Not repeat_interleave
            output_pos = token_offsets[sorted_request_ids] + 1 + local_pos
            draft_tokens_with_verified[output_pos] = draft_tokens.to(torch.long)

        draft_tokens = draft_tokens_with_verified
    else:
        # Uniform case: original logic (2D tensors)
        draft_tokens = torch.cat((verified_id.unsqueeze(1), draft_tokens), dim=1).flatten()

    # seq_lens_sum == sum(seq_lens); seq_lens: sequence length without draft tokens
    bs = seq_lens.numel()
    device = seq_lens.device

    # Calculate actual verify token counts (uniform or per-request)
    # Compute all needed values once to avoid redundant GPU→CPU syncs
    if per_request_draft_token_num is not None:
        total_verify_tokens = per_request_draft_token_num.sum().item()  # Sum for flattened buffers
        max_verify_tokens = per_request_draft_token_num.max().item()    # Max for per-batch buffers
        squared_sum = (per_request_draft_token_num ** 2).sum().item()   # For mask size calculations
    else:
        total_verify_tokens = bs * num_verify_tokens
        max_verify_tokens = num_verify_tokens
        squared_sum = num_verify_tokens * num_verify_tokens * bs
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
        mask_size = squared_sum  # Use pre-computed value
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
        mask_size = total_verify_tokens  # Use pre-computed value
        tree_mask = torch.zeros(
            (mask_size,),
            dtype=packed_dtypes[packed_dtype_idx],
            device=device,
        )
    elif tree_mask_mode == TreeMaskMode.FULL_MASK:
        # FULL_MASK: prompt→draft + draft→draft attention
        if per_request_draft_token_num is not None:
            # Use pre-computed values from above (avoid redundant .item() calls)
            mask_size = seq_lens_sum * total_verify_tokens + squared_sum
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

    # TileSpec ragged case: call ragged kernel once (no loop!)
    if per_request_draft_token_num is not None and sorted_request_ids is not None:
        if _TILESPEC_DEBUG:
            torch.cuda.synchronize()
            _t0 = time.perf_counter()

        # Calculate indptr arrays for ragged kernel
        # token_indptr: cumsum of per_request_draft_token_num (includes verified)
        token_indptr = torch.cat([
            torch.zeros(1, device=device, dtype=torch.long),
            per_request_draft_token_num.cumsum(0)
        ])

        # score_indptr: cumsum of (per_request_draft_token_num - 1) (excludes verified)
        draft_counts = per_request_draft_token_num - 1
        score_indptr = torch.cat([
            torch.zeros(1, device=device, dtype=torch.long),
            draft_counts.cumsum(0)
        ])

        # mask_indptr: depends on tree_mask_mode
        if tree_mask_mode == TreeMaskMode.QLEN_ONLY:
            # mask_size per request = count^2
            mask_indptr = torch.cat([
                torch.zeros(1, device=device, dtype=torch.long),
                (per_request_draft_token_num ** 2).cumsum(0)
            ])
        elif tree_mask_mode == TreeMaskMode.QLEN_ONLY_BITPACKING:
            # mask_size per request = count
            mask_indptr = token_indptr.clone()
        elif tree_mask_mode == TreeMaskMode.FULL_MASK:
            # mask_size per request = seq_len * count + count^2
            mask_sizes = seq_lens * per_request_draft_token_num + per_request_draft_token_num ** 2
            mask_indptr = torch.cat([
                torch.zeros(1, device=device, dtype=torch.long),
                mask_sizes.cumsum(0)
            ])
        else:
            raise NotImplementedError(f"Invalid tree mask mode: {tree_mask_mode}")

        if _is_npu:
            raise AssertionError("TileSpec ragged mode is not supported on NPU")

        # CUDA: use optimized ragged kernel (single kernel call, no loop!)
        # The kernel computes global padded indices directly (no post-processing needed)
        sgl_build_tree_kernel_ragged(
            parent_list,
            top_scores_index,
            seq_lens,
            tree_mask,
            positions,
            retrive_index,
            retrive_next_token,
            retrive_next_sibling,
            token_indptr,
            score_indptr,
            mask_indptr,
            topk,
            spec_steps,
            max_verify_tokens,
            tree_mask_mode,
        )

        if _TILESPEC_DEBUG:
            torch.cuda.synchronize()
            _t1 = time.perf_counter()
            print(f"[TileSpec] build_tree_kernel_efficient ragged: {(_t1-_t0)*1000:.3f}ms (bs={bs})")
    else:
        # Uniform case: call kernel once with batched inputs (original behavior)
        if _TILESPEC_DEBUG:
            torch.cuda.synchronize()
            _t0 = time.perf_counter()

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

        if _TILESPEC_DEBUG:
            torch.cuda.synchronize()
            _t1 = time.perf_counter()
            print(f"[EAGLE] build_tree_kernel_efficient uniform: {(_t1-_t0)*1000:.3f}ms (bs={bs})")

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
