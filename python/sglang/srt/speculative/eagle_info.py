import logging
import os
import time
from copy import copy
from dataclasses import dataclass
from typing import ClassVar, List, Optional, Tuple

import torch
import torch.nn.functional as F

_TILESPEC_DEBUG = os.environ.get("TILESPEC_DEBUG", "0") == "1"

from sglang.srt.constrained.base_grammar_backend import BaseGrammarObject
from sglang.srt.environ import envs
from sglang.srt.layers.attention.utils import create_flashinfer_kv_indices_triton
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.layers.sampler import apply_custom_logit_processor
from sglang.srt.managers.overlap_utils import FutureIndices
from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.mem_cache.allocator import BaseTokenToKVPoolAllocator
from sglang.srt.mem_cache.common import (
    alloc_paged_token_slots_extend,
    alloc_token_slots,
    get_last_loc,
)
from sglang.srt.model_executor.forward_batch_info import CaptureHiddenMode
from sglang.srt.server_args import get_global_server_args
from sglang.srt.speculative.eagle_info_v2 import (
    EagleDraftInputV2Mixin,
    EagleVerifyInputV2Mixin,
)
from sglang.srt.speculative.eagle_utils import (
    verify_tree_greedy_func,
    sgl_verify_tree_greedy_ragged,
)
from sglang.srt.speculative.spec_info import SpecInput, SpecInputType
from sglang.srt.speculative.spec_utils import (
    SIMULATE_ACC_LEN,
    TREE_SPEC_KERNEL_AVAILABLE,
    align_evict_mask_to_page_size,
    assign_req_to_token_pool_func,
    create_accept_length_filter,
    create_extend_after_decode_spec_info,
    filter_finished_cache_loc_kernel,
    generate_simulated_accept_index,
    get_src_tgt_cache_loc,
    get_target_cache_loc,
)
from sglang.srt.utils import is_cuda, next_power_of_2

if is_cuda():
    from sgl_kernel import (
        top_k_renorm_prob,
        top_p_renorm_prob,
        tree_speculative_sampling_target_only,
    )

logger = logging.getLogger(__name__)


def _compute_pad_indices(per_request_counts, total_tokens, device):
    """
    Compute row and column indices for padding ragged tensors.
    These indices can be reused across multiple _pad_to_2d_with_indices calls.

    Args:
        per_request_counts: Tensor [bs] with actual counts per request
        total_tokens: Total number of tokens across all requests
        device: Device to create tensors on

    Returns:
        Tuple of (row_indices, col_indices)
    """
    # Compute cumulative offsets: [count0, count0+count1, ...]
    cumsum = per_request_counts.cumsum(0)

    # Row indices using bucketize (faster than repeat_interleave)
    # bucketize with right=True: for index i, find smallest j where cumsum[j] > i
    token_indices = torch.arange(total_tokens, device=device)
    row_indices = torch.bucketize(token_indices, cumsum, right=True)

    # Column indices using indexing (faster than repeat_interleave)
    offsets = torch.cat([torch.zeros(1, device=device, dtype=torch.long), cumsum[:-1]])
    col_indices = token_indices - offsets[row_indices]

    return row_indices, col_indices


def _pad_to_2d_with_indices(flat_tensor, row_indices, col_indices, bs, max_count, pad_value=-1):
    """
    Pad ragged flattened tensor using pre-computed indices.
    Faster than _pad_to_2d when padding multiple tensors with same layout.

    Args:
        flat_tensor: Flattened ragged tensor [total_tokens] or [total_tokens, ...]
        row_indices: Pre-computed row indices from _compute_pad_indices
        col_indices: Pre-computed column indices from _compute_pad_indices
        bs: Batch size
        max_count: Maximum count to pad to
        pad_value: Value to use for padding

    Returns:
        Padded tensor [bs, max_count] or [bs, max_count, ...]
    """
    device = flat_tensor.device
    dtype = flat_tensor.dtype

    # Determine output shape based on input dimensions
    if flat_tensor.ndim == 1:
        output_shape = (bs, max_count)
    elif flat_tensor.ndim == 2:
        output_shape = (bs, max_count, flat_tensor.shape[1])
    else:
        raise ValueError(f"Unsupported tensor dimension: {flat_tensor.ndim}")

    # Create padded tensor filled with pad_value
    padded = torch.full(output_shape, pad_value, dtype=dtype, device=device)

    # Scatter into padded tensor using pre-computed indices
    if flat_tensor.ndim == 1:
        padded[row_indices, col_indices] = flat_tensor
    else:  # ndim == 2
        padded[row_indices, col_indices, :] = flat_tensor

    return padded


def _pad_to_2d(flat_tensor, per_request_counts, max_count, pad_value=-1):
    """
    Pad ragged flattened tensor to uniform 2D [bs, max_count, ...].

    Args:
        flat_tensor: Flattened ragged tensor [total_tokens] or [total_tokens, ...]
        per_request_counts: Tensor [bs] with actual counts per request
        max_count: Maximum count to pad to
        pad_value: Value to use for padding (default: -1 for token IDs)

    Returns:
        Padded tensor [bs, max_count] or [bs, max_count, ...]

    Example:
        flat_tensor = [V0, t01, t02, V1, t11, t12, t13, t14, V2, t21]
        per_request_counts = [3, 5, 2]
        max_count = 5

        Returns:
        [[V0, t01, t02, -1, -1],
         [V1, t11, t12, t13, t14],
         [V2, t21, -1, -1, -1]]
    """
    bs = per_request_counts.shape[0]
    total_tokens = flat_tensor.shape[0]
    row_indices, col_indices = _compute_pad_indices(per_request_counts, total_tokens, flat_tensor.device)
    return _pad_to_2d_with_indices(flat_tensor, row_indices, col_indices, bs, max_count, pad_value)


@dataclass
class EagleVerifyInput(SpecInput, EagleVerifyInputV2Mixin):
    draft_token: torch.Tensor
    custom_mask: torch.Tensor
    positions: torch.Tensor
    retrive_index: torch.Tensor
    retrive_next_token: torch.Tensor
    retrive_next_sibling: torch.Tensor
    retrive_cum_len: torch.Tensor
    spec_steps: int
    topk: int
    draft_token_num: int
    capture_hidden_mode: CaptureHiddenMode
    seq_lens_sum: int
    seq_lens_cpu: torch.Tensor
    grammar: BaseGrammarObject = None

    # Per-request draft token counts (TileSpec support)
    # None = uniform (all requests have draft_token_num drafts)
    # Tensor [bs] = per-request counts (can vary)
    per_request_draft_token_num: Optional[torch.Tensor] = None

    # Selected scores for TileSpec calibration
    # Flattened tensor [total_tokens] of cumulative path scores
    selected_scores: Optional[torch.Tensor] = None

    # Shape info for padding
    num_tokens_per_batch: int = -1

    # Cached values for TileSpec ragged case (computed once in prepare_for_verify)
    # Avoids redundant GPU syncs across methods
    _cached_max_count: Optional[int] = None
    _cached_total_tokens: Optional[int] = None
    _cached_squared_sum: Optional[int] = None

    # Pre-computed indptr array for ragged kernels (from organize_draft_results)
    token_indptr: Optional[torch.Tensor] = None

    def __post_init__(self):
        super().__init__(SpecInputType.EAGLE_VERIFY)

    def get_spec_adjust_token_coefficient(self) -> Tuple[int, int]:
        return self.draft_token_num, self.draft_token_num

    @classmethod
    def create_idle_input(cls, topk: int, spec_steps: int, num_verify_tokens: int):
        return cls(
            draft_token=torch.empty((0,), dtype=torch.long, device="cuda"),
            custom_mask=torch.full((0,), True, dtype=torch.bool, device="cuda"),
            positions=torch.empty((0,), dtype=torch.int64, device="cuda"),
            retrive_index=torch.full(
                (0, num_verify_tokens), -1, dtype=torch.long, device="cuda"
            ),
            retrive_next_token=torch.full(
                (0, num_verify_tokens), -1, dtype=torch.long, device="cuda"
            ),
            retrive_next_sibling=torch.full(
                (0, num_verify_tokens), -1, dtype=torch.long, device="cuda"
            ),
            retrive_cum_len=None,
            topk=topk,
            draft_token_num=num_verify_tokens,
            spec_steps=spec_steps,
            capture_hidden_mode=CaptureHiddenMode.FULL,
            seq_lens_sum=0,
            seq_lens_cpu=torch.empty((0,), dtype=torch.int32),
        )

    def prepare_for_verify(self, batch: ScheduleBatch, page_size: int):

        if batch.forward_mode.is_idle():
            return

        if _TILESPEC_DEBUG:
            torch.cuda.synchronize()
            _t0 = time.perf_counter()

        batch.input_ids = self.draft_token

        # For TileSpec: use per-request draft counts
        # For uniform: use scalar draft_token_num (broadcasts to all requests)
        if self.per_request_draft_token_num is not None:
            draft_token_counts = self.per_request_draft_token_num
            # Batch all GPU->CPU transfers in one sync by using .tolist()
            counts_list = draft_token_counts.tolist()
            draft_token_counts_cpu = torch.tensor(counts_list, dtype=torch.int64)
            # Cache computed values from CPU list (no additional GPU sync!)
            self._cached_max_count = max(counts_list)
            self._cached_total_tokens = sum(counts_list)
            self._cached_squared_sum = sum(c * c for c in counts_list)
        else:
            draft_token_counts = self.draft_token_num
            draft_token_counts_cpu = self.draft_token_num

        if page_size == 1:
            batch.out_cache_loc = alloc_token_slots(
                batch.tree_cache,
                len(batch.input_ids),  # Exact size (ragged sum or uniform total)
            )
            end_offset = batch.seq_lens + draft_token_counts
        else:
            prefix_lens = batch.seq_lens
            prefix_lens_cpu = batch.seq_lens_cpu
            end_offset = prefix_lens + draft_token_counts
            end_offset_cpu = prefix_lens_cpu + draft_token_counts_cpu
            last_loc = get_last_loc(
                batch.req_to_token_pool.req_to_token,
                batch.req_pool_indices,
                prefix_lens,
            )
            batch.out_cache_loc = alloc_paged_token_slots_extend(
                batch.tree_cache,
                prefix_lens,
                prefix_lens_cpu,
                end_offset,
                end_offset_cpu,
                last_loc,
                len(batch.input_ids),  # Exact size
            )
            self.last_loc = last_loc

        bs = batch.batch_size()
        assign_req_to_token_pool_func(
            batch.req_pool_indices,
            batch.req_to_token_pool.req_to_token,
            batch.seq_lens,
            end_offset,
            batch.out_cache_loc,
            bs,
        )

        if get_global_server_args().enable_mamba_extra_buffer():
            batch.mamba_track_indices = torch.tensor(
                [
                    req.mamba_ping_pong_track_buffer[req.mamba_next_track_idx]
                    for req in batch.reqs
                ],
                dtype=torch.int64,
                device=batch.device,
            )

        if _TILESPEC_DEBUG:
            torch.cuda.synchronize()
            _t1 = time.perf_counter()
            path = "ragged" if self.per_request_draft_token_num is not None else "uniform"
            print(f"[{'TileSpec' if self.per_request_draft_token_num is not None else 'EAGLE'}] prepare_for_verify {path}: {(_t1-_t0)*1000:.3f}ms")

    def generate_attn_arg_prefill(
        self,
        req_pool_indices: torch.Tensor,
        paged_kernel_lens: torch.Tensor,
        paged_kernel_lens_sum: int,
        req_to_token: torch.Tensor,
    ):
        if _TILESPEC_DEBUG:
            torch.cuda.synchronize()
            _t0 = time.perf_counter()

        device = req_pool_indices.device
        batch_size = len(req_pool_indices)

        # Handle both uniform and per-request draft token counts
        if self.per_request_draft_token_num is not None:
            # Ragged case: use actual per-request counts
            draft_counts = self.per_request_draft_token_num
            total_draft_tokens = self._cached_total_tokens  # Use cached value (no GPU sync!)
            # qo_indptr: cumulative offsets for each request
            qo_indptr = torch.cat([
                torch.zeros(1, dtype=torch.int32, device=device),
                draft_counts.cumsum(0).to(torch.int32)
            ])
        else:
            # Uniform case: all requests have same count
            draft_counts = self.draft_token_num
            total_draft_tokens = self.draft_token_num * batch_size
            qo_indptr = torch.arange(
                0,
                (1 + batch_size) * self.draft_token_num,
                step=self.draft_token_num,
                dtype=torch.int32,
                device=device,
            )

        cum_kv_seq_len = torch.zeros(
            (batch_size + 1,), dtype=torch.int32, device=device
        )

        paged_kernel_lens = paged_kernel_lens + draft_counts
        cum_kv_seq_len[1:] = torch.cumsum(paged_kernel_lens, dim=0)

        kv_indices = torch.empty(
            paged_kernel_lens_sum + total_draft_tokens,
            dtype=torch.int32,
            device=device,
        )
        create_flashinfer_kv_indices_triton[(batch_size,)](
            req_to_token,
            req_pool_indices,
            paged_kernel_lens,
            cum_kv_seq_len,
            None,
            kv_indices,
            req_to_token.size(1),
        )
        # Calculate custom mask size (handles both uniform and per-request draft counts)
        if self.per_request_draft_token_num is not None:
            # Per-request: use cached values (computed in prepare_for_verify)
            mask_numel = paged_kernel_lens_sum * self._cached_total_tokens + self._cached_squared_sum
        else:
            # Uniform counts: original formula
            mask_numel = (
                paged_kernel_lens_sum * self.draft_token_num
                + (self.draft_token_num**2) * batch_size
            )
        if self.custom_mask.numel() < mask_numel:
            # FIXME(attn): temporary fix for custom mask padding with cuda graph
            self.custom_mask = torch.cat(
                [
                    self.custom_mask,
                    torch.full(
                        (mask_numel - self.custom_mask.numel(),),
                        True,
                        dtype=torch.bool,
                        device=device,
                    ),
                ],
                dim=0,
            )

        if _TILESPEC_DEBUG:
            torch.cuda.synchronize()
            _t1 = time.perf_counter()
            path = "ragged" if self.per_request_draft_token_num is not None else "uniform"
            print(f"[{'TileSpec' if self.per_request_draft_token_num is not None else 'EAGLE'}] generate_attn_arg_prefill {path}: {(_t1-_t0)*1000:.3f}ms")

        return kv_indices, cum_kv_seq_len, qo_indptr, self.custom_mask

    def verify(
        self,
        batch: ScheduleBatch,
        logits_output: LogitsProcessorOutput,
        token_to_kv_pool_allocator: BaseTokenToKVPoolAllocator,
        page_size: int,
        vocab_mask: Optional[torch.Tensor] = None,  # For grammar
    ) -> torch.Tensor:
        """
        Verify and find accepted tokens based on logits output and batch
        (which contains spec decoding information).

        WARNING: This API in-place modifies the states of logits_output

        This API updates values inside logits_output based on the accepted
        tokens. I.e., logits_output.next_token_logits only contains
        accepted token logits.
        """
        if batch.forward_mode.is_idle():
            return EagleVerifyOutput(
                draft_input=EagleDraftInput.create_idle_input(
                    device=batch.device,
                    hidden_size=batch.model_config.hidden_size,
                    dtype=batch.model_config.dtype,
                    topk=self.topk,
                    capture_hidden_mode=CaptureHiddenMode.LAST,
                ),
                logits_output=logits_output,
                verified_id=torch.empty(0, dtype=torch.long, device=batch.device),
                accept_length_per_req_cpu=[],
                accepted_indices=torch.full(
                    (0, self.spec_steps + 1),
                    -1,
                    dtype=torch.int32,
                    device=batch.device,
                ),
            )

        bs = self.retrive_index.shape[0]

        # Compute max_count for TileSpec ragged case
        # Padding indices are computed lazily only when needed (non-greedy path)
        if _TILESPEC_DEBUG and self.per_request_draft_token_num is not None:
            torch.cuda.synchronize()
            _t_verify_start = time.perf_counter()

        if self.per_request_draft_token_num is not None:
            max_count = self._cached_max_count  # Use cached value from prepare_for_verify
        else:
            max_count = self.draft_token_num
            if _TILESPEC_DEBUG:
                torch.cuda.synchronize()
                _t_verify_start = time.perf_counter()

        # Handle both uniform and per-request draft tokens
        if self.per_request_draft_token_num is not None:
            # Ragged case: keep as 1D (no padding needed for ragged verify kernel!)
            candidates = self.draft_token  # [total_tokens] - already 1D ragged
            # Use pre-computed token_indptr from organize_draft_results
            token_indptr = self.token_indptr
        else:
            # Uniform case: simple reshape
            candidates = self.draft_token.reshape(bs, self.draft_token_num)
            token_indptr = None

        sampling_info = batch.sampling_info

        # predict must match the layout of target_predict (ragged or uniform)
        # The CUDA kernel writes predicts[idx] using same indices as target_predict[idx]
        if self.per_request_draft_token_num is not None:
            # Ragged case: [total_tokens] to match ragged target_predict layout
            predict_size = self.draft_token.shape[0]
        else:
            # Uniform case: [bs * draft_token_num]
            predict_size = bs * self.draft_token_num
        predict = torch.empty(predict_size, dtype=torch.int32, device=batch.device)
        accept_index = torch.full(
            (bs, self.spec_steps + 1), -1, dtype=torch.int32, device=batch.device
        )
        accept_length = torch.empty((bs,), dtype=torch.int32, device=batch.device)

        if bs != len(sampling_info):
            sampling_info = copy.deepcopy(sampling_info)
            # NOTE: retrive_index are the indices of the requests that are kept.
            sampling_info.filter_batch(self.retrive_index.tolist(), self.retrive_index)

        # Determine tokens per request for penalties and processors
        if self.per_request_draft_token_num is not None:
            # Per-request: use actual counts
            tokens_per_request = self.per_request_draft_token_num
            num_tokens_in_batch = max_count  # Reuse already computed value
        else:
            # Uniform: use constant draft_token_num
            tokens_per_request = self.draft_token_num
            num_tokens_in_batch = self.draft_token_num

        # Apply the custom logit processors if registered in the sampling info.
        if sampling_info.has_custom_logit_processor:
            apply_custom_logit_processor(
                logits_output.next_token_logits,
                sampling_info,
                num_tokens_in_batch=num_tokens_in_batch,
            )

        # Apply penalty
        if (
            sampling_info.penalizer_orchestrator.is_required
            or sampling_info.logit_bias is not None
        ):
            # TileSpec greedy path doesn't support penalties yet
            assert self.per_request_draft_token_num is None, (
                "TileSpec doesn't support penalties/logit_bias yet. "
                "Disable repetition_penalty and logit_bias when using --tile-spec."
            )
            # This is a relaxed version of penalties for speculative decoding.
            linear_penalty = torch.zeros(
                (bs, logits_output.next_token_logits.shape[1]),
                dtype=torch.float32,
                device=batch.device,
            )
            sampling_info.apply_logits_bias(linear_penalty)
            logits_output.next_token_logits.add_(
                torch.repeat_interleave(linear_penalty, tokens_per_request, dim=0)
            )

        # Apply grammar mask
        if vocab_mask is not None:
            assert self.grammar is not None
            self.grammar.apply_vocab_mask(
                logits=logits_output.next_token_logits, vocab_mask=vocab_mask
            )

        # Sample tokens. Force greedy sampling on AMD
        is_all_greedy = sampling_info.is_all_greedy
        if (not is_all_greedy) and (not TREE_SPEC_KERNEL_AVAILABLE):
            logger.warning(
                "Tree speculative sampling kernel unavailable (likely AMD/HIP build). "
                "Falling back to greedy verification."
            )

        if is_all_greedy or not TREE_SPEC_KERNEL_AVAILABLE:
            target_predict = torch.argmax(logits_output.next_token_logits, dim=-1)

            # Handle both uniform and per-request draft tokens
            if self.per_request_draft_token_num is not None:
                # Ragged case: keep as 1D (no padding needed for ragged verify kernel!)
                # target_predict is already [total_tokens] from argmax
                sgl_verify_tree_greedy_ragged(
                    predicts=predict,  # [total_tokens] - mutable
                    accept_index=accept_index,  # [bs, spec_steps+1] - mutable
                    accept_token_num=accept_length,  # [bs] - mutable
                    candidates=candidates,  # [total_tokens] - 1D ragged
                    retrive_index=self.retrive_index,  # [bs, max_count] - padded, VALUES are ragged
                    retrive_next_token=self.retrive_next_token,  # [bs, max_count] - padded
                    retrive_next_sibling=self.retrive_next_sibling,  # [bs, max_count] - padded
                    target_predict=target_predict,  # [total_tokens] - 1D ragged
                    token_indptr=token_indptr,  # [bs+1]
                    num_speculative_tokens=self.spec_steps + 1,
                )
                # accept_index now has RAGGED indices (matching ragged predict layout)
            else:
                # Uniform case: simple reshape and use original kernel
                target_predict = target_predict.reshape(bs, self.draft_token_num)
                predict, accept_index, accept_length = verify_tree_greedy_func(
                    predicts=predict,  # mutable
                    accept_index=accept_index,  # mutable
                    accept_token_num=accept_length,  # mutable
                    candidates=candidates,
                    retrive_index=self.retrive_index,
                    retrive_next_token=self.retrive_next_token,
                    retrive_next_sibling=self.retrive_next_sibling,
                    target_predict=target_predict,
                    topk=self.topk,
                )

        else:
            # Non-greedy sampling path
            # TileSpec ragged path not yet optimized for non-greedy sampling
            assert self.per_request_draft_token_num is None, (
                "TileSpec (per_request_draft_token_num) only supports greedy sampling. "
                "Set temperature=0 or use --sampling-defaults greedy."
            )

            # apply temperature and get target probs
            if self.per_request_draft_token_num is not None:
                # Per-request: use pad_row_indices for indexing (faster than repeat_interleave)
                expanded_temperature = sampling_info.temperatures[pad_row_indices]
                expanded_top_ks = sampling_info.top_ks[pad_row_indices]
                expanded_top_ps = sampling_info.top_ps[pad_row_indices]
            else:
                # Uniform: repeat_interleave with scalar is efficient
                expanded_temperature = torch.repeat_interleave(
                    sampling_info.temperatures, self.draft_token_num, dim=0
                )
                expanded_top_ks = torch.repeat_interleave(
                    sampling_info.top_ks, self.draft_token_num, dim=0
                )
                expanded_top_ps = torch.repeat_interleave(
                    sampling_info.top_ps, self.draft_token_num, dim=0
                )

            target_probs = F.softmax(
                logits_output.next_token_logits / expanded_temperature, dim=-1
            )  # (total_tokens, vocab_size)
            target_probs = top_k_renorm_prob(
                target_probs,
                expanded_top_ks,
            )  # (total_tokens, vocab_size)
            if not torch.all(sampling_info.top_ps == 1.0):
                target_probs = top_p_renorm_prob(
                    target_probs,
                    expanded_top_ps,
                )

            # Handle both uniform and per-request draft tokens
            if self.per_request_draft_token_num is not None:
                # Ragged case: pad to max count (use cached indices)
                target_probs = _pad_to_2d_with_indices(
                    target_probs, pad_row_indices, pad_col_indices, bs, max_count, pad_value=0.0
                )
            else:
                # Uniform case: simple reshape
                target_probs = target_probs.reshape(bs, self.draft_token_num, -1)

            draft_probs = torch.zeros(
                target_probs.shape, dtype=torch.float32, device=batch.device
            )

            # coins for rejection sampling
            coins = torch.rand_like(
                candidates, dtype=torch.float32, device=batch.device
            )
            # coins for final sampling
            coins_for_final_sampling = torch.rand(
                (bs,), dtype=torch.float32, device=batch.device
            )
            tree_speculative_sampling_target_only(
                predicts=predict,  # mutable
                accept_index=accept_index,  # mutable
                accept_token_num=accept_length,  # mutable
                candidates=candidates,
                retrive_index=self.retrive_index,
                retrive_next_token=self.retrive_next_token,
                retrive_next_sibling=self.retrive_next_sibling,
                uniform_samples=coins,
                uniform_samples_for_final_sampling=coins_for_final_sampling,
                target_probs=target_probs,
                draft_probs=draft_probs,
                threshold_single=get_global_server_args().speculative_accept_threshold_single,
                threshold_acc=get_global_server_args().speculative_accept_threshold_acc,
                deterministic=True,
            )

        if SIMULATE_ACC_LEN > 0.0:
            # Do simulation
            accept_index = generate_simulated_accept_index(
                accept_index=accept_index,
                predict=predict,  # mutable
                accept_length=accept_length,  # mutable
                bs=bs,
                spec_steps=self.spec_steps,
            )

        unfinished_index = []
        unfinished_accept_index = []
        accept_index_cpu = accept_index.tolist()
        predict_cpu = predict.tolist()
        has_finished = False

        # Iterate every accepted token and check if req has finished after append the token
        # should be checked BEFORE free kv cache slots
        for i, (req, accept_index_row) in enumerate(zip(batch.reqs, accept_index_cpu)):
            for j, idx in enumerate(accept_index_row):
                if idx == -1:
                    break
                id = predict_cpu[idx]
                req.output_ids.append(id)
                req.check_finished()
                if req.finished():
                    has_finished = True
                    # set all tokens after finished token to -1 and break
                    accept_index[i, j + 1 :] = -1
                    break
                else:
                    if req.grammar is not None:
                        try:
                            req.grammar.accept_token(id)
                        except ValueError as e:
                            logger.info(
                                f"{i=}, {req=}\n" f"{accept_index=}\n" f"{predict=}\n"
                            )
                            raise e
            if not req.finished():
                unfinished_index.append(i)
                if idx == -1:
                    unfinished_accept_index.append(accept_index[i, :j])
                else:
                    unfinished_accept_index.append(accept_index[i])
            req.spec_verify_ct += 1
            req.spec_accepted_tokens += (
                sum(1 for idx in accept_index_row if idx != -1) - 1
            )

        if has_finished:
            accept_length = (accept_index != -1).sum(dim=1) - 1

        # Build accepted mask for TileSpec calibration (only during profiling)
        # selected_scores has only draft tokens, but accept_index points into draft_token
        # which includes verified tokens. We must map indices correctly.
        if self.selected_scores is not None:
            num_scores = self.selected_scores.shape[0]
            accepted_mask = torch.zeros(num_scores, dtype=torch.bool, device=batch.device)

            # Get tokens per request (includes verified token)
            if self.per_request_draft_token_num is not None:
                tokens_per_req = self.per_request_draft_token_num.tolist()
            else:
                tokens_per_req = [self.draft_token_num] * bs

            # Map accept_index to selected_scores indices
            # Note: accept_index has PADDED indices when per_request_draft_token_num is set
            # Convert to CPU once outside the loop to avoid repeated GPU→CPU syncs
            accept_index_cpu = accept_index.tolist()
            score_offset = 0  # offset into selected_scores (no verified)
            for req_idx in range(bs):
                num_tokens = tokens_per_req[req_idx]
                num_drafts = num_tokens - 1  # exclude verified
                # Use padded offset for ragged case, ragged offset for uniform case
                if self.per_request_draft_token_num is not None:
                    req_offset = req_idx * max_count  # padded offset
                else:
                    req_offset = req_idx * self.draft_token_num  # uniform offset

                for idx in accept_index_cpu[req_idx]:
                    if idx == -1:
                        break
                    pos = idx - req_offset  # position within this request
                    if pos > 0:  # skip verified token (pos 0)
                        score_idx = score_offset + (pos - 1)
                        if 0 <= score_idx < num_scores:
                            accepted_mask[score_idx] = True

                score_offset += num_drafts

            self._accepted_mask = accepted_mask

        # Free the KV cache for unaccepted tokens
        # TODO: fuse them

        # With ragged verify kernel: accept_index already has RAGGED indices
        # predict is also RAGGED [total_tokens], so no conversion needed!
        accept_index = accept_index[accept_index != -1]
        verified_id = predict[accept_index]

        if _TILESPEC_DEBUG:
            torch.cuda.synchronize()
            _t_verify_end = time.perf_counter()
            if self.per_request_draft_token_num is not None:
                print(f"[TileSpec] verify() ragged overhead: {(_t_verify_end-_t_verify_start)*1000:.3f}ms (bs={bs})")
            else:
                print(f"[EAGLE] verify() uniform: {(_t_verify_end-_t_verify_start)*1000:.3f}ms (bs={bs})")

        evict_mask = torch.full_like(self.draft_token, True, dtype=torch.bool)
        evict_mask[accept_index] = False
        accept_length_cpu = accept_length.cpu()
        # FIXME: this `tolist()` fixes the numerical calculation consistency
        # try to unify the tensor representation and list representation
        accept_length_list = accept_length_cpu.tolist()

        if page_size == 1:
            # TODO: boolean array index leads to a device sync. Remove it.
            token_to_kv_pool_allocator.free(batch.out_cache_loc[evict_mask])
            for i, req in enumerate(batch.reqs):
                req.kv_committed_len += accept_length_list[i] + 1
                req.kv_allocated_len = req.kv_committed_len
        else:
            if self.topk == 1:
                # Only evict full empty page. Do not evict partial empty page
                align_evict_mask_to_page_size[len(batch.seq_lens),](
                    batch.seq_lens,
                    evict_mask,
                    page_size,
                    self.draft_token_num,
                    next_power_of_2(self.draft_token_num),
                )
                token_to_kv_pool_allocator.free(batch.out_cache_loc[evict_mask])
                for i, req in enumerate(batch.reqs):
                    req.kv_committed_len += accept_length_list[i] + 1
                    req.kv_allocated_len = req.kv_committed_len
            else:
                # Shift the accepted tokens to the beginning.
                # Only evict the last part
                src_cache_loc, tgt_cache_loc, to_free_num_slots = get_src_tgt_cache_loc(
                    batch.seq_lens,
                    batch.out_cache_loc,
                    accept_index,
                    accept_length,
                    self.draft_token_num,
                    page_size,
                )
                to_free_slots = torch.empty(
                    (to_free_num_slots.sum().item(),),
                    dtype=torch.int64,
                    device=to_free_num_slots.device,
                )

                # out_cache_loc: [0  1  2,  3  4  5,  6  7  8]
                # accept_index:  [0 -1  2,  3  4 -1,  6 -1 -1]
                # tgt_cache_loc: [0  1   ,  3  4   ,  6      ]
                # to_free_slots: [      2,        5,     7  8]
                # to_free_slots also needs to be page-aligned without the first partial page
                #
                # split each row of out_cache_loc into two parts.
                # 1. the first part goes to tgt_cache_loc. length = accept_length[i] + 1
                # 2. the second part goes to to_free_slots.
                get_target_cache_loc[(bs,)](
                    tgt_cache_loc,
                    to_free_slots,
                    accept_length,
                    to_free_num_slots,
                    batch.out_cache_loc,
                    self.draft_token_num,
                    next_power_of_2(self.draft_token_num),
                    next_power_of_2(bs),
                )

                # Free the kv cache
                token_to_kv_pool_allocator.free(to_free_slots)

                # Copy the kv cache
                batch.token_to_kv_pool_allocator.get_kvcache().move_kv_cache(
                    tgt_cache_loc, src_cache_loc
                )

        # Construct EagleVerifyOutput
        if not has_finished:
            if page_size == 1 or self.topk == 1:
                batch.out_cache_loc = batch.out_cache_loc[accept_index]
                assign_req_to_token_pool_func(
                    batch.req_pool_indices,
                    batch.req_to_token_pool.req_to_token,
                    batch.seq_lens,
                    batch.seq_lens + accept_length + 1,
                    batch.out_cache_loc,
                    bs,
                )
            else:
                batch.out_cache_loc = tgt_cache_loc
            batch.seq_lens.add_(accept_length + 1)
            batch.seq_lens_cpu.add_(accept_length_cpu + 1)

            draft_input = EagleDraftInput(
                hidden_states=batch.spec_info.hidden_states[accept_index],
                verified_id=verified_id,
                accept_length=accept_length,
                accept_length_cpu=accept_length_list,
                seq_lens_for_draft_extend=batch.seq_lens,
                seq_lens_for_draft_extend_cpu=batch.seq_lens_cpu,
                req_pool_indices_for_draft_extend=batch.req_pool_indices,
            )

            return EagleVerifyOutput(
                draft_input=draft_input,
                logits_output=logits_output,
                verified_id=verified_id,
                accept_length_per_req_cpu=draft_input.accept_length_cpu,
                accepted_indices=accept_index,
            )
        else:
            if page_size == 1 or self.topk == 1:
                assign_req_to_token_pool_func(
                    batch.req_pool_indices,
                    batch.req_to_token_pool.req_to_token,
                    batch.seq_lens,
                    batch.seq_lens + accept_length + 1,
                    batch.out_cache_loc[accept_index],
                    bs,
                )
                batch.seq_lens.add_(accept_length + 1)
                batch.seq_lens_cpu.add_(accept_length_cpu + 1)

            if len(unfinished_accept_index) > 0:
                unfinished_accept_index = torch.cat(unfinished_accept_index)

                # With ragged verify kernel: accept_index already has RAGGED indices
                # predict is also RAGGED, so no conversion needed
                unfinished_accept_index_ragged = unfinished_accept_index

                unfinished_index_device = torch.tensor(
                    unfinished_index, dtype=torch.int64, device=predict.device
                )
                draft_input_accept_length_cpu = [
                    accept_length_list[i] for i in unfinished_index
                ]
                if page_size == 1 or self.topk == 1:
                    batch.out_cache_loc = batch.out_cache_loc[unfinished_accept_index_ragged]
                else:
                    batch.out_cache_loc = torch.empty(
                        len(unfinished_index) + sum(draft_input_accept_length_cpu),
                        dtype=torch.int64,
                        device=predict.device,
                    )
                    accept_length_filter = create_accept_length_filter(
                        accept_length,
                        unfinished_index_device,
                        batch.seq_lens,
                    )
                    batch.seq_lens_cpu.add_(accept_length_cpu + 1)
                    filter_finished_cache_loc_kernel[(bs,)](
                        batch.out_cache_loc,
                        tgt_cache_loc,
                        accept_length,
                        accept_length_filter,
                        next_power_of_2(bs),
                        next_power_of_2(self.draft_token_num),
                    )

                draft_input = EagleDraftInput(
                    hidden_states=batch.spec_info.hidden_states[
                        unfinished_accept_index_ragged  # ragged indices
                    ],
                    verified_id=predict[unfinished_accept_index],  # ragged indices, predict is ragged
                    accept_length_cpu=draft_input_accept_length_cpu,
                    accept_length=accept_length[unfinished_index_device],
                    seq_lens_for_draft_extend=batch.seq_lens[unfinished_index_device],
                    seq_lens_for_draft_extend_cpu=batch.seq_lens_cpu[unfinished_index],
                    req_pool_indices_for_draft_extend=batch.req_pool_indices[
                        unfinished_index_device
                    ],
                )
            else:
                draft_input = EagleDraftInput.create_idle_input(
                    device=batch.device,
                    hidden_size=batch.model_config.hidden_size,
                    dtype=batch.model_config.dtype,
                    topk=self.topk,
                    capture_hidden_mode=CaptureHiddenMode.LAST,
                )

            return EagleVerifyOutput(
                draft_input=draft_input,
                logits_output=logits_output,
                verified_id=verified_id,
                accept_length_per_req_cpu=accept_length_list,
                accepted_indices=accept_index,
            )


@dataclass
class EagleDraftInput(SpecInput, EagleDraftInputV2Mixin):
    # Constant: alloc length per decode step
    ALLOC_LEN_PER_DECODE: ClassVar[int] = None

    # The inputs for decode
    # shape: (b, topk)
    topk_p: torch.Tensor = None
    topk_index: torch.Tensor = None
    # shape: (b, hidden_size)
    hidden_states: torch.Tensor = None
    capture_hidden_mode: CaptureHiddenMode = CaptureHiddenMode.FULL

    # Inputs for extend
    # shape: (b,)
    verified_id: torch.Tensor = None
    accept_length: torch.Tensor = None
    accept_length_cpu: List[int] = None

    # Inputs for the attention backends
    # shape: (b + 1,)
    kv_indptr: torch.Tensor = None
    kv_indices: torch.Tensor = None

    # Shape info for padding
    num_tokens_per_batch: int = -1
    num_tokens_for_logprob_per_batch: int = -1

    # Inputs for draft extend
    # shape: (b,)
    seq_lens_for_draft_extend: torch.Tensor = None
    seq_lens_for_draft_extend_cpu: torch.Tensor = None
    req_pool_indices_for_draft_extend: torch.Tensor = None

    # Inputs for V2 overlap worker
    future_indices: Optional[FutureIndices] = None
    new_seq_lens: Optional[torch.Tensor] = None
    verify_done: Optional[torch.cuda.Event] = None

    def __post_init__(self):
        super().__init__(SpecInputType.EAGLE_DRAFT)

    def get_spec_adjust_token_coefficient(self) -> Tuple[int, int]:
        return self.num_tokens_per_batch, self.num_tokens_for_logprob_per_batch

    def prepare_for_extend(self, batch: ScheduleBatch):

        if batch.forward_mode.is_idle():
            return

        # Prefill only generate 1 token.
        assert len(self.verified_id) == len(batch.seq_lens)

        pt = 0
        for i, extend_len in enumerate(batch.extend_lens):
            input_ids = batch.input_ids[pt : pt + extend_len]
            batch.input_ids[pt : pt + extend_len] = torch.cat(
                (input_ids[1:], self.verified_id[i].reshape(1))
            )
            pt += extend_len

    @classmethod
    def create_idle_input(
        cls,
        device: torch.device,
        hidden_size: int,
        dtype: torch.dtype,
        topk: int,
        capture_hidden_mode: CaptureHiddenMode,
    ):
        return cls(
            verified_id=torch.empty((0,), device=device, dtype=torch.int32),
            hidden_states=torch.empty((0, hidden_size), device=device, dtype=dtype),
            topk_p=torch.empty((0, topk), device=device, dtype=torch.float32),
            topk_index=torch.empty((0, topk), device=device, dtype=torch.int64),
            capture_hidden_mode=capture_hidden_mode,
            new_seq_lens=torch.empty((0,), device=device, dtype=torch.int32),
            accept_length=torch.empty((0,), device=device, dtype=torch.int32),
            accept_length_cpu=[],
        )

    def prepare_extend_after_decode(
        self,
        batch: ScheduleBatch,
        speculative_num_steps: int,
    ):

        if batch.forward_mode.is_idle():
            return

        batch.input_ids = self.verified_id
        batch.extend_lens = [x + 1 for x in batch.spec_info.accept_length_cpu]
        batch.extend_num_tokens = sum(batch.extend_lens)
        batch.seq_lens = batch.spec_info.seq_lens_for_draft_extend
        batch.seq_lens_cpu = batch.spec_info.seq_lens_for_draft_extend_cpu
        batch.req_pool_indices = batch.spec_info.req_pool_indices_for_draft_extend
        batch.return_logprob = False
        batch.return_hidden_states = False

        self.capture_hidden_mode = CaptureHiddenMode.LAST
        self.accept_length.add_(1)
        self.positions = torch.empty_like(batch.input_ids, dtype=torch.long)
        self.verified_id = torch.empty_like(self.accept_length, dtype=torch.int32)

        create_extend_after_decode_spec_info[(len(batch.seq_lens),)](
            batch.input_ids,
            batch.seq_lens,
            self.accept_length,
            self.positions,
            self.verified_id,
            next_power_of_2(max(speculative_num_steps + 1, len(batch.seq_lens))),
        )

    def generate_attn_arg_prefill(
        self,
        req_pool_indices: torch.Tensor,
        paged_kernel_lens: torch.Tensor,
        paged_kernel_lens_sum: int,
        req_to_token: torch.Tensor,
    ):
        device = req_pool_indices.device
        bs = self.accept_length.numel()
        qo_indptr = torch.zeros((bs + 1,), dtype=torch.int32, device=device)
        qo_indptr[1:] = torch.cumsum(self.accept_length, dim=0)
        cum_kv_seq_len = torch.zeros((bs + 1,), dtype=torch.int32, device=device)
        cum_kv_seq_len[1:] = torch.cumsum(paged_kernel_lens, dim=0)

        if paged_kernel_lens_sum is None:
            paged_kernel_lens_sum = cum_kv_seq_len[-1]

        kv_indices = torch.empty(
            paged_kernel_lens_sum, dtype=torch.int32, device=device
        )

        create_flashinfer_kv_indices_triton[(bs,)](
            req_to_token,
            req_pool_indices,
            paged_kernel_lens,
            cum_kv_seq_len,
            None,
            kv_indices,
            req_to_token.size(1),
        )
        return kv_indices, cum_kv_seq_len, qo_indptr, None

    def filter_batch(self, new_indices: torch.Tensor, has_been_filtered: bool = True):
        if self.future_indices is not None:
            self.future_indices.indices = self.future_indices.indices[new_indices]
            return

        strict_check = envs.SGLANG_SPEC_ENABLE_STRICT_FILTER_CHECK.get()
        if has_been_filtered:
            # in eagle_utils.py:verify, we have already filtered the batch by `unfinished_index`
            # therefore, we don't need to filter the batch again in scheduler
            error_msg = f"length of new_indices: {len(new_indices)} != length of topk_p: {len(self.topk_p)}, this should not happen"
            if len(new_indices) != len(self.topk_p):
                if strict_check:
                    raise ValueError(error_msg)
                else:
                    logger.warning(error_msg)

            self.topk_p = self.topk_p[: len(new_indices)]
            self.topk_index = self.topk_index[: len(new_indices)]
            self.hidden_states = self.hidden_states[: len(new_indices)]
            self.verified_id = self.verified_id[: len(new_indices)]
        else:
            # in some cases(e.g draft_extend), we have not filtered the batch by `unfinished_index`
            self.topk_p = self.topk_p[new_indices]
            self.topk_index = self.topk_index[new_indices]
            self.hidden_states = self.hidden_states[new_indices]
            self.verified_id = self.verified_id[new_indices]

    def merge_batch(self, spec_info: "EagleDraftInput"):
        if self.future_indices is not None:
            assert spec_info.future_indices is not None
            self.future_indices = FutureIndices(
                indices=torch.cat(
                    [self.future_indices.indices, spec_info.future_indices.indices]
                )
            )
            return

        if self.hidden_states is None:
            self.hidden_states = spec_info.hidden_states
            self.verified_id = spec_info.verified_id
            self.topk_p = spec_info.topk_p
            self.topk_index = spec_info.topk_index
            return
        if spec_info.hidden_states is None:
            return
        self.hidden_states = torch.cat(
            [self.hidden_states, spec_info.hidden_states], axis=0
        )
        self.verified_id = torch.cat([self.verified_id, spec_info.verified_id], axis=0)
        self.topk_p = torch.cat([self.topk_p, spec_info.topk_p])
        self.topk_index = torch.cat([self.topk_index, spec_info.topk_index])


@dataclass
class EagleVerifyOutput:
    # Draft input batch
    draft_input: EagleDraftInput
    # Logit outputs from target worker
    logits_output: LogitsProcessorOutput
    # Accepted token ids including the bonus token
    verified_id: torch.Tensor
    # Accepted token length per sequence in a batch in CPU.
    accept_length_per_req_cpu: List[int]
    # Accepted indices from logits_output.next_token_logits
    accepted_indices: torch.Tensor
