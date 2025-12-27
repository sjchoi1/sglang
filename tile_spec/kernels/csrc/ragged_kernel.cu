/*
 * TileSpec Ragged Kernels for Speculative Decoding
 *
 * These kernels support variable draft token counts per request (ragged layout).
 *
 * Key design:
 * - positions, candidates, target_predict, predicts: 1D ragged (use token_indptr)
 * - retrive_next_token, retrive_next_sibling: 2D padded (internal tree structure)
 * - retrive_index: 2D padded storage, but VALUES are ragged indices
 *
 * This allows verify_tree_greedy_ragged to work without padding candidates/target_predict.
 */

#include <torch/extension.h>
#include <cuda_runtime.h>

typedef enum { FULL_MASK = 0, QLEN_ONLY = 1, QLEN_ONLY_BITPACKING = 2 } TreeMaskMode;

/*
 * Build tree kernel for ragged draft tokens.
 *
 * Key difference from uniform: retrive_index stores RAGGED indices (token_offset + i),
 * not padded indices (retrive_offset + i). This allows verify kernel to work with
 * ragged candidates/target_predict without padding.
 */
__global__ void build_tree_efficient_ragged_kernel(
    int64_t* parent_list,
    int64_t* selected_index,
    int64_t* verified_seq_len,
    bool* tree_mask,
    int64_t* positions,
    int64_t* retrive_index,
    int64_t* retrive_next_token,
    int64_t* retrive_next_sibling,
    int64_t* token_indptr,
    int64_t* score_indptr,
    int64_t* mask_indptr,
    int topk,
    int depth,
    int tree_mask_mode,
    int parent_list_stride) {

  int bid = blockIdx.x;
  int tid = threadIdx.x;

  // Get per-request counts from indptr
  int64_t token_offset = token_indptr[bid];
  int64_t score_offset = score_indptr[bid];
  int64_t mask_offset = mask_indptr[bid];
  int draft_token_num = token_indptr[bid + 1] - token_offset;
  int num_scores = score_indptr[bid + 1] - score_offset;

  // retrive_next_token/sibling use padded 2D layout (internal tree structure)
  int retrive_stride = blockDim.x;
  int retrive_offset = bid * retrive_stride;

  // Early exit for threads beyond this request's token count
  if (tid >= draft_token_num) {
    return;
  }

  int seq_len = verified_seq_len[bid];

  // Compute tree_mask index based on mode
  int token_tree_idx;
  if (tree_mask_mode == FULL_MASK) {
    token_tree_idx = mask_offset + (seq_len + draft_token_num) * tid + seq_len + 1;
  } else {
    token_tree_idx = mask_offset + draft_token_num * tid + 1;
  }

  // Initialize tree mask for this token
  tree_mask[token_tree_idx - 1] = true;
  for (int i = 0; i < draft_token_num - 1; i++) {
    tree_mask[token_tree_idx + i] = false;
  }

  int position = 0;
  if (tid == 0) {
    // Thread 0: handles positions[0] and builds retrieval structures
    positions[token_offset] = seq_len;

    // Build retrieval tree structure (must be done sequentially)
    for (int i = draft_token_num - 1; i > 0; --i) {
      // KEY CHANGE: retrive_index stores RAGGED index (for verify kernel)
      retrive_index[retrive_offset + i] = token_offset + i;

      // Find parent position using selected_index
      int parent_tb_idx = selected_index[score_offset + i - 1] / topk;
      int parent_position = 0;

      if (parent_tb_idx > 0) {
        int parent_token_idx = parent_list[bid * parent_list_stride + parent_tb_idx];
        for (; parent_position < num_scores; ++parent_position) {
          if (selected_index[score_offset + parent_position] == parent_token_idx) {
            ++parent_position;
            break;
          }
        }
      }

      if (parent_position == draft_token_num) {
        printf(
            "WARNING: invalid eagle tree (ragged)!!! Detected a token with no parent token selected. "
            "Please check if the logprob has nan. The token will be ignored to keep proceeding.\n");
        continue;
      }

      // Link this token to its parent (using padded indices for tree structure)
      if (retrive_next_token[retrive_offset + parent_position] == -1) {
        retrive_next_token[retrive_offset + parent_position] = i;
      } else {
        int origin_next_token = retrive_next_token[retrive_offset + parent_position];
        retrive_next_token[retrive_offset + parent_position] = i;
        retrive_next_sibling[retrive_offset + i] = origin_next_token;
      }
    }
    // KEY CHANGE: retrive_index[0] stores ragged index
    retrive_index[retrive_offset] = token_offset;

  } else {
    // Other threads: compute positions and tree_mask by walking up the tree
    int cur_position = tid - 1;
    while (true) {
      position += 1;
      tree_mask[token_tree_idx + cur_position] = true;

      int parent_tb_idx = selected_index[score_offset + cur_position] / topk;
      if (parent_tb_idx == 0) {
        break;
      }

      int token_idx = parent_list[bid * parent_list_stride + parent_tb_idx];
      for (cur_position = 0; cur_position < num_scores; ++cur_position) {
        if (selected_index[score_offset + cur_position] == token_idx) {
          break;
        }
      }
      if (cur_position >= num_scores) {
        break;
      }
    }
    positions[token_offset + tid] = position + seq_len;
  }
}

void build_tree_kernel_efficient_ragged(
    at::Tensor parent_list,
    at::Tensor selected_index,
    at::Tensor verified_seq_len,
    at::Tensor tree_mask,
    at::Tensor positions,
    at::Tensor retrive_index,
    at::Tensor retrive_next_token,
    at::Tensor retrive_next_sibling,
    at::Tensor token_indptr,
    at::Tensor score_indptr,
    at::Tensor mask_indptr,
    int64_t topk,
    int64_t depth,
    int64_t max_draft_token_num,
    int64_t tree_mask_mode) {

  int bs = token_indptr.size(0) - 1;
  int parent_list_stride = parent_list.size(1);

  build_tree_efficient_ragged_kernel<<<bs, max_draft_token_num>>>(
      parent_list.data_ptr<int64_t>(),
      selected_index.data_ptr<int64_t>(),
      verified_seq_len.data_ptr<int64_t>(),
      tree_mask.data_ptr<bool>(),
      positions.data_ptr<int64_t>(),
      retrive_index.data_ptr<int64_t>(),
      retrive_next_token.data_ptr<int64_t>(),
      retrive_next_sibling.data_ptr<int64_t>(),
      token_indptr.data_ptr<int64_t>(),
      score_indptr.data_ptr<int64_t>(),
      mask_indptr.data_ptr<int64_t>(),
      topk,
      depth,
      tree_mask_mode,
      parent_list_stride);
}

/*
 * Verify tree greedy kernel for ragged layout.
 *
 * Key differences from uniform verify_tree_greedy:
 * - candidates: 1D ragged [total_tokens], use token_indptr for per-request offset
 * - target_predict: 1D ragged [total_tokens]
 * - predicts: 1D ragged [total_tokens]
 * - retrive_index: stores RAGGED indices (from build_tree_ragged)
 * - retrive_next_token/sibling: still 2D padded [bs, max_draft_tokens]
 *
 * This eliminates the need for padding candidates/target_predict in Python!
 */
template <typename IdType, typename IdType2>
__global__ void VerifyTreeGreedyRagged(
    IdType* predicts,           // [total_tokens] - ragged, mutable
    IdType* accept_index,       // [bs, num_spec_step] - output
    IdType* accept_token_num,   // [bs] - output
    IdType2* candidates,        // [total_tokens] - ragged
    IdType2* retrive_index,     // [bs, max_draft_tokens] - padded, but VALUES are ragged
    IdType2* retrive_next_token,  // [bs, max_draft_tokens] - padded
    IdType2* retrive_next_sibling, // [bs, max_draft_tokens] - padded
    IdType2* target_predict,    // [total_tokens] - ragged
    IdType2* token_indptr,      // [bs + 1] - cumulative offsets
    uint32_t batch_size,
    uint32_t num_speculative_tokens,
    uint32_t max_draft_tokens) {

  uint32_t bx = blockIdx.x;

  // Get this request's token offset
  IdType2 token_offset = token_indptr[bx];

  // retrive_index stores RAGGED indices, so this gives us the ragged index directly
  IdType2 last_accepted_retrive_idx = retrive_index[bx * max_draft_tokens];
  accept_index[bx * num_speculative_tokens] = last_accepted_retrive_idx;
  uint32_t num_accepted_tokens = 0;
  IdType2 cur_index = 0;  // Local index within padded retrive_* arrays

  for (uint32_t j = 1; j < num_speculative_tokens; ++j) {
    // Navigate tree using padded retrive_next_token
    cur_index = retrive_next_token[bx * max_draft_tokens + cur_index];

    while (cur_index != -1) {
      // retrive_index gives us the RAGGED index directly
      IdType2 draft_index = retrive_index[bx * max_draft_tokens + cur_index];

      // Access candidates using ragged index: token_offset + local_index
      IdType2 draft_token_id = candidates[token_offset + cur_index];

      // target_predict uses the ragged index from last accepted
      IdType2 target_token_id = target_predict[last_accepted_retrive_idx];

      if (draft_token_id == target_token_id) {
        // Accept token
        predicts[last_accepted_retrive_idx] = target_token_id;
        ++num_accepted_tokens;
        accept_index[bx * num_speculative_tokens + num_accepted_tokens] = draft_index;
        last_accepted_retrive_idx = draft_index;
        break;
      } else {
        // Try sibling
        cur_index = retrive_next_sibling[bx * max_draft_tokens + cur_index];
      }
    }
    if (cur_index == -1) break;
  }

  accept_token_num[bx] = num_accepted_tokens;
  // Write final prediction
  predicts[last_accepted_retrive_idx] = target_predict[last_accepted_retrive_idx];
}

void verify_tree_greedy_ragged(
    at::Tensor predicts,           // [total_tokens] - mutable
    at::Tensor accept_index,       // [bs, num_spec_step] - mutable
    at::Tensor accept_token_num,   // [bs] - mutable
    at::Tensor candidates,         // [total_tokens]
    at::Tensor retrive_index,      // [bs, max_draft_tokens]
    at::Tensor retrive_next_token, // [bs, max_draft_tokens]
    at::Tensor retrive_next_sibling, // [bs, max_draft_tokens]
    at::Tensor target_predict,     // [total_tokens]
    at::Tensor token_indptr,       // [bs + 1]
    int64_t num_speculative_tokens) {

  int bs = token_indptr.size(0) - 1;
  int max_draft_tokens = retrive_index.size(1);

  dim3 grid(bs);
  dim3 block(1);

  VerifyTreeGreedyRagged<int32_t, int64_t><<<grid, block>>>(
      predicts.data_ptr<int32_t>(),
      accept_index.data_ptr<int32_t>(),
      accept_token_num.data_ptr<int32_t>(),
      candidates.data_ptr<int64_t>(),
      retrive_index.data_ptr<int64_t>(),
      retrive_next_token.data_ptr<int64_t>(),
      retrive_next_sibling.data_ptr<int64_t>(),
      target_predict.data_ptr<int64_t>(),
      token_indptr.data_ptr<int64_t>(),
      bs,
      num_speculative_tokens,
      max_draft_tokens);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("build_tree_kernel_efficient_ragged", &build_tree_kernel_efficient_ragged,
        "Build tree kernel for ragged (variable-length) draft tokens per request",
        py::arg("parent_list"),
        py::arg("selected_index"),
        py::arg("verified_seq_len"),
        py::arg("tree_mask"),
        py::arg("positions"),
        py::arg("retrive_index"),
        py::arg("retrive_next_token"),
        py::arg("retrive_next_sibling"),
        py::arg("token_indptr"),
        py::arg("score_indptr"),
        py::arg("mask_indptr"),
        py::arg("topk"),
        py::arg("depth"),
        py::arg("max_draft_token_num"),
        py::arg("tree_mask_mode"));
  m.def("verify_tree_greedy_ragged", &verify_tree_greedy_ragged,
        "Verify tree greedy kernel for ragged layout (no padding needed)",
        py::arg("predicts"),
        py::arg("accept_index"),
        py::arg("accept_token_num"),
        py::arg("candidates"),
        py::arg("retrive_index"),
        py::arg("retrive_next_token"),
        py::arg("retrive_next_sibling"),
        py::arg("target_predict"),
        py::arg("token_indptr"),
        py::arg("num_speculative_tokens"));
}
