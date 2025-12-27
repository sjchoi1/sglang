/*
 * TileSpec Ragged Tree Building Kernel
 *
 * This kernel builds verification trees for speculative decoding when
 * different requests have different numbers of draft tokens (ragged layout).
 *
 * Based on sgl-kernel's build_tree_efficient, modified for ragged support.
 *
 * Key differences from uniform kernel:
 * - Uses indptr arrays to locate per-request data in flattened tensors
 * - selected_index (top_scores_index): 1D ragged (uses score_indptr)
 * - positions: 1D ragged (uses token_indptr)
 * - tree_mask: 1D ragged (uses mask_indptr)
 * - retrive_*: 2D padded layout with stride = max_draft_token_num
 * - parent_list: 2D [bs, topk*(depth-1)+1] - unchanged from uniform
 */

#include <torch/extension.h>
#include <cuda_runtime.h>

typedef enum { FULL_MASK = 0, QLEN_ONLY = 1, QLEN_ONLY_BITPACKING = 2 } TreeMaskMode;

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
    int parent_list_stride) {  // parent_list is 2D [bs, parent_list_stride]

  int bid = blockIdx.x;
  int tid = threadIdx.x;

  // Get per-request counts from indptr
  int64_t token_offset = token_indptr[bid];
  int64_t score_offset = score_indptr[bid];
  int64_t mask_offset = mask_indptr[bid];
  int draft_token_num = token_indptr[bid + 1] - token_offset;
  int num_scores = score_indptr[bid + 1] - score_offset;

  // retrive_* use padded 2D layout with stride = max_draft_token_num (blockDim.x)
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
    // FULL_MASK: mask includes seq_len context
    // Each row is (seq_len + draft_token_num) entries
    token_tree_idx = mask_offset + (seq_len + draft_token_num) * tid + seq_len + 1;
  } else {
    // QLEN_ONLY or QLEN_ONLY_BITPACKING: mask is draft_token_num x draft_token_num
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
      // retrive_index stores index into padded layout
      int current_token_idx = retrive_offset + i;
      retrive_index[retrive_offset + i] = current_token_idx;

      // Find parent position using selected_index
      int parent_tb_idx = selected_index[score_offset + i - 1] / topk;
      int parent_position = 0;

      if (parent_tb_idx > 0) {
        // Look up parent token in parent_list (2D strided access)
        int parent_token_idx = parent_list[bid * parent_list_stride + parent_tb_idx];

        // Search for parent in selected_index
        for (; parent_position < num_scores; ++parent_position) {
          if (selected_index[score_offset + parent_position] == parent_token_idx) {
            ++parent_position;  // Convert to 1-indexed position
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

      // Link this token to its parent
      if (retrive_next_token[retrive_offset + parent_position] == -1) {
        retrive_next_token[retrive_offset + parent_position] = i;
      } else {
        int origin_next_token = retrive_next_token[retrive_offset + parent_position];
        retrive_next_token[retrive_offset + parent_position] = i;
        retrive_next_sibling[retrive_offset + i] = origin_next_token;
      }
    }
    retrive_index[retrive_offset] = retrive_offset;

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

      // Look up parent token in parent_list (2D strided access)
      int token_idx = parent_list[bid * parent_list_stride + parent_tb_idx];

      // Search for parent in selected_index
      for (cur_position = 0; cur_position < num_scores; ++cur_position) {
        if (selected_index[score_offset + cur_position] == token_idx) {
          break;
        }
      }
      if (cur_position >= num_scores) {
        break;  // Parent not found, stop
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
  int parent_list_stride = parent_list.size(1);  // parent_list is [bs, stride]

  // Launch one block per request, max_draft_token_num threads per block
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

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("build_tree_kernel_efficient_ragged", &build_tree_kernel_efficient_ragged,
        "Build tree kernel for ragged (variable-length) draft tokens per request");
}
