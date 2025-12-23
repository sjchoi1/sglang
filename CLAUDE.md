# CLAUDE.md

This document captures key insights about the SGLang codebase, particularly speculative decoding.

**Base commit:** `0071fe9c407ad59f2803cc319e1bcaa3ac2021f1`

## Build & Test Commands

```bash
# Install in development mode
pip install -e "python[dev]"

# Run tests
pytest python/sglang/test/

# Run specific speculative decoding tests
pytest python/sglang/test/ -k "speculative"

# Start server with speculative decoding
python -m sglang.launch_server \
    --model-path meta-llama/Llama-3.1-8B \
    --speculative-algorithm EAGLE \
    --speculative-draft-model-path meta-llama/Llama-3.1-8B-Instruct
```

## Repository Structure

```
sglang/
├── python/sglang/srt/          # Core runtime
│   ├── speculative/            # Speculative decoding implementations
│   │   ├── eagle_worker.py     # EAGLE/EAGLE3 speculative worker
│   │   ├── standalone_worker.py # Draft LM-based speculative worker
│   │   ├── ngram_worker.py     # NGRAM speculative worker
│   │   ├── eagle_utils.py      # Tree building, verification kernels
│   │   └── tile_spec/          # TileSpec: tile-aware dynamic speculation
│   │       ├── core.py         # Calibration, latency model, compute_optimal_k
│   │       └── profiler.py     # Automatic profiling and caching
│   ├── managers/
│   │   └── schedule_batch.py   # Batch management, ScheduleBatch class
│   ├── layers/sampler.py       # Sampling logic
│   └── server_args.py          # CLI argument definitions
├── sgl-kernel/                 # CUDA kernels
│   └── csrc/speculative/       # Speculative sampling CUDA code
├── tile_spec/                  # TileSpec benchmarks and tests
│   └── run_bench.py            # Benchmark script
└── test/                       # Test files
```

## Speculative Decoding Architecture

### Key Data Flow

1. **Scheduler** (`scheduler.py`) orchestrates the speculative loop
2. **EagleWorker** (`eagle_worker.py`) handles draft generation and verification
3. **ScheduleBatch** (`schedule_batch.py`) manages batch state and token data

### Speculative Decoding Workflow

```
┌─────────────────────────────────────────────────────────────────┐
│                     Scheduler Loop                               │
├─────────────────────────────────────────────────────────────────┤
│  1. draft_forward()     → Generate draft tokens with scores     │
│  2. draft()             → Build verification tree               │
│  3. forward_verify()    → Run target model on tree              │
│  4. verify()            → Accept/reject draft tokens            │
└─────────────────────────────────────────────────────────────────┘
```

### Key Classes

- **EagleWorker**: Main worker class for EAGLE-based speculation
  - `draft_forward()`: Runs draft model, returns (parent_list, top_scores_index, draft_tokens, num_draft_tokens)
  - `draft()`: Builds tree mask and positions using `build_tree_kernel_efficient`
  - `verify()`: Processes verification results, updates batch state

- **ScheduleBatch**: Batch container
  - `input_ids`: Flattened token IDs for current step
  - `seq_lens`: Sequence lengths per request
  - `spec_info`: Speculative decoding metadata (EagleDraftInput/EagleVerifyInput)

### Token Flow During Verification

Draft tokens are **flattened globally** across all requests:
```python
batch.input_ids = self.draft_token  # Shape: [bs * num_draft_tokens]
```

This means the total token count affects GPU tile efficiency (MLP tiling).

## TileSpec: Tile-Aware Dynamic Speculation

### Purpose

GPU MLPs have tile boundaries (e.g., 64, 128, 256 tokens) where latency jumps discontinuously. TileSpec dynamically selects the optimal draft token count to maximize throughput:

```
Objective: maximize E[accepted_tokens] / Latency(total_tokens)
```

### High-Level Flow

```
┌─────────────────────────────────────────────────────────────────────┐
│  1. PROFILING (automatic on first run)                              │
│     - Engine runs warmup requests with varying batch sizes          │
│     - Records (token_count, latency) pairs during verify() calls    │
│     - Collects (score, accepted) pairs for calibration              │
│     - Fits latency model + calibration, caches to disk              │
├─────────────────────────────────────────────────────────────────────┤
│  2. RUNTIME (after profiling or cache load)                         │
│     - draft_forward(): generate draft tokens with confidence scores │
│     - compute_optimal_k(): find best k at tile boundaries           │
│       • Calibrate scores → acceptance probabilities                 │
│       • Search only at segment endpoints (64, 128, 192, 256...)     │
│       • Select k with best E[accepted]/Latency ratio                │
│     - verify(): run target model, record data if still profiling    │
└─────────────────────────────────────────────────────────────────────┘
```

### Key Components

| Component | File | Purpose |
|-----------|------|---------|
| `Calibration` | `tile_spec/core.py` | Maps cumulative draft scores → acceptance probability (linear regression) |
| `PiecewiseLinearLatency` | `tile_spec/core.py` | Models latency with tile boundary detection (15% jump threshold) |
| `compute_optimal_k()` | `tile_spec/core.py` | Finds optimal k by searching segment endpoints |
| `TileSpecProfiler` | `tile_spec/profiler.py` | Automatic profiling during warmup, caching |

### TileSpec Algorithm Details

**Score Structure:**
- SGLang draft scores are **cumulative probabilities** (products of per-position draft probs)
- Computed in `select_top_k_tokens()`: `expand_scores = scores * topk_p` (multiplication)
- Scores decrease along the sequence (product of values < 1)

**Calibration:**
- Linear regression: `P_accept = clamp(slope * cumulative_score + intercept, 0.01, 0.99)`
- Training data: `(cumulative_score, path_accepted)` pairs from verify() calls
- `accepted[i, j] = True` if entire path 0..j was accepted for request i

**Optimal K Selection:**
```python
# 1. Calibrate cumulative scores to acceptance probabilities
probs = calibration.predict(score_list)  # [bs, n_candidates]

# 2. Sort globally by probability (descending)
sorted_probs = torch.sort(probs.flatten(), descending=True)

# 3. For each tile boundary candidate k, compute E[accepted]/Latency
# 4. Select k with best ratio
```

### Configuration

```bash
--tile-spec    # Enable TileSpec optimization (EAGLE/STANDALONE only)
```

Cache stored in: `tile_spec/cache/{model}_{gpu}_tp{N}/`

### Supported Algorithms

| Algorithm | Support | Notes |
|-----------|---------|-------|
| EAGLE/EAGLE3 | ✓ Full | Dynamic per-batch k selection |
| STANDALONE | ✓ Full | Inherits from EAGLEWorker |
| NGRAM | ✗ | Would require C++ changes to expose scores |

## CUDA Kernels

Key speculative decoding kernels in `sgl-kernel/`:

- `build_tree_kernel_efficient`: Builds tree mask and positions for verification
- `verify_tree_greedy`: Greedy tree verification (accept/reject)
- `TreeSpeculativeSamplingTargetOnly`: Target-only speculative sampling with threshold-based acceptance

## Common Patterns

### Adding New Speculative Features

1. Add config in `server_args.py` (dataclass field + argparse)
2. Initialize in `EagleWorker.__init__()`
3. Modify `draft_forward()` or `draft()` as needed
4. Update `verify()` if acceptance logic changes

### Score Handling

EAGLE scores are **cumulative path scores** (products of probabilities along draft path, NOT log probabilities). They represent confidence that the entire path up to that token is correct.

```python
# In select_top_k_tokens() - spec_utils.py
# Step 0: scores = topk_p (raw probabilities)
# Step 1+: expand_scores = scores * topk_p (cumulative product)
```

Since scores are products of probabilities (each < 1), they naturally decrease along the sequence.

## Debugging Tips

- Check `batch.spec_info` for current speculative state
- Use `torch.cuda.synchronize()` before timing measurements
- Latency profiling should use median of multiple runs after warmup

---

# TileSpec Implementation: Ragged Draft Token Support

## Summary

TileSpec enables per-request variable draft token counts to optimize for GPU MLP tile boundaries. This requires modifying SGLang's uniform speculation code to support **ragged (variable-length) draft tokens per request**.

**Key Design Decision:** Use **exact-size ragged allocation and eviction** to minimize code changes and memory waste.

## Memory Layout Strategy

**Ragged Packed Layout** (what we use):
```
Allocation:  [req0_tok0, req0_tok1, req1_tok0, req1_tok1, req1_tok2]
Indices:      0          1          2          3          4
Sizes:        2 tokens for req0, 3 tokens for req1
```

**Why ragged?**
- ✅ No memory waste (no padding)
- ✅ Minimal code changes (closest to original SGLang)
- ✅ Original eviction code works as-is
- ✅ Single kernel call for `assign_req_to_token_pool` (uses cumsum internally)

**Alternative (uniform strided - REJECTED):**
```
[req0_t0, req0_t1, PAD, PAD, PAD, req1_t0, req1_t1, req1_t2, PAD, PAD]
```
- ❌ Wastes memory (padding slots)
- ❌ Requires loops for req_to_token mapping
- ❌ Needs ragged→strided index conversion
- ❌ More complex code

## Changes vs Base Commit (0071fe9c)

### File: `python/sglang/srt/speculative/eagle_info.py` (314 lines changed)

#### 1. Added Helper Function (52 lines)
```python
def _pad_to_2d(flat_tensor, per_request_counts, max_count, pad_value=-1):
    """Pad ragged tensor to uniform 2D for verification kernel."""
```
**Why:** Verification kernel (`verify_tree_greedy`) expects uniform 2D tensors [bs, max_count].

#### 2. Added Field to Dataclass (5 lines)
```python
@dataclass
class EagleVerifyInput:
    per_request_draft_token_num: Optional[torch.Tensor] = None  # [bs] or None
```
**Why:** Track per-request draft counts throughout the pipeline.

#### 3. Modified `prepare_for_verify()` (17 lines changed)
```python
# Before (original):
end_offset = batch.seq_lens + self.draft_token_num  # Scalar broadcast

# After (TileSpec):
if self.per_request_draft_token_num is not None:
    draft_token_counts = self.per_request_draft_token_num  # [bs]
else:
    draft_token_counts = self.draft_token_num  # Scalar
end_offset = batch.seq_lens + draft_token_counts  # Per-request or broadcast
```
**Why:** Tell `assign_req_to_token_pool` the actual per-request counts.  
**Verified minimal:** Only changes `end_offset` calculation. Allocation size (`len(batch.input_ids)`) already correct for both uniform and ragged.

#### 4. Modified `generate_attn_arg_prefill()` (~50 lines)
```python
# Calculate qo_indptr for ragged case
if self.per_request_draft_token_num is not None:
    qo_indptr = torch.cat([
        torch.zeros(1, dtype=torch.int32, device=device),
        draft_counts.cumsum(0).to(torch.int32)  # Cumulative offsets
    ])
else:
    qo_indptr = torch.arange(0, (1+bs)*draft_token_num, step=draft_token_num, ...)
```
**Why:** FlashInfer attention backend needs cumulative offsets for ragged token sequences.  
**Verified minimal:** Only adds conditional for ragged case, keeps original uniform path.

#### 5. Modified `verify()` to Pad for Verification Kernel (~80 lines)
```python
# Before (original):
candidates = self.draft_token.reshape(bs, self.draft_token_num)
target_predict = target_predict.reshape(bs, self.draft_token_num)

# After (TileSpec):
if self.per_request_draft_token_num is not None:
    max_count = self.per_request_draft_token_num.max().item()
    candidates = _pad_to_2d(self.draft_token, self.per_request_draft_token_num, max_count)
    target_predict = _pad_to_2d(target_predict, self.per_request_draft_token_num, max_count)
else:
    candidates = self.draft_token.reshape(bs, self.draft_token_num)
    target_predict = target_predict.reshape(bs, self.draft_token_num)
```
**Why:** Verification kernel requires uniform 2D tensors. Padding is harmless (-1 values never match).  
**Verified minimal:** Only adds ragged case, keeps original uniform path intact.

**Eviction code:** No changes! Original code already uses `torch.full_like(self.draft_token, True)` which creates ragged-sized mask. ✅

### File: `python/sglang/srt/speculative/eagle_utils.py` (~200 lines changed)

#### 1. Modified `organize_draft_results()` (~140 lines)
Added TileSpec logic:
- **Profiling mode:** Random draft counts per request
- **Runtime mode:** Global optimization to select optimal counts
- **Vectorized token selection:** Single topk, vectorized gather/split

**Why:** This is where TileSpec selects per-request draft counts.  
**Verified minimal:** All changes guarded by `if tilespec_active`. Original path unchanged.

#### 2. Modified `build_tree_kernel_efficient()` (~100 lines)
Added loop for ragged case:
```python
if per_request_draft_token_num is not None:
    for req_idx in range(bs):
        req_count = per_request_draft_token_num[req_idx].item()
        # Call kernel with uniform count for this request
        sgl_build_tree_kernel[...](
            ...,
            num_verify_tokens=req_count,  # Actual count for this request
        )
else:
    # Original: single kernel call for all requests
    sgl_build_tree_kernel[...](...)
```
**Why:** Tree building kernel doesn't support ragged input, so we loop per request.  
**Verified minimal:** Original single-call path preserved when `per_request_draft_token_num is None`.

### File: `python/sglang/srt/speculative/eagle_worker.py` (~50 lines changed)

1. Pass `per_request_draft_token_num` through pipeline
2. Add timing for profiling
3. Call profiler to record latency data

**Verified minimal:** Only adds plumbing for new field and profiling hooks.

## Verification: All Changes Are Essential and Minimal

### Essential Changes (Required for Ragged Support)
1. ✅ `_pad_to_2d()` helper - Required to interface with uniform verification kernel
2. ✅ `per_request_draft_token_num` field - Required to track ragged counts
3. ✅ `end_offset` calculation - Required to allocate correct memory per request
4. ✅ `qo_indptr` calculation - Required for FlashInfer attention with ragged tokens
5. ✅ Padding in `verify()` - Required for verification kernel interface
6. ✅ `organize_draft_results()` TileSpec logic - Core TileSpec algorithm
7. ✅ `build_tree_kernel_efficient()` loop - Required because kernel doesn't support ragged
8. ✅ Profiling hooks - Required for TileSpec auto-tuning

### Minimal Impact on Original Code
- **Eviction:** 0 changes (original code already supports ragged via `torch.full_like`)
- **Allocation:** 0 changes (original code already uses `len(batch.input_ids)`)
- **req_to_token mapping:** 0 changes (kernel already handles ragged via internal cumsum)
- **All changes guarded:** Original uniform path preserved when `per_request_draft_token_num is None`

### Efficiency
- **Memory:** No waste (exact-size allocation)
- **Computation:** One extra loop in `build_tree_kernel_efficient` (~16μs for bs=8)
- **Code complexity:** Adds ~400 lines, all isolated to TileSpec paths

## Testing Strategy

```bash
# 1. Uniform EAGLE (original behavior, TileSpec disabled)
python -m sglang.launch_server \
    --model-path meta-llama/Llama-3.1-8B \
    --speculative-algorithm EAGLE3 \
    --speculative-draft-model-path meta-llama/Llama-3.1-8B-Instruct
# Expected: Works exactly as before (no code path changes)

# 2. TileSpec profiling
python -m sglang.launch_server \
    --model-path meta-llama/Llama-3.1-8B \
    --speculative-algorithm EAGLE3 \
    --speculative-draft-model-path meta-llama/Llama-3.1-8B-Instruct \
    --tile-spec \
    --disable-cuda-graph
# Expected: Profiling completes, cache files generated, no memory leaks

# 3. TileSpec runtime
# (Same command after profiling)
# Expected: Per-request dynamic k selection, 10-30% speedup
```

## Key Insights

1. **Ragged is simpler than uniform strided** - Original SGLang already supports ragged allocation/eviction via `len(batch.input_ids)` and `torch.full_like()`.

2. **Only one loop needed** - `build_tree_kernel_efficient` requires a loop because the CUDA kernel doesn't support ragged input. All other operations work with single kernel calls.

3. **Padding is free** - Verification kernel with -1 padding is harmless and avoids kernel modifications.

4. **Changes are isolated** - All TileSpec logic guarded by `if per_request_draft_token_num is not None`. Zero impact on original EAGLE behavior.
