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
