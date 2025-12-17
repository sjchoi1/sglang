# CLAUDE.md

This document captures key insights about the SGLang codebase, particularly speculative decoding.

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
│   │   ├── tile_aware.py       # Tile-aware dynamic speculation
│   │   └── profile_tile_aware.py # Profiling script
│   ├── managers/
│   │   └── schedule_batch.py   # Batch management, ScheduleBatch class
│   ├── layers/sampler.py       # Sampling logic
│   └── server_args.py          # CLI argument definitions
├── sgl-kernel/                 # CUDA kernels
│   └── csrc/speculative/       # Speculative sampling CUDA code
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

## Tile-Aware Dynamic Speculation

### Motivation

GPU MLPs have tile boundaries (e.g., 64, 128, 256 tokens) where latency jumps. Fixed draft token counts may land inefficiently between tiles. The tile-aware algorithm dynamically selects the optimal number of draft tokens to maximize:

```
E[accepted_tokens + bonus_tokens] / Latency(prefill + draft_tokens)
```

### Supported Algorithms

Tile-aware optimization works with all three speculative decoding algorithms:

| Algorithm | Worker | Tile-Aware Approach |
|-----------|--------|---------------------|
| **EAGLE/EAGLE3** | `eagle_worker.py` | Dynamic per-batch k selection using calibrated scores |
| **STANDALONE** (Draft LM) | `standalone_worker.py` | Same as EAGLE (inherits from EAGLEWorker) |
| **NGRAM** | `ngram_worker.py` | Static k adjustment to tile boundary at init |

### Algorithm Overview

1. **Calibration** (EAGLE/STANDALONE only): Map cumulative draft confidence scores to acceptance probabilities
   - Histogram binning (50 bins by default)
   - Fitted offline from (score, accepted) pairs

2. **Latency Model** (all algorithms): Piecewise linear with automatic boundary detection
   - Detects tile boundaries via 15% latency jumps
   - Fits linear regression per segment

3. **Optimal k Selection**:
   - **EAGLE/STANDALONE**: Per-batch dynamic selection
     - Calibrate draft scores → acceptance probabilities
     - Sort globally by probability (descending)
     - Compute cumulative expected accepted tokens
     - Search only tile boundaries (O(5) not O(N)) to find best E/L ratio
   - **NGRAM**: Static adjustment at init
     - Snap `draft_token_num` to nearest tile boundary

### Calibration Dataset Considerations

**Does the calibration dataset matter?** Yes, but moderately:

1. **What affects calibration**:
   - The draft-target model pair (same pair = same calibration)
   - Domain distribution (code vs chat vs reasoning)
   - Prompt length distribution

2. **Practical recommendations**:
   - Use a diverse dataset (mix of domains) for robustness
   - Or calibrate per-domain if workloads are distinct
   - The latency model is domain-agnostic (GPU hardware dependent)

3. **When to recalibrate**:
   - Changing draft or target model
   - Significant shift in workload distribution

### Files

- `tile_aware.py`: Core algorithm
  - `Calibration`: Score → probability mapping (EAGLE/STANDALONE)
  - `PiecewiseLinearLatency`: Boundary detection + linear regression (all)
  - `compute_optimal_k()`: Finds optimal draft token count

- `profile_tile_aware.py`: Offline profiling script
  - Measures verification latency at various token counts
  - Collects calibration data
  - Saves models as `.npz` files

### Configuration

```bash
--speculative-tile-aware              # Enable feature
--speculative-calibration-path PATH   # Path to calibration.npz (EAGLE/STANDALONE)
--speculative-latency-path PATH       # Path to latency_model.npz (all algorithms)
```

### Integration Points

**EAGLE/STANDALONE** (`eagle_worker.py`, `standalone_worker.py`):
```python
# In draft_forward(), after generating draft scores:
if self.enable_tile_aware:
    num_draft_tokens = compute_optimal_k(
        scores_cat,
        self.calibration,
        self.latency_model,
        prefill_tokens=0,
        max_k=self.speculative_num_draft_tokens,
    )
```

**NGRAM** (`ngram_worker.py`):
```python
# At init, snap draft_token_num to tile boundary:
if self.enable_tile_aware and self.latency_model:
    boundaries = self.latency_model.get_boundaries()
    valid = [b for b in boundaries if b <= self.draft_token_num]
    if valid:
        self.draft_token_num = max(valid)
```

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

EAGLE scores are **cumulative path scores** (log probabilities along draft path). They represent confidence that the entire path up to that token is correct.

## Debugging Tips

- Check `batch.spec_info` for current speculative state
- Use `torch.cuda.synchronize()` before timing measurements
- Latency profiling should use median of multiple runs after warmup
