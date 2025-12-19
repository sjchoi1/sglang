# TileSpec Profiling Guide

## Quick Start

```bash
cd /home/user/sglang/tile_spec
./run_profiling.sh
```

## What Happens

### 1. Profiling Coverage
- **Batch sizes**: [1, 2, 4, 8, 16, 32, 64]
- **K value**: 8 (speculative-num-draft-tokens)
- **Iterations**: 3 per batch size = 21 total warmup runs
- **Token counts explored**: [8, 16, 32, 64, 128, 256, 512]
- **Duration**: ~2-5 minutes
- **Min samples**: 70 with at least 6 unique token counts

### 2. Collected Data
- **Latency data**: (num_tokens, latency_ms) pairs
- **Calibration data**: (cumulative_score, accepted) pairs

### 3. Model Fitting
- **Latency model**: Piecewise linear with automatic tile boundary detection
  - Jump threshold: 15% latency increase
- **Calibration**: Linear regression mapping scores → P(accept)

### 4. Cache Output
Location: `tile_spec/cache/{model}_{gpu}_tp{N}/`

Files:
- `latency_model.npz` - Piecewise latency model
- `calibration.npz` - Score calibration model
- `plots/latency_model.png` - Latency visualization
- `plots/calibration.png` - Calibration curve
- `plots/token_distribution.png` - Sample coverage

## Visualizations

### 1. Latency Model (`latency_model.png`)
- **Raw samples**: Gray scatter points (all measurements)
- **Median latency**: Blue line with markers
- **Fitted model**: Dashed line (piecewise linear interpolation)
- **Tile boundaries**: Red vertical dotted lines

### 2. Calibration Curve (`calibration.png`)
- **Empirical data**: Scatter points (binned acceptance rates)
- **Fitted model**: Blue line (linear regression)
- Shows how draft token confidence scores map to acceptance probability

### 3. Token Distribution (`token_distribution.png`)
- Bar chart showing number of samples collected per token count
- Helps verify comprehensive coverage across the space

## Next Steps

After profiling completes:
1. Check the plots in `tile_spec/cache/{model}/plots/`
2. Verify boundaries detected make sense for your GPU
3. Run production workload with optimized per-request K selection

## Configuration

To adjust profiling parameters, edit `profiler.py`:
- `WARMUP_BATCH_SIZES`: List of batch sizes to sweep
- `min_samples`: Minimum total samples before auto-finish (default: 70)
- `min_unique_counts`: Minimum unique token counts (default: 6)
- `iterations_per_size`: Warmup runs per batch size (default: 3)

## Troubleshooting

**Issue**: Profiling doesn't start
- Check logs for "TileSpec: Already profiled, skipping warmup"
- Delete cache directory to force re-profiling

**Issue**: Not enough samples
- Increase `iterations_per_size` in profiler.py
- Add more batch sizes to `WARMUP_BATCH_SIZES`

**Issue**: Visualizations not generated
- Install matplotlib: `pip install matplotlib`
- Check logs for warnings
