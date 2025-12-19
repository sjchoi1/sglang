#!/bin/bash
# Run sglang with tile-spec for profiling
# Uses K=8 for efficient profiling with varied batch sizes

export SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1

echo "Starting TileSpec profiling with K=8..."
echo "This will sweep batch sizes [1,2,4,8,16,32,64] for ~5 minutes"
echo "Token counts explored: [8, 16, 32, 64, 128, 256, 512]"
echo "Visualizations will be saved to: tile_spec/cache/{model}/plots/"
echo ""

sglang serve --model-path meta-llama/Llama-3.1-8B-Instruct \
    --speculative-algorithm EAGLE3 \
    --speculative-draft-model-path lmsys/sglang-EAGLE3-LLaMA3.1-Instruct-8B \
    --speculative-num-steps 5 \
    --speculative-eagle-topk 8 \
    --speculative-num-draft-tokens 8 \
    --dtype float16 \
    --tile-spec
