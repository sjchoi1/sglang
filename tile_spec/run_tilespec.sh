#!/bin/bash
# Run sglang with tile-spec enabled (chain-based for profiling)

export SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1
export CUDA_LAUNCH_BLOCKING=1

sglang serve --model-path meta-llama/Llama-3.1-8B-Instruct \
    --speculative-algorithm EAGLE3 \
    --speculative-draft-model-path lmsys/sglang-EAGLE3-LLaMA3.1-Instruct-8B \
    --speculative-num-steps 4 \
    --speculative-eagle-topk 1 \
    --speculative-num-draft-tokens 4 \
    --dtype float16 \
    --tile-spec \
    --disable-cuda-graph
